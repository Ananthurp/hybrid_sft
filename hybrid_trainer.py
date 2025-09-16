import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer import is_torch_xla_available, SaveStrategy
from entmax import sparsemax, sparsemax_loss


class HybridSFTTrainer(Trainer):
    """
    SFT Trainer for:
      - hybrid:   Fenchel–Young(sparsemax) + ns_alpha * Negative Sampling (NS)
      - sparsemax: Fenchel–Young(sparsemax) only
      - ns_only:  ns_alpha * Negative Sampling only

    NS modes:
      - top_k: suppress everything outside Top-K (by softmax prob), excluding GT label.
      - bottom_p: suppress the bottom p% *by count* (NOT nucleus), excluding GT label.
      - support_set: suppress everything outside sparsemax support, excluding GT label.

    Notes:
      * Temperature (ns_temperature) applies to the sparse path (sparsemax & FY loss).
      * Keep grads through softmax_probs for the NS term (no no_grad on softmax).
      * Clamp inside log to avoid -inf.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorded = {}  # per-step diagnostics

    # ---------- helpers ----------

    @staticmethod
    def _empty_loss_like(logits):
        # Return a scalar zero that keeps the graph intact if a microbatch has no valid tokens
        return torch.zeros((), device=logits.device, dtype=logits.dtype, requires_grad=True)

    @torch.no_grad()
    def compute_training_logs(self, logits, labels):
        """
        Optional diagnostics:
          - entropy over valid (shifted) tokens
        """
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        mask = shift_labels != -100
        shift_logits = shift_logits[mask]  # [N, V]

        logs = {}
        if getattr(self.args, "print_entropy", False) and shift_logits.numel() > 0:
            entropy = chunked_entropy_from_logits(
                shift_logits,
                batch_size=max(1, shift_logits.size(0) // 4),
            ).mean()
            logs["entropy"] = round(float(entropy.item()), 4)
        return logs

    # ---------- loss variants (each can serve hybrid/sparsemax/ns_only) ----------

    def hybrid_loss_support_set(self, logits, labels, num_items_in_batch=None):
        """
        Negative set = all tokens OUTSIDE sparsemax support (excluding the ground-truth label).
        """
        mode = getattr(self.args, "loss", "hybrid")
        T = float(getattr(self.args, "ns_temperature", 1.0))

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        mask = shift_labels != -100
        shift_logits, shift_labels = shift_logits[mask], shift_labels[mask]

        if shift_logits.numel() == 0:
            return self._empty_loss_like(logits)

        s_loss = None
        ns_loss = None

        # 1) Fenchel–Young loss (sparsemax) with temperature on the sparse path
        if mode != "ns_only":
            s_loss = sparsemax_loss(shift_logits * T, shift_labels)

        # 2) Negative sampling: penalize softmax mass outside sparsemax support
        if mode != "sparsemax":
            softmax_probs = F.softmax(shift_logits, dim=-1)  # keep grads
            with torch.no_grad():
                sparsemax_probs = sparsemax(shift_logits * T, dim=-1)
                support_set_mask = (sparsemax_probs > 0)
                one_hot = F.one_hot(shift_labels, num_classes=shift_logits.size(-1)).bool()
                neg_mask = ~support_set_mask & ~one_hot
                support_size = support_set_mask.sum(dim=-1).float()  # [N]

            suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
            ns_safe = torch.clamp(1.0 - suppressed_mass, min=1e-6)
            ns_loss = -torch.log(ns_safe)

        # combine according to mode
        if mode == "sparsemax":
            loss = s_loss
        elif mode == "ns_only":
            # scale by ns_alpha to preserve your CLI semantics
            loss = self.args.ns_alpha * ns_loss
        else:  # hybrid
            loss = s_loss + self.args.ns_alpha * ns_loss

        # logging means before reduction
        with torch.no_grad():
            if mode != "sparsemax":
                self.recorded["suppressed_mass"] = float(suppressed_mass.mean().item())
                self.recorded["ns_loss"] = float(ns_loss.mean().item())
                self.recorded["support_size"] = float(support_size.mean().item())
            if mode != "ns_only":
                self.recorded["s_loss"] = float(s_loss.mean().item())

        # reduce
        if num_items_in_batch is not None:
            loss = loss.sum() / num_items_in_batch
        else:
            loss = loss.mean()
        return loss

    def hybrid_loss_topk(self, logits, labels, num_items_in_batch=None):
        """
        Negative set = everything OUTSIDE Top-K (by softmax prob), excluding the ground-truth label.
        """
        mode = getattr(self.args, "loss", "hybrid")
        T = float(getattr(self.args, "ns_temperature", 1.0))
        K = int(getattr(self.args, "ns_top_k", 10))

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        mask = shift_labels != -100
        shift_logits, shift_labels = shift_logits[mask], shift_labels[mask]

        if shift_logits.numel() == 0:
            return self._empty_loss_like(logits)

        s_loss = None
        ns_loss = None

        if mode != "ns_only":
            s_loss = sparsemax_loss(shift_logits * T, shift_labels)

        if mode != "sparsemax":
            softmax_probs = F.softmax(shift_logits, dim=-1)  # keep grads
            with torch.no_grad():
                k = max(1, min(softmax_probs.size(-1) - 1, K))
                _, topk_idx = torch.topk(softmax_probs, k=k, dim=-1)
                topk_mask = torch.zeros_like(softmax_probs, dtype=torch.bool).scatter_(1, topk_idx, True)
                one_hot = F.one_hot(shift_labels, num_classes=softmax_probs.size(-1)).bool()
                neg_mask = ~topk_mask & ~one_hot

            suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
            ns_safe = torch.clamp(1.0 - suppressed_mass, min=1e-6)
            ns_loss = -torch.log(ns_safe)

        if mode == "sparsemax":
            loss = s_loss
        elif mode == "ns_only":
            loss = self.args.ns_alpha * ns_loss
        else:
            loss = s_loss + self.args.ns_alpha * ns_loss

        with torch.no_grad():
            if mode != "sparsemax":
                self.recorded["suppressed_mass"] = float(suppressed_mass.mean().item())
                self.recorded["ns_loss"] = float(ns_loss.mean().item())
            if mode != "ns_only":
                self.recorded["s_loss"] = float(s_loss.mean().item())

        if num_items_in_batch is not None:
            loss = loss.sum() / num_items_in_batch
        else:
            loss = loss.mean()
        return loss

    def hybrid_loss_bottom_p(self, logits, labels, num_items_in_batch=None):
        """
        Negative set = bottom p% *by count* (lowest-prob tokens), excluding the ground-truth label.
        NOTE: This is NOT nucleus/top-p by cumulative mass.
        """
        mode = getattr(self.args, "loss", "hybrid")
        T = float(getattr(self.args, "ns_temperature", 1.0))
        p = float(getattr(self.args, "ns_bottom_p", 0.9))

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        mask = shift_labels != -100
        shift_logits, shift_labels = shift_logits[mask], shift_labels[mask]

        if shift_logits.numel() == 0:
            return self._empty_loss_like(logits)

        s_loss = None
        ns_loss = None

        if mode != "ns_only":
            s_loss = sparsemax_loss(shift_logits * T, shift_labels)

        if mode != "sparsemax":
            softmax_probs = F.softmax(shift_logits, dim=-1)  # keep grads
            with torch.no_grad():
                V = softmax_probs.size(-1)
                k = max(1, min(V - 1, int(round(V * p))))  # [1, V-1]
                _, bottom_idx = torch.topk(softmax_probs, k=k, dim=-1, largest=False)
                bottom_mask = torch.zeros_like(softmax_probs, dtype=torch.bool).scatter_(1, bottom_idx, True)
                one_hot = F.one_hot(shift_labels, num_classes=V).bool()
                neg_mask = bottom_mask & ~one_hot

            suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
            ns_safe = torch.clamp(1.0 - suppressed_mass, min=1e-6)
            ns_loss = -torch.log(ns_safe)

        if mode == "sparsemax":
            loss = s_loss
        elif mode == "ns_only":
            loss = self.args.ns_alpha * ns_loss
        else:
            loss = s_loss + self.args.ns_alpha * ns_loss

        with torch.no_grad():
            if mode != "sparsemax":
                self.recorded["suppressed_mass"] = float(suppressed_mass.mean().item())
                self.recorded["ns_loss"] = float(ns_loss.mean().item())
            if mode != "ns_only":
                self.recorded["s_loss"] = float(s_loss.mean().item())

        if num_items_in_batch is not None:
            loss = loss.sum() / num_items_in_batch
        else:
            loss = loss.mean()
        return loss

    # ---------- Trainer overrides ----------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Keep labels for BOTH: (a) model CE loss (for logging), (b) our custom loss
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)  # logits + standard CE loss
        logits = outputs.get("logits")

        # Route to chosen NS strategy (still used even if mode=sparsemax/ns_only)
        if self.args.ns_type == "top_k":
            loss = self.hybrid_loss_topk(logits, labels, num_items_in_batch)
        elif self.args.ns_type == "bottom_p":
            loss = self.hybrid_loss_bottom_p(logits, labels, num_items_in_batch)
        elif self.args.ns_type == "support_set":
            loss = self.hybrid_loss_support_set(logits, labels, num_items_in_batch)
        else:
            raise ValueError(f"Unsupported ns_type: {self.args.ns_type}")

        # Build logs
        self.training_logs = self.compute_training_logs(logits, labels)
        ce = outputs.get("loss", None)
        if ce is not None:
            self.training_logs["ce_loss"] = round(float(ce.item()), 4)
        if "suppressed_mass" in self.recorded:
            self.training_logs["suppressed_mass"] = round(self.recorded["suppressed_mass"], 4)
        if "s_loss" in self.recorded:
            self.training_logs["s_loss"] = round(self.recorded["s_loss"], 4)
        if "ns_loss" in self.recorded:
            self.training_logs["ns_loss"] = round(self.recorded["ns_loss"], 4)
        if "support_size" in self.recorded:
            self.training_logs["support_size"] = round(self.recorded["support_size"], 2)

        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
        learning_rate=None,
    ):
        # log
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                import torch_xla.core.xla_model as xm
                xm.mark_step()

            logs = {}
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_loss -= tr_loss  # reset

            logs["loss"] = round(
                tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4
            )
            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                )
            logs["learning_rate"] = self._get_learning_rate()
            if getattr(self, "training_logs", None):
                logs.update(self.training_logs)

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            try:
                self.log(logs, start_time)
            except TypeError:
                self.log(logs)

        # evaluate: honor HF control OR fallback to eval every eval_steps if provided
        metrics = None
        should_eval = self.control.should_evaluate
        if not should_eval:
            es = getattr(self.args, "eval_steps", None)
            if es and self.state.global_step > 0 and (self.state.global_step % es == 0) and self.eval_dataset is not None:
                should_eval = True

        if should_eval:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            try:
                self.log(metrics, start_time)
            except TypeError:
                self.log(metrics)
            try:
                self.save_metrics("eval", metrics)
            except Exception:
                pass

            if self.args.save_strategy == SaveStrategy.BEST:
                is_new = self._determine_best_metric(metrics=metrics, trial=trial)
                self.control.should_save = is_new

        # save
        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


# ---------- utility ----------

def chunked_entropy_from_logits(chunk_logits, batch_size=None):
    """
    Memory-frugal entropy from logits.
    chunk_logits: [N, V]
    returns: [N]
    """
    total_samples, _ = chunk_logits.shape
    entropy_list = []
    if batch_size is None:
        batch_size = total_samples

    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        lb = chunk_logits[start:end]                 # [B, V]
        logsumexp = torch.logsumexp(lb, dim=-1)      # [B]
        norm_logits = lb - logsumexp.unsqueeze(-1)   # [B, V]
        probs = torch.exp(norm_logits)               # [B, V]
        entropy = logsumexp - (lb * probs).sum(-1)   # [B]
        entropy_list.append(entropy)

    if len(entropy_list) > 0:
        return torch.cat(entropy_list, dim=0)
    else:
        return torch.tensor(0.0, device=chunk_logits.device)
