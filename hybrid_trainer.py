# import torch
# import torch.nn.functional as F
# from transformers import Trainer
# from entmax import sparsemax # type: ignore # We will use the entmax library for sparsemax

# class HybridSFTTrainer(Trainer):
#     """
#     A custom trainer for the hybrid loss function combining Fenchel-Young (with sparsemax)
#     and Negative Sampling.
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # Extract custom hyperparameters from the TrainingArguments
#         self.alpha = self.args.ns_alpha
#         self.tau_ns = self.args.ns_tau

#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         # Get the ground-truth labels and remove them from inputs
#         labels = inputs.pop("labels")

#         # Get the raw logits from the model
#         outputs = model(**inputs)
#         logits = outputs.get("logits")

#         # --- 1. Calculate Fenchel-Young (Sparsemax) Loss ---
#         vocab_size = logits.shape[-1]
#         labels_for_one_hot = labels.clamp(0, vocab_size - 1)
#         y_one_hot = F.one_hot(labels_for_one_hot, num_classes=vocab_size).float()
        
#         loss_mask = (labels != -100).float()

#         l2_dist_y_z = torch.sum(((y_one_hot - logits) ** 2), dim=-1)
#         sparse_probs = sparsemax(logits, dim=-1)
#         l2_dist_sparsemax_z = torch.sum(((sparse_probs - logits) ** 2), dim=-1)

#         fy_loss_per_token = 0.5 * (l2_dist_y_z - l2_dist_sparsemax_z)
#         masked_fy_loss = fy_loss_per_token * loss_mask
#         fy_loss = masked_fy_loss.sum() / loss_mask.sum()

#         # --- 2. Calculate Negative Sampling Loss ---
#         # We use standard softmax here to get the probabilities for NS loss
#         probs = F.softmax(logits, dim=-1)
        
#         # Create a mask for probabilities below the threshold tau_ns
#         indicator = (probs < self.tau_ns).float()
        
#         # Calculate the sum of probabilities of negative samples
#         sum_neg_probs = torch.sum(indicator * probs, dim=-1)
        
#         # Calculate the NS loss for each token
#         # Add a small epsilon to prevent log(0)
#         ns_loss_per_token = -torch.log(1 - sum_neg_probs + 1e-9)

#         masked_ns_loss = ns_loss_per_token * loss_mask
#         ns_loss = masked_ns_loss.sum() / loss_mask.sum()

#         # --- 3. Combine the losses ---
#         total_loss = fy_loss + self.alpha * ns_loss

#         return (total_loss, outputs) if return_outputs else total_loss


# import torch
# import torch.nn.functional as F
# from transformers import Trainer
# from entmax import sparsemax

# class HybridSFTTrainer(Trainer):
#     """
#     A custom trainer for the hybrid loss function combining Fenchel-Young (with sparsemax)
#     and Negative Sampling.
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.alpha = self.args.ns_alpha
#         self.tau_ns = self.args.ns_tau

#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs.get("logits")

#         # --- 1. Calculate Fenchel-Young (Sparsemax) Loss ---
#         vocab_size = logits.shape[-1]
#         labels_for_one_hot = labels.clamp(0, vocab_size - 1)
#         y_one_hot = F.one_hot(labels_for_one_hot, num_classes=vocab_size).float()
        
#         loss_mask = (labels != -100).float()

#         l2_dist_y_z = torch.sum(((y_one_hot - logits) ** 2), dim=-1)
#         sparse_probs = sparsemax(logits, dim=-1)
#         l2_dist_sparsemax_z = torch.sum(((sparse_probs - logits) ** 2), dim=-1)

#         fy_loss_per_token = 0.5 * (l2_dist_y_z - l2_dist_sparsemax_z)
#         masked_fy_loss = fy_loss_per_token * loss_mask
#         fy_loss = masked_fy_loss.sum() / loss_mask.sum()

#         # --- 2. Calculate Negative Sampling Loss ---
#         probs = F.softmax(logits, dim=-1)
#         indicator = (probs < self.tau_ns).float()
#         sum_neg_probs = torch.sum(indicator * probs, dim=-1)
        
#         # --- THIS IS THE CORRECTED, MORE STABLE LINE ---
#         ns_loss_per_token = -torch.log(torch.clamp(1 - sum_neg_probs, min=1e-9))

#         masked_ns_loss = ns_loss_per_token * loss_mask
#         ns_loss = masked_ns_loss.sum() / loss_mask.sum()

#         # --- 3. Combine the losses ---
#         total_loss = fy_loss + self.alpha * ns_loss

#         return (total_loss, outputs) if return_outputs else total_loss








# import torch
# import torch.nn.functional as F
# from transformers import Trainer
# from entmax import sparsemax, sparsemax_loss
# import torch.distributed as dist

# class HybridSFTTrainer(Trainer):
#     """
#     An advanced SFT Trainer for a hybrid loss function, adapted from the reference repository.
#     Combines Fenchel-Young (sparsemax) loss with Negative Sampling.
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.recorded = {} # For logging detailed metrics

#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs.get("logits")

#         # Route to the correct loss calculation based on the ns_type argument
#         if self.args.ns_type == "top_k":
#             loss = self.hybrid_loss_topk(logits, labels, num_items_in_batch=num_items_in_batch)
#         elif self.args.ns_type == "bottom_p":
#             loss = self.hybrid_loss_bottom_p(logits, labels, num_items_in_batch=num_items_in_batch)
#         else:
#             raise ValueError(f"Unsupported ns_type: {self.args.ns_type}")

#         return (loss, outputs) if return_outputs else loss

#     def hybrid_loss_topk(self, logits, labels, num_items_in_batch=None):
#         """Calculates hybrid loss, suppressing all tokens NOT in the top-k."""
#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         mask = shift_labels != -100
#         shift_logits, shift_labels = shift_logits[mask], shift_labels[mask]

#         s_loss = sparsemax_loss(shift_logits, shift_labels)

#         with torch.no_grad():
#             softmax_probs = F.softmax(shift_logits, dim=-1)
#             topk_values, topk_indices = torch.topk(softmax_probs, k=self.args.ns_top_k, dim=-1)
            
#             topk_mask = torch.zeros_like(softmax_probs, dtype=torch.bool).scatter_(1, topk_indices, True)
#             one_hot_labels = F.one_hot(shift_labels, num_classes=softmax_probs.size(-1)).bool()
            
#             neg_mask = ~topk_mask & ~one_hot_labels

#         suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
#         suppressed_mass = torch.clamp(suppressed_mass, max=0.999)
        
#         ns_loss = -torch.log(1.0 - suppressed_mass)
        
#         loss = s_loss + self.args.ns_alpha * ns_loss

#         if num_items_in_batch is not None:
#             loss = loss.sum() / num_items_in_batch
#         else:
#             loss = loss.mean()
            
#         return loss

#     def hybrid_loss_bottom_p(self, logits, labels, num_items_in_batch=None):
#         """Calculates hybrid loss, suppressing the bottom p% of probability mass."""
#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         mask = shift_labels != -100
#         shift_logits, shift_labels = shift_logits[mask], shift_labels[mask]

#         s_loss = sparsemax_loss(shift_logits, shift_labels)

#         with torch.no_grad():
#             softmax_probs = F.softmax(shift_logits, dim=-1)
            
#             # This logic is adapted from your colleague's sft_trainer_v2.py
#             # It uses ranking to determine the threshold for negative sampling
#             ranks = torch.argsort(torch.argsort(softmax_probs, dim=-1, descending=True), dim=-1) + 1
#             one_hot_labels = F.one_hot(shift_labels, num_classes=softmax_probs.size(-1)).bool()
            
#             # The threshold tau is now interpreted as a percentage of the vocabulary size
#             vocab_size = softmax_probs.size(-1)
#             rank_threshold = int(vocab_size * self.args.ns_bottom_p)
            
#             neg_mask = (ranks > rank_threshold) & (~one_hot_labels)

#         suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
#         suppressed_mass = torch.clamp(suppressed_mass, max=0.999)
        
#         ns_loss = -torch.log(1.0 - suppressed_mass)

#         loss = s_loss + self.args.ns_alpha * ns_loss

#         if num_items_in_batch is not None:
#             loss = loss.sum() / num_items_in_batch
#         else:
#             loss = loss.mean()

#         return loss






# import torch
# import torch.nn.functional as F
# from transformers import Trainer
# from entmax import sparsemax, sparsemax_loss
# import torch.distributed as dist

# class HybridSFTTrainer(Trainer):
#     """
#     An advanced SFT Trainer for a hybrid loss function.
#     Supports multiple modes for Negative Sampling.
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.recorded = {} # For logging detailed metrics

#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs.get("logits")

#         # Route to the correct loss calculation based on the ns_type argument
#         if self.args.ns_type == "top_k":
#             loss = self.hybrid_loss_topk(logits, labels, num_items_in_batch)
#         elif self.args.ns_type == "bottom_p":
#             loss = self.hybrid_loss_bottom_p(logits, labels, num_items_in_batch)
#         elif self.args.ns_type == "support_set":
#             loss = self.hybrid_loss_support_set(logits, labels, num_items_in_batch)
#         else:
#             raise ValueError(f"Unsupported ns_type: {self.args.ns_type}")

#         return (loss, outputs) if return_outputs else loss

#     def hybrid_loss_support_set(self, logits, labels, num_items_in_batch=None):
#         """
#         Calculates hybrid loss, suppressing all tokens OUTSIDE the sparsemax support set.
#         """
#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         mask = shift_labels != -100
#         shift_logits, shift_labels = shift_logits[mask], shift_labels[mask]

#         # 1. Fenchel-Young (Sparsemax) Loss
#         s_loss = sparsemax_loss(shift_logits, shift_labels)

#         # 2. Negative Sampling Loss (Support Set variant)
#         with torch.no_grad():
#             # First, get the sparsemax probabilities to identify the support set
#             sparsemax_probs = sparsemax(shift_logits, dim=-1)

#             # The negative mask is all tokens where sparsemax probability is zero
#             # (i.e., they are outside the support set)
#             support_set_mask = (sparsemax_probs > 0)

#             # The ground truth label should never be suppressed
#             one_hot_labels = F.one_hot(shift_labels, num_classes=shift_logits.size(-1)).bool()

#             # The final negative mask is everything outside the support set AND not the true label
#             neg_mask = ~support_set_mask & ~one_hot_labels

#             # We use standard softmax for the NS loss calculation
#             softmax_probs = F.softmax(shift_logits, dim=-1)
#             suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
#             suppressed_mass = torch.clamp(suppressed_mass, max=0.999)

#         ns_loss = -torch.log(1.0 - suppressed_mass)

#         # 3. Combine losses
#         loss = s_loss + self.args.ns_alpha * ns_loss

#         if num_items_in_batch is not None:
#             loss = loss.sum() / num_items_in_batch
#         else:
#             loss = loss.mean()

#         return loss

#     def hybrid_loss_topk(self, logits, labels, num_items_in_batch=None):
#         # ... (this function remains unchanged)
#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         mask = shift_labels != -100
#         shift_logits, shift_labels = shift_logits[mask], shift_labels[mask]
#         s_loss = sparsemax_loss(shift_logits, shift_labels)
#         with torch.no_grad():
#             softmax_probs = F.softmax(shift_logits, dim=-1)
#             _, topk_indices = torch.topk(softmax_probs, k=self.args.ns_top_k, dim=-1)
#             topk_mask = torch.zeros_like(softmax_probs, dtype=torch.bool).scatter_(1, topk_indices, True)
#             one_hot_labels = F.one_hot(shift_labels, num_classes=softmax_probs.size(-1)).bool()
#             neg_mask = ~topk_mask & ~one_hot_labels
#         suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
#         suppressed_mass = torch.clamp(suppressed_mass, max=0.999)
#         ns_loss = -torch.log(1.0 - suppressed_mass)
#         loss = s_loss + self.args.ns_alpha * ns_loss
#         if num_items_in_batch is not None:
#             loss = loss.sum() / num_items_in_batch
#         else:
#             loss = loss.mean()
#         return loss

#     def hybrid_loss_bottom_p(self, logits, labels, num_items_in_batch=None):
#         """
#         Calculates hybrid loss, suppressing the bottom p% of tokens by count.
#         """
#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         mask = shift_labels != -100
#         shift_logits, shift_labels = shift_logits[mask], shift_labels[mask]

#         # 1. Fenchel-Young (Sparsemax) Loss
#         s_loss = sparsemax_loss(shift_logits, shift_labels)

#         # 2. Negative Sampling Loss (Bottom-P by count variant)
#         with torch.no_grad():
#             softmax_probs = F.softmax(shift_logits, dim=-1)
#             vocab_size = softmax_probs.size(-1)
            
#             # Calculate the number of tokens to suppress
#             num_to_suppress = int(vocab_size * self.args.ns_bottom_p)
            
#             # Find the indices of the tokens with the lowest probabilities
#             _, bottom_indices = torch.topk(softmax_probs, k=num_to_suppress, dim=-1, largest=False)

#             # Create a mask that is True for all bottom tokens
#             bottom_mask = torch.zeros_like(softmax_probs, dtype=torch.bool).scatter_(1, bottom_indices, True)
            
#             # The ground truth label should never be suppressed
#             one_hot_labels = F.one_hot(shift_labels, num_classes=vocab_size).bool()
            
#             # The final negative mask is the bottom tokens, excluding the true label
#             neg_mask = bottom_mask & ~one_hot_labels

#         suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
#         suppressed_mass = torch.clamp(suppressed_mass, max=0.999)
        
#         ns_loss = -torch.log(1.0 - suppressed_mass)

#         loss = s_loss + self.args.ns_alpha * ns_loss

#         if num_items_in_batch is not None:
#             loss = loss.sum() / num_items_in_batch
#         else:
#             loss = loss.mean()

#         return loss




# import os
# import torch
# import torch.nn.functional as F
# from transformers import Trainer
# from transformers.trainer import is_torch_xla_available, SaveStrategy
# from entmax import sparsemax, sparsemax_loss
# import torch.distributed as dist

# class HybridSFTTrainer(Trainer):
#     """
#     SFT Trainer for the hybrid loss: Fenchel–Young(sparsemax) + alpha * Negative Sampling.
#     Negative sets: top_k, bottom_p (by COUNT), or outside sparsemax support.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.recorded = {}  # for extra logs per step

#     # ---------- Logging helpers ----------

#     @torch.no_grad()
#     def compute_training_logs(self, logits, labels):
#         """
#         Compute optional training diagnostics:
#           - entropy over valid (shifted) tokens
#           - CE loss from model forward (added in compute_loss)
#           - optional suppressed_mass (added in compute_loss)
#         """
#         # shift + mask
#         shift_logits = logits[..., :-1, :]
#         shift_labels = labels[..., 1:]
#         mask = shift_labels != -100
#         shift_logits = shift_logits[mask]  # [N, V]

#         logs = {}
#         if getattr(self.args, "print_entropy", False) and shift_logits.numel() > 0:
#             entropy = chunked_entropy_from_logits(
#                 shift_logits,
#                 batch_size=max(1, shift_logits.size(0) // 4),
#             ).mean()
#             logs["entropy"] = round(float(entropy.item()), 4)
#         return logs

#     # ---------- Loss variants ----------

#     def hybrid_loss_support_set(self, logits, labels, num_items_in_batch=None):
#         """
#         Suppress all tokens OUTSIDE sparsemax support (excluding the ground-truth label).
#         Temperature is applied to the sparsemax path (like your colleague).
#         """
#         T = float(getattr(self.args, "ns_temperature", 1.0))

#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         mask = shift_labels != -100
#         shift_logits, shift_labels = shift_logits[mask], shift_labels[mask]

#         # 1) Fenchel–Young (sparsemax) loss with temperature on the sparse path
#         s_loss = sparsemax_loss(shift_logits * T, shift_labels)

#         # 2) Negative sampling penalty: use SOFTMAX mass outside support
#         softmax_probs = F.softmax(shift_logits, dim=-1)  # keep gradients!
#         with torch.no_grad():
#             sparsemax_probs = sparsemax(shift_logits * T, dim=-1)
#             support_set_mask = (sparsemax_probs > 0)  # positives
#             one_hot_labels = F.one_hot(shift_labels, num_classes=shift_logits.size(-1)).bool()
#             neg_mask = ~support_set_mask & ~one_hot_labels  # negatives
#         suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
#         ns_loss = -torch.log(torch.clamp(1.0 - suppressed_mass, min=1e-6))

#         loss = s_loss + self.args.ns_alpha * ns_loss
#         if num_items_in_batch is not None:
#             loss = loss.sum() / num_items_in_batch
#         else:
#             loss = loss.mean()

#         # record for logs
#         with torch.no_grad():
#             self.recorded["suppressed_mass"] = float(suppressed_mass.mean().item())
#         return loss

#     def hybrid_loss_topk(self, logits, labels, num_items_in_batch=None):
#         """
#         Suppress everything OUTSIDE the Top-K (by softmax prob), excluding the ground-truth label.
#         """
#         T = float(getattr(self.args, "ns_temperature", 1.0))
#         K = int(getattr(self.args, "ns_top_k", 10))

#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         mask = shift_labels != -100
#         shift_logits, shift_labels = shift_logits[mask], shift_labels[mask]

#         s_loss = sparsemax_loss(shift_logits * T, shift_labels)

#         softmax_probs = F.softmax(shift_logits, dim=-1)  # keep grads
#         with torch.no_grad():
#             _, topk_indices = torch.topk(softmax_probs, k=min(K, softmax_probs.size(-1)), dim=-1)
#             topk_mask = torch.zeros_like(softmax_probs, dtype=torch.bool).scatter_(1, topk_indices, True)
#             one_hot_labels = F.one_hot(shift_labels, num_classes=softmax_probs.size(-1)).bool()
#             neg_mask = ~topk_mask & ~one_hot_labels
#         suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
#         ns_loss = -torch.log(torch.clamp(1.0 - suppressed_mass, min=1e-6))

#         loss = s_loss + self.args.ns_alpha * ns_loss
#         if num_items_in_batch is not None:
#             loss = loss.sum() / num_items_in_batch
#         else:
#             loss = loss.mean()

#         with torch.no_grad():
#             self.recorded["suppressed_mass"] = float(suppressed_mass.mean().item())
#         return loss

#     def hybrid_loss_bottom_p(self, logits, labels, num_items_in_batch=None):
#         """
#         Suppress the bottom p% by COUNT (lowest-prob tokens), excluding the ground-truth label.
#         NOTE: This is NOT nucleus/top-p; it's by *count* as requested.
#         """
#         T = float(getattr(self.args, "ns_temperature", 1.0))
#         p = float(getattr(self.args, "ns_bottom_p", 0.9))

#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         mask = shift_labels != -100
#         shift_logits, shift_labels = shift_logits[mask], shift_labels[mask]

#         s_loss = sparsemax_loss(shift_logits * T, shift_labels)

#         softmax_probs = F.softmax(shift_logits, dim=-1)  # keep grads
#         with torch.no_grad():
#             V = softmax_probs.size(-1)
#             k = max(1, min(V - 1, int(round(V * p))))  # clamp to [1, V-1]
#             _, bottom_idx = torch.topk(softmax_probs, k=k, dim=-1, largest=False)
#             bottom_mask = torch.zeros_like(softmax_probs, dtype=torch.bool).scatter_(1, bottom_idx, True)
#             one_hot_labels = F.one_hot(shift_labels, num_classes=V).bool()
#             neg_mask = bottom_mask & ~one_hot_labels
#         suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
#         ns_loss = -torch.log(torch.clamp(1.0 - suppressed_mass, min=1e-6))

#         loss = s_loss + self.args.ns_alpha * ns_loss
#         if num_items_in_batch is not None:
#             loss = loss.sum() / num_items_in_batch
#         else:
#             loss = loss.mean()

#         with torch.no_grad():
#             self.recorded["suppressed_mass"] = float(suppressed_mass.mean().item())
#         return loss

#     # ---------- Trainer overrides ----------

#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         # Keep labels for BOTH: (a) model CE loss (for logging), (b) our custom hybrid loss
#         labels = inputs.pop("labels")
#         outputs = model(**inputs, labels=labels)  # gives logits + ce loss
#         logits = outputs.get("logits")

#         # Route to the selected negative sampling type
#         if self.args.ns_type == "top_k":
#             loss = self.hybrid_loss_topk(logits, labels, num_items_in_batch)
#         elif self.args.ns_type == "bottom_p":
#             loss = self.hybrid_loss_bottom_p(logits, labels, num_items_in_batch)
#         elif self.args.ns_type == "support_set":
#             loss = self.hybrid_loss_support_set(logits, labels, num_items_in_batch)
#         else:
#             raise ValueError(f"Unsupported ns_type: {self.args.ns_type}")

#         # Prepare logs for _maybe_log_save_evaluate
#         self.training_logs = self.compute_training_logs(logits, labels)
#         ce = outputs.get("loss", None)
#         if ce is not None:
#             self.training_logs["ce_loss"] = round(float(ce.item()), 4)
#         if "suppressed_mass" in self.recorded:
#             self.training_logs["suppressed_mass"] = round(self.recorded["suppressed_mass"], 4)

#         return (loss, outputs) if return_outputs else loss

#     def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None):
#         # Copied (lightly) from your colleague so we can inject self.training_logs into the step logs
#         if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
#             if is_torch_xla_available():
#                 import torch_xla.core.xla_model as xm
#                 xm.mark_step()

#             logs = {}
#             tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
#             tr_loss -= tr_loss  # reset

#             logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
#             if grad_norm is not None:
#                 logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
#             logs["learning_rate"] = self._get_learning_rate()
#             if getattr(self, "training_logs", None):
#                 logs.update(self.training_logs)

#             self._total_loss_scalar += tr_loss_scalar
#             self._globalstep_last_logged = self.state.global_step
#             self.store_flos()
#             self.log(logs, start_time)

#         metrics = None
#         if self.control.should_evaluate:
#             metrics = self._evaluate(trial, ignore_keys_for_eval)
#             is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)
#             if self.args.save_strategy == SaveStrategy.BEST:
#                 self.control.should_save = is_new_best_metric

#         if self.control.should_save:
#             self._save_checkpoint(model, trial)
#             self.control = self.callback_handler.on_save(self.args, self.state, self.control)


# # ---------- Utility (copied from colleague) ----------

# def chunked_entropy_from_logits(chunk_logits, batch_size=None):
#     """
#     Memory-frugal entropy from logits.
#     chunk_logits: [N, V]
#     returns: [N]
#     """
#     total_samples, num_classes = chunk_logits.shape
#     entropy_list = []
#     if batch_size is None:
#         batch_size = total_samples

#     for start in range(0, total_samples, batch_size):
#         end = min(start + batch_size, total_samples)
#         lb = chunk_logits[start:end]  # [B, V]
#         logsumexp = torch.logsumexp(lb, dim=-1)         # [B]
#         norm_logits = lb - logsumexp.unsqueeze(-1)      # [B, V]
#         probs = torch.exp(norm_logits)                  # [B, V]
#         entropy = logsumexp - (lb * probs).sum(-1)      # [B]
#         entropy_list.append(entropy)

#     if len(entropy_list) > 0:
#         return torch.cat(entropy_list, dim=0)
#     else:
#         return torch.tensor(0.0, device=chunk_logits.device)



# import torch
# import torch.nn.functional as F
# from transformers import Trainer
# from transformers.trainer import is_torch_xla_available, SaveStrategy
# from entmax import sparsemax, sparsemax_loss


# class HybridSFTTrainer(Trainer):
#     """
#     SFT Trainer for the hybrid loss:
#         Fenchel–Young(sparsemax) + ns_alpha * Negative Sampling (NS).

#     NS modes:
#       - top_k: suppress everything outside top-K (by softmax prob), excluding GT label.
#       - bottom_p: suppress the bottom p% *by count* (NOT nucleus), excluding GT label.
#       - support_set: suppress everything outside sparsemax support, excluding GT label.

#     Notes:
#       * Temperature (ns_temperature) is applied to the sparse path (sparsemax & FY loss),
#         matching your colleague’s implementation.
#       * We keep gradients through softmax_probs for the NS term (do NOT wrap in no_grad).
#       * We clamp inside log to avoid -inf.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.recorded = {}  # per-step diagnostics

#     # ---------- helpers ----------

#     @staticmethod
#     def _empty_loss_like(logits):
#         # Return a scalar zero that keeps the graph intact if a microbatch has no valid tokens
#         return torch.zeros((), device=logits.device, dtype=logits.dtype, requires_grad=True)

#     @torch.no_grad()
#     def compute_training_logs(self, logits, labels):
#         """
#         Optional diagnostics:
#           - entropy over valid (shifted) tokens
#         """
#         shift_logits = logits[..., :-1, :]
#         shift_labels = labels[..., 1:]
#         mask = shift_labels != -100
#         shift_logits = shift_logits[mask]  # [N, V]

#         logs = {}
#         if getattr(self.args, "print_entropy", False) and shift_logits.numel() > 0:
#             entropy = chunked_entropy_from_logits(
#                 shift_logits,
#                 batch_size=max(1, shift_logits.size(0) // 4),
#             ).mean()
#             logs["entropy"] = round(float(entropy.item()), 4)
#         return logs

#     # ---------- loss variants ----------

#     def hybrid_loss_support_set(self, logits, labels, num_items_in_batch=None):
#         """
#         Suppress all tokens OUTSIDE sparsemax support (excluding the ground-truth label).
#         """
#         T = float(getattr(self.args, "ns_temperature", 1.0))

#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         mask = shift_labels != -100
#         shift_logits, shift_labels = shift_logits[mask], shift_labels[mask]

#         if shift_logits.numel() == 0:
#             return self._empty_loss_like(logits)

#         # 1) Fenchel–Young loss (sparsemax) with temperature on the sparse path
#         s_loss = sparsemax_loss(shift_logits * T, shift_labels)

#         # 2) Negative sampling: penalize softmax mass outside sparsemax support
#         softmax_probs = F.softmax(shift_logits, dim=-1)  # keep grads
#         with torch.no_grad():
#             sparsemax_probs = sparsemax(shift_logits * T, dim=-1)
#             support_set_mask = (sparsemax_probs > 0)
#             one_hot = F.one_hot(shift_labels, num_classes=shift_logits.size(-1)).bool()
#             neg_mask = ~support_set_mask & ~one_hot

#         suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
#         ns_safe = torch.clamp(1.0 - suppressed_mass, min=1e-6)
#         ns_loss = -torch.log(ns_safe)

#         # combine
#         loss = s_loss + self.args.ns_alpha * ns_loss

#         # logging means before reduction
#         with torch.no_grad():
#             self.recorded["suppressed_mass"] = float(suppressed_mass.mean().item())
#             self.recorded["s_loss"] = float(s_loss.mean().item())
#             self.recorded["ns_loss"] = float(ns_loss.mean().item())

#         # reduce
#         if num_items_in_batch is not None:
#             loss = loss.sum() / num_items_in_batch
#         else:
#             loss = loss.mean()
#         return loss

#     def hybrid_loss_topk(self, logits, labels, num_items_in_batch=None):
#         """
#         Suppress everything OUTSIDE Top-K (by softmax prob), excluding the ground-truth label.
#         """
#         T = float(getattr(self.args, "ns_temperature", 1.0))
#         K = int(getattr(self.args, "ns_top_k", 10))

#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         mask = shift_labels != -100
#         shift_logits, shift_labels = shift_logits[mask], shift_labels[mask]

#         if shift_logits.numel() == 0:
#             return self._empty_loss_like(logits)

#         s_loss = sparsemax_loss(shift_logits * T, shift_labels)

#         softmax_probs = F.softmax(shift_logits, dim=-1)  # keep grads
#         with torch.no_grad():
#             k = max(1, min(softmax_probs.size(-1) - 1, K))
#             _, topk_idx = torch.topk(softmax_probs, k=k, dim=-1)
#             topk_mask = torch.zeros_like(softmax_probs, dtype=torch.bool).scatter_(1, topk_idx, True)
#             one_hot = F.one_hot(shift_labels, num_classes=softmax_probs.size(-1)).bool()
#             neg_mask = ~topk_mask & ~one_hot

#         suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
#         ns_safe = torch.clamp(1.0 - suppressed_mass, min=1e-6)
#         ns_loss = -torch.log(ns_safe)

#         loss = s_loss + self.args.ns_alpha * ns_loss

#         with torch.no_grad():
#             self.recorded["suppressed_mass"] = float(suppressed_mass.mean().item())
#             self.recorded["s_loss"] = float(s_loss.mean().item())
#             self.recorded["ns_loss"] = float(ns_loss.mean().item())

#         if num_items_in_batch is not None:
#             loss = loss.sum() / num_items_in_batch
#         else:
#             loss = loss.mean()
#         return loss

#     def hybrid_loss_bottom_p(self, logits, labels, num_items_in_batch=None):
#         """
#         Suppress the bottom p% *by count* (lowest-prob tokens), excluding the ground-truth label.
#         NOTE: This is NOT nucleus/top-p by cumulative mass.
#         """
#         T = float(getattr(self.args, "ns_temperature", 1.0))
#         p = float(getattr(self.args, "ns_bottom_p", 0.9))

#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         mask = shift_labels != -100
#         shift_logits, shift_labels = shift_logits[mask], shift_labels[mask]

#         if shift_logits.numel() == 0:
#             return self._empty_loss_like(logits)

#         s_loss = sparsemax_loss(shift_logits * T, shift_labels)

#         softmax_probs = F.softmax(shift_logits, dim=-1)  # keep grads
#         with torch.no_grad():
#             V = softmax_probs.size(-1)
#             k = max(1, min(V - 1, int(round(V * p))))  # [1, V-1]
#             _, bottom_idx = torch.topk(softmax_probs, k=k, dim=-1, largest=False)
#             bottom_mask = torch.zeros_like(softmax_probs, dtype=torch.bool).scatter_(1, bottom_idx, True)
#             one_hot = F.one_hot(shift_labels, num_classes=V).bool()
#             neg_mask = bottom_mask & ~one_hot

#         suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
#         ns_safe = torch.clamp(1.0 - suppressed_mass, min=1e-6)
#         ns_loss = -torch.log(ns_safe)

#         loss = s_loss + self.args.ns_alpha * ns_loss

#         with torch.no_grad():
#             self.recorded["suppressed_mass"] = float(suppressed_mass.mean().item())
#             self.recorded["s_loss"] = float(s_loss.mean().item())
#             self.recorded["ns_loss"] = float(ns_loss.mean().item())

#         if num_items_in_batch is not None:
#             loss = loss.sum() / num_items_in_batch
#         else:
#             loss = loss.mean()
#         return loss

#     # ---------- Trainer overrides ----------

#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         # Keep labels for BOTH: (a) model CE loss (for logging), (b) our hybrid loss
#         labels = inputs.pop("labels")
#         outputs = model(**inputs, labels=labels)  # logits + standard CE loss
#         logits = outputs.get("logits")

#         # Route to chosen NS strategy
#         if self.args.ns_type == "top_k":
#             loss = self.hybrid_loss_topk(logits, labels, num_items_in_batch)
#         elif self.args.ns_type == "bottom_p":
#             loss = self.hybrid_loss_bottom_p(logits, labels, num_items_in_batch)
#         elif self.args.ns_type == "support_set":
#             loss = self.hybrid_loss_support_set(logits, labels, num_items_in_batch)
#         else:
#             raise ValueError(f"Unsupported ns_type: {self.args.ns_type}")

#         # Build logs
#         self.training_logs = self.compute_training_logs(logits, labels)
#         ce = outputs.get("loss", None)
#         if ce is not None:
#             self.training_logs["ce_loss"] = round(float(ce.item()), 4)
#         if "suppressed_mass" in self.recorded:
#             self.training_logs["suppressed_mass"] = round(self.recorded["suppressed_mass"], 4)
#         if "s_loss" in self.recorded:
#             self.training_logs["s_loss"] = round(self.recorded["s_loss"], 4)
#         if "ns_loss" in self.recorded:
#             self.training_logs["ns_loss"] = round(self.recorded["ns_loss"], 4)

#         return (loss, outputs) if return_outputs else loss

#     def _maybe_log_save_evaluate(
#         self,
#         tr_loss,
#         grad_norm,
#         model,
#         trial,
#         epoch,
#         ignore_keys_for_eval,
#         start_time,
#         learning_rate=None,
#     ):
#         # log
#         if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
#             if is_torch_xla_available():
#                 import torch_xla.core.xla_model as xm
#                 xm.mark_step()

#             logs = {}
#             tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
#             tr_loss -= tr_loss  # reset

#             logs["loss"] = round(
#                 tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4
#             )
#             if grad_norm is not None:
#                 logs["grad_norm"] = (
#                     grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
#                 )
#             logs["learning_rate"] = self._get_learning_rate()
#             if getattr(self, "training_logs", None):
#                 logs.update(self.training_logs)

#             self._total_loss_scalar += tr_loss_scalar
#             self._globalstep_last_logged = self.state.global_step
#             self.store_flos()

#             # Some repos override `log(logs, start_time)`. Try both signatures for safety.
#             try:
#                 self.log(logs, start_time)
#             except TypeError:
#                 self.log(logs)

#         # evaluate
#         metrics = None
#         if self.control.should_evaluate:
#             metrics = self._evaluate(trial, ignore_keys_for_eval)
#             is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)
#             if self.args.save_strategy == SaveStrategy.BEST:
#                 self.control.should_save = is_new_best_metric

#         # save
#         if self.control.should_save:
#             self._save_checkpoint(model, trial)
#             self.control = self.callback_handler.on_save(self.args, self.state, self.control)


# # ---------- utility ----------

# def chunked_entropy_from_logits(chunk_logits, batch_size=None):
#     """
#     Memory-frugal entropy from logits.
#     chunk_logits: [N, V]
#     returns: [N]
#     """
#     total_samples, _ = chunk_logits.shape
#     entropy_list = []
#     if batch_size is None:
#         batch_size = total_samples

#     for start in range(0, total_samples, batch_size):
#         end = min(start + batch_size, total_samples)
#         lb = chunk_logits[start:end]                 # [B, V]
#         logsumexp = torch.logsumexp(lb, dim=-1)      # [B]
#         norm_logits = lb - logsumexp.unsqueeze(-1)   # [B, V]
#         probs = torch.exp(norm_logits)               # [B, V]
#         entropy = logsumexp - (lb * probs).sum(-1)   # [B]
#         entropy_list.append(entropy)

#     if len(entropy_list) > 0:
#         return torch.cat(entropy_list, dim=0)
#     else:
#         return torch.tensor(0.0, device=chunk_logits.device)



# import torch
# import torch.nn.functional as F
# from transformers import Trainer
# from transformers.trainer import is_torch_xla_available, SaveStrategy
# from entmax import sparsemax, sparsemax_loss


# class HybridSFTTrainer(Trainer):
#     """
#     SFT Trainer for the hybrid loss:
#         Fenchel–Young(sparsemax) + ns_alpha * Negative Sampling (NS).

#     NS modes:
#       - top_k: suppress everything outside Top-K (by softmax prob), excluding GT label.
#       - bottom_p: suppress the bottom p% *by count* (NOT nucleus), excluding GT label.
#       - support_set: suppress everything outside sparsemax support, excluding GT label.

#     Notes:
#       * Temperature (ns_temperature) is applied to the sparse path (sparsemax & FY loss),
#         matching your colleague’s implementation.
#       * We keep gradients through softmax_probs for the NS term (do NOT wrap in no_grad).
#       * We clamp inside log to avoid -inf.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.recorded = {}  # per-step diagnostics

#     # ---------- helpers ----------

#     @staticmethod
#     def _empty_loss_like(logits):
#         # Return a scalar zero that keeps the graph intact if a microbatch has no valid tokens
#         return torch.zeros((), device=logits.device, dtype=logits.dtype, requires_grad=True)

#     @torch.no_grad()
#     def compute_training_logs(self, logits, labels):
#         """
#         Optional diagnostics:
#           - entropy over valid (shifted) tokens
#         """
#         shift_logits = logits[..., :-1, :]
#         shift_labels = labels[..., 1:]
#         mask = shift_labels != -100
#         shift_logits = shift_logits[mask]  # [N, V]

#         logs = {}
#         if getattr(self.args, "print_entropy", False) and shift_logits.numel() > 0:
#             entropy = chunked_entropy_from_logits(
#                 shift_logits,
#                 batch_size=max(1, shift_logits.size(0) // 4),
#             ).mean()
#             logs["entropy"] = round(float(entropy.item()), 4)
#         return logs

#     # ---------- loss variants ----------

#     def hybrid_loss_support_set(self, logits, labels, num_items_in_batch=None):
#         """
#         Suppress all tokens OUTSIDE sparsemax support (excluding the ground-truth label).
#         """
#         T = float(getattr(self.args, "ns_temperature", 1.0))

#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         mask = shift_labels != -100
#         shift_logits, shift_labels = shift_logits[mask], shift_labels[mask]

#         if shift_logits.numel() == 0:
#             return self._empty_loss_like(logits)

#         # 1) Fenchel–Young loss (sparsemax) with temperature on the sparse path
#         s_loss = sparsemax_loss(shift_logits * T, shift_labels)

#         # 2) Negative sampling: penalize softmax mass outside sparsemax support
#         softmax_probs = F.softmax(shift_logits, dim=-1)  # keep grads
#         with torch.no_grad():
#             sparsemax_probs = sparsemax(shift_logits * T, dim=-1)
#             support_set_mask = (sparsemax_probs > 0)
#             one_hot = F.one_hot(shift_labels, num_classes=shift_logits.size(-1)).bool()
#             neg_mask = ~support_set_mask & ~one_hot
#             # For monitoring sparsity:
#             support_size = support_set_mask.sum(dim=-1).float()  # [N]

#         suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
#         ns_safe = torch.clamp(1.0 - suppressed_mass, min=1e-6)
#         ns_loss = -torch.log(ns_safe)

#         # combine
#         loss = s_loss + self.args.ns_alpha * ns_loss

#         # logging means before reduction
#         with torch.no_grad():
#             self.recorded["suppressed_mass"] = float(suppressed_mass.mean().item())
#             self.recorded["s_loss"] = float(s_loss.mean().item())
#             self.recorded["ns_loss"] = float(ns_loss.mean().item())
#             self.recorded["support_size"] = float(support_size.mean().item())

#         # reduce
#         if num_items_in_batch is not None:
#             loss = loss.sum() / num_items_in_batch
#         else:
#             loss = loss.mean()
#         return loss

#     def hybrid_loss_topk(self, logits, labels, num_items_in_batch=None):
#         """
#         Suppress everything OUTSIDE Top-K (by softmax prob), excluding the ground-truth label.
#         """
#         T = float(getattr(self.args, "ns_temperature", 1.0))
#         K = int(getattr(self.args, "ns_top_k", 10))

#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         mask = shift_labels != -100
#         shift_logits, shift_labels = shift_logits[mask], shift_labels[mask]

#         if shift_logits.numel() == 0:
#             return self._empty_loss_like(logits)

#         s_loss = sparsemax_loss(shift_logits * T, shift_labels)

#         softmax_probs = F.softmax(shift_logits, dim=-1)  # keep grads
#         with torch.no_grad():
#             k = max(1, min(softmax_probs.size(-1) - 1, K))
#             _, topk_idx = torch.topk(softmax_probs, k=k, dim=-1)
#             topk_mask = torch.zeros_like(softmax_probs, dtype=torch.bool).scatter_(1, topk_idx, True)
#             one_hot = F.one_hot(shift_labels, num_classes=softmax_probs.size(-1)).bool()
#             neg_mask = ~topk_mask & ~one_hot

#         suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
#         ns_safe = torch.clamp(1.0 - suppressed_mass, min=1e-6)
#         ns_loss = -torch.log(ns_safe)

#         loss = s_loss + self.args.ns_alpha * ns_loss

#         with torch.no_grad():
#             self.recorded["suppressed_mass"] = float(suppressed_mass.mean().item())
#             self.recorded["s_loss"] = float(s_loss.mean().item())
#             self.recorded["ns_loss"] = float(ns_loss.mean().item())

#         if num_items_in_batch is not None:
#             loss = loss.sum() / num_items_in_batch
#         else:
#             loss = loss.mean()
#         return loss

#     def hybrid_loss_bottom_p(self, logits, labels, num_items_in_batch=None):
#         """
#         Suppress the bottom p% *by count* (lowest-prob tokens), excluding the ground-truth label.
#         NOTE: This is NOT nucleus/top-p by cumulative mass.
#         """
#         T = float(getattr(self.args, "ns_temperature", 1.0))
#         p = float(getattr(self.args, "ns_bottom_p", 0.9))

#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         mask = shift_labels != -100
#         shift_logits, shift_labels = shift_logits[mask], shift_labels[mask]

#         if shift_logits.numel() == 0:
#             return self._empty_loss_like(logits)

#         s_loss = sparsemax_loss(shift_logits * T, shift_labels)

#         softmax_probs = F.softmax(shift_logits, dim=-1)  # keep grads
#         with torch.no_grad():
#             V = softmax_probs.size(-1)
#             k = max(1, min(V - 1, int(round(V * p))))  # [1, V-1]
#             _, bottom_idx = torch.topk(softmax_probs, k=k, dim=-1, largest=False)
#             bottom_mask = torch.zeros_like(softmax_probs, dtype=torch.bool).scatter_(1, bottom_idx, True)
#             one_hot = F.one_hot(shift_labels, num_classes=V).bool()
#             neg_mask = bottom_mask & ~one_hot

#         suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
#         ns_safe = torch.clamp(1.0 - suppressed_mass, min=1e-6)
#         ns_loss = -torch.log(ns_safe)

#         loss = s_loss + self.args.ns_alpha * ns_loss

#         with torch.no_grad():
#             self.recorded["suppressed_mass"] = float(suppressed_mass.mean().item())
#             self.recorded["s_loss"] = float(s_loss.mean().item())
#             self.recorded["ns_loss"] = float(ns_loss.mean().item())

#         if num_items_in_batch is not None:
#             loss = loss.sum() / num_items_in_batch
#         else:
#             loss = loss.mean()
#         return loss

#     # ---------- Trainer overrides ----------

#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         # Keep labels for BOTH: (a) model CE loss (for logging), (b) our hybrid loss
#         labels = inputs.pop("labels")
#         outputs = model(**inputs, labels=labels)  # logits + standard CE loss
#         logits = outputs.get("logits")

#         # Route to chosen NS strategy
#         if self.args.ns_type == "top_k":
#             loss = self.hybrid_loss_topk(logits, labels, num_items_in_batch)
#         elif self.args.ns_type == "bottom_p":
#             loss = self.hybrid_loss_bottom_p(logits, labels, num_items_in_batch)
#         elif self.args.ns_type == "support_set":
#             loss = self.hybrid_loss_support_set(logits, labels, num_items_in_batch)
#         else:
#             raise ValueError(f"Unsupported ns_type: {self.args.ns_type}")

#         # Build logs
#         self.training_logs = self.compute_training_logs(logits, labels)
#         ce = outputs.get("loss", None)
#         if ce is not None:
#             self.training_logs["ce_loss"] = round(float(ce.item()), 4)
#         if "suppressed_mass" in self.recorded:
#             self.training_logs["suppressed_mass"] = round(self.recorded["suppressed_mass"], 4)
#         if "s_loss" in self.recorded:
#             self.training_logs["s_loss"] = round(self.recorded["s_loss"], 4)
#         if "ns_loss" in self.recorded:
#             self.training_logs["ns_loss"] = round(self.recorded["ns_loss"], 4)
#         if "support_size" in self.recorded:
#             self.training_logs["support_size"] = round(self.recorded["support_size"], 2)

#         return (loss, outputs) if return_outputs else loss

#     def _maybe_log_save_evaluate(
#         self,
#         tr_loss,
#         grad_norm,
#         model,
#         trial,
#         epoch,
#         ignore_keys_for_eval,
#         start_time,
#         learning_rate=None,
#     ):
#         # log
#         if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
#             if is_torch_xla_available():
#                 import torch_xla.core.xla_model as xm
#                 xm.mark_step()

#             logs = {}
#             tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
#             tr_loss -= tr_loss  # reset

#             logs["loss"] = round(
#                 tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4
#             )
#             if grad_norm is not None:
#                 logs["grad_norm"] = (
#                     grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
#                 )
#             logs["learning_rate"] = self._get_learning_rate()
#             if getattr(self, "training_logs", None):
#                 logs.update(self.training_logs)

#             self._total_loss_scalar += tr_loss_scalar
#             self._globalstep_last_logged = self.state.global_step
#             self.store_flos()

#             # Try both signatures for safety.
#             try:
#                 self.log(logs, start_time)
#             except TypeError:
#                 self.log(logs)

#         # evaluate: honor HF control OR fallback to eval every eval_steps if provided
#         metrics = None
#         should_eval = self.control.should_evaluate
#         if not should_eval:
#             es = getattr(self.args, "eval_steps", None)
#             if es and self.state.global_step > 0 and (self.state.global_step % es == 0) and self.eval_dataset is not None:
#                 should_eval = True

#         if should_eval:
#             metrics = self._evaluate(trial, ignore_keys_for_eval)
#             try:
#                 self.log(metrics, start_time)
#             except TypeError:
#                 self.log(metrics)
#             try:
#                 self.save_metrics("eval", metrics)
#             except Exception:
#                 pass

#             # keep BEST save behavior if user asked for it via SaveStrategy.BEST
#             if self.args.save_strategy == SaveStrategy.BEST:
#                 is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)
#                 self.control.should_save = is_new_best_metric

#         # save
#         if self.control.should_save:
#             self._save_checkpoint(model, trial)
#             self.control = self.callback_handler.on_save(self.args, self.state, self.control)


# # ---------- utility ----------

# def chunked_entropy_from_logits(chunk_logits, batch_size=None):
#     """
#     Memory-frugal entropy from logits.
#     chunk_logits: [N, V]
#     returns: [N]
#     """
#     total_samples, _ = chunk_logits.shape
#     entropy_list = []
#     if batch_size is None:
#         batch_size = total_samples

#     for start in range(0, total_samples, batch_size):
#         end = min(start + batch_size, total_samples)
#         lb = chunk_logits[start:end]                 # [B, V]
#         logsumexp = torch.logsumexp(lb, dim=-1)      # [B]
#         norm_logits = lb - logsumexp.unsqueeze(-1)   # [B, V]
#         probs = torch.exp(norm_logits)               # [B, V]
#         entropy = logsumexp - (lb * probs).sum(-1)   # [B]
#         entropy_list.append(entropy)

#     if len(entropy_list) > 0:
#         return torch.cat(entropy_list, dim=0)
#     else:
#         return torch.tensor(0.0, device=chunk_logits.device)


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
