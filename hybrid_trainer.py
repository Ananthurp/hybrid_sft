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


import torch
import torch.nn.functional as F
from transformers import Trainer
from entmax import sparsemax, sparsemax_loss
import torch.distributed as dist

class HybridSFTTrainer(Trainer):
    """
    An advanced SFT Trainer for a hybrid loss function, adapted from the reference repository.
    Combines Fenchel-Young (sparsemax) loss with Negative Sampling.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorded = {} # For logging detailed metrics

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Route to the correct loss calculation based on the ns_type argument
        if self.args.ns_type == "top_k":
            loss = self.hybrid_loss_topk(logits, labels, num_items_in_batch=num_items_in_batch)
        elif self.args.ns_type == "bottom_p":
            loss = self.hybrid_loss_bottom_p(logits, labels, num_items_in_batch=num_items_in_batch)
        else:
            raise ValueError(f"Unsupported ns_type: {self.args.ns_type}")

        return (loss, outputs) if return_outputs else loss

    def hybrid_loss_topk(self, logits, labels, num_items_in_batch=None):
        """Calculates hybrid loss, suppressing all tokens NOT in the top-k."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        mask = shift_labels != -100
        shift_logits, shift_labels = shift_logits[mask], shift_labels[mask]

        s_loss = sparsemax_loss(shift_logits, shift_labels)

        with torch.no_grad():
            softmax_probs = F.softmax(shift_logits, dim=-1)
            topk_values, topk_indices = torch.topk(softmax_probs, k=self.args.ns_top_k, dim=-1)
            
            topk_mask = torch.zeros_like(softmax_probs, dtype=torch.bool).scatter_(1, topk_indices, True)
            one_hot_labels = F.one_hot(shift_labels, num_classes=softmax_probs.size(-1)).bool()
            
            neg_mask = ~topk_mask & ~one_hot_labels

        suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
        suppressed_mass = torch.clamp(suppressed_mass, max=0.999)
        
        ns_loss = -torch.log(1.0 - suppressed_mass)
        
        loss = s_loss + self.args.ns_alpha * ns_loss

        if num_items_in_batch is not None:
            loss = loss.sum() / num_items_in_batch
        else:
            loss = loss.mean()
            
        return loss

    def hybrid_loss_bottom_p(self, logits, labels, num_items_in_batch=None):
        """Calculates hybrid loss, suppressing the bottom p% of probability mass."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        mask = shift_labels != -100
        shift_logits, shift_labels = shift_logits[mask], shift_labels[mask]

        s_loss = sparsemax_loss(shift_logits, shift_labels)

        with torch.no_grad():
            softmax_probs = F.softmax(shift_logits, dim=-1)
            
            # This logic is adapted from your colleague's sft_trainer_v2.py
            # It uses ranking to determine the threshold for negative sampling
            ranks = torch.argsort(torch.argsort(softmax_probs, dim=-1, descending=True), dim=-1) + 1
            one_hot_labels = F.one_hot(shift_labels, num_classes=softmax_probs.size(-1)).bool()
            
            # The threshold tau is now interpreted as a percentage of the vocabulary size
            vocab_size = softmax_probs.size(-1)
            rank_threshold = int(vocab_size * self.args.ns_bottom_p)
            
            neg_mask = (ranks > rank_threshold) & (~one_hot_labels)

        suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
        suppressed_mass = torch.clamp(suppressed_mass, max=0.999)
        
        ns_loss = -torch.log(1.0 - suppressed_mass)

        loss = s_loss + self.args.ns_alpha * ns_loss

        if num_items_in_batch is not None:
            loss = loss.sum() / num_items_in_batch
        else:
            loss = loss.mean()

        return loss