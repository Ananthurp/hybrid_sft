import torch
import torch.nn.functional as F
from transformers import Trainer
from entmax import sparsemax, sparsemax_loss # type: ignore

class HybridSFTTrainer(Trainer):
    """
    An advanced SFT Trainer for a hybrid loss function.
    Combines Fenchel-Young (sparsemax) loss with Negative Sampling.
    Supports two modes for Negative Sampling:
    1. top_k: Suppresses all tokens NOT in the top-k most probable.
    2. bottom_p: Suppresses the bottom p% of the probability mass.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Extract custom hyperparameters from the TrainingArguments
        self.ns_alpha = self.args.ns_alpha
        self.ns_type = self.args.ns_type
        self.ns_top_k = self.args.ns_top_k
        self.ns_bottom_p = self.args.ns_bottom_p

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Main loss computation function.
        Delegates to the correct hybrid loss calculation based on ns_type.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.ns_type == "top_k":
            loss = self.hybrid_loss_topk(logits, labels)
        elif self.ns_type == "bottom_p":
            loss = self.hybrid_loss_bottom_p(logits, labels)
        else:
            raise ValueError(f"Unsupported ns_type: {self.ns_type}")

        return (loss, outputs) if return_outputs else loss

    def hybrid_loss_topk(self, logits, labels):
        """Calculates hybrid loss, suppressing all tokens NOT in the top-k."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]

        # 1. Fenchel-Young (Sparsemax) Loss
        fy_loss = sparsemax_loss(shift_logits, shift_labels).mean()

        # 2. Negative Sampling Loss (Top-K variant)
        softmax_probs = F.softmax(shift_logits, dim=-1)

        # Get the top-k probabilities and their indices
        _, topk_indices = torch.topk(softmax_probs, k=self.ns_top_k, dim=-1)

        # Create a mask that is True for all top-k tokens
        topk_mask = torch.zeros_like(softmax_probs, dtype=torch.bool).scatter_(1, topk_indices, True)
        
        # The ground truth label should never be suppressed
        one_hot_labels = F.one_hot(shift_labels, num_classes=softmax_probs.size(-1)).bool()
        
        # The negative mask is everything that is NOT in the top-k AND is NOT the true label
        neg_mask = ~topk_mask & ~one_hot_labels

        suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
        # Clamp to prevent log(0) for numerical stability
        suppressed_mass = torch.clamp(suppressed_mass, max=0.999)
        
        ns_loss = -torch.log(1.0 - suppressed_mass).mean()

        # 3. Combine losses
        total_loss = fy_loss + self.ns_alpha * ns_loss
        return total_loss

    def hybrid_loss_bottom_p(self, logits, labels):
        """Calculates hybrid loss, suppressing the bottom p% of probability mass."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]

        # 1. Fenchel-Young (Sparsemax) Loss
        fy_loss = sparsemax_loss(shift_logits, shift_labels).mean()

        # 2. Negative Sampling Loss (Bottom-P variant)
        softmax_probs = F.softmax(shift_logits, dim=-1)
        
        # Sort probabilities in descending order to find the threshold
        sorted_probs, _ = torch.sort(softmax_probs, dim=-1, descending=True)
        
        # Calculate the cumulative sum of probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find the threshold index where cumulative probability exceeds (1 - bottom_p)
        # This marks the boundary of the top tokens we want to KEEP
        threshold_mask = cumulative_probs > (1.0 - self.ns_bottom_p)
        
        # Get the probability value at that threshold
        # We need to gather the threshold probability for each row in the batch
        threshold_indices = torch.argmax(threshold_mask.int(), dim=1, keepdim=True)
        threshold_values = torch.gather(sorted_probs, 1, threshold_indices)

        # The negative mask is all probabilities less than or equal to this threshold value
        neg_mask = (softmax_probs <= threshold_values)
        
        # The ground truth label should never be suppressed
        one_hot_labels = F.one_hot(shift_labels, num_classes=softmax_probs.size(-1)).bool()
        neg_mask = neg_mask & ~one_hot_labels

        suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
        suppressed_mass = torch.clamp(suppressed_mass, max=0.999)
        
        ns_loss = -torch.log(1.0 - suppressed_mass).mean()

        # 3. Combine losses
        total_loss = fy_loss + self.ns_alpha * ns_loss
        return total_loss