import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AdamW
from typing import Optional, List, Dict, Any, Tuple

class SMARTLoss(nn.Module):
    """
    SMART (Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models)
    implementation as described in the paper by Jiang et al.
    
    This loss combines:
    1. Smoothness-inducing adversarial regularization
    2. Bregman proximal point optimization
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        epsilon: float = 1e-3,
        perturb_step_size: float = 1e-3,
        perturb_noise_magnitude: float = 1e-5,
        adv_steps: int = 1,
        symmetrical: bool = False,
        kl_control: float = 0.5,
        bregman_control: float = 0.5,
    ):
        super(SMARTLoss, self).__init__()
        self.model = model
        self.epsilon = epsilon
        self.perturb_step_size = perturb_step_size
        self.perturb_noise_magnitude = perturb_noise_magnitude
        self.adv_steps = adv_steps
        self.symmetrical = symmetrical
        self.kl_control = kl_control
        self.bregman_control = bregman_control
        
        # Store previous model state parameters for Bregman divergence calculation
        self.previous_params = None
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        task_loss_fn: callable,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the SMART loss for the given inputs
        
        Args:
            inputs: Dictionary containing the model inputs
            task_loss_fn: Original task loss function (e.g., CrossEntropyLoss)
            
        Returns:
            Tuple of (total_loss, task_loss, smart_loss)
        """
        # Store current parameters for Bregman divergence if not already stored
        if self.previous_params is None:
            self.previous_params = {name: param.clone().detach()
                                   for name, param in self.model.named_parameters()}
        
        # Get model embedding inputs
        embedding_inputs = self._get_embeddings(inputs)
        
        # Original forward pass for task loss and clean logits
        outputs = self.model(**inputs)
        task_loss = task_loss_fn(outputs)
        clean_logits = outputs.logits
            
        # Generate perturbations for embeddings
        perturbed_embeddings = self._generate_adversarial_perturbation(
            embedding_inputs, 
            inputs.copy(), 
            clean_logits
        )
        
        # Forward pass with perturbed embeddings
        perturbed_inputs = inputs.copy()
        for key, perturbed_embedding in perturbed_embeddings.items():
            perturbed_inputs[key] = perturbed_embedding
            
        perturbed_outputs = self.model(**perturbed_inputs)
        perturbed_logits = perturbed_outputs.logits
        
        # Calculate KL divergence between clean and perturbed logits
        smart_loss = self._compute_kl_loss(clean_logits, perturbed_logits)
        
        # Calculate Bregman proximal regularization
        bregman_loss = self._compute_bregman_loss()
        
        # Combine losses
        smart_loss = self.kl_control * smart_loss + self.bregman_control * bregman_loss
        
        # Update stored parameters for next iteration
        self._update_previous_params()
        
        return task_loss + smart_loss, task_loss, smart_loss
    
    def _get_embeddings(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract embedding representations from the model based on inputs
        """
        embeddings = {}
        
        # Get word embeddings if input_ids are present
        if 'input_ids' in inputs:
            embeddings['word_embeddings'] = self.model.get_input_embeddings()(inputs['input_ids'])
            
        # Could be expanded to include other types of embeddings as needed
        
        return embeddings
    
    def _generate_adversarial_perturbation(
        self, 
        embeddings: Dict[str, torch.Tensor],
        inputs: Dict[str, torch.Tensor],
        clean_logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Generate adversarial perturbations for embeddings
        """
        perturbed_embeddings = {}
        
        for emb_name, embedding in embeddings.items():
            # Initialize perturbation with random noise
            delta = torch.zeros_like(embedding)
            delta.uniform_(-self.perturb_noise_magnitude, self.perturb_noise_magnitude)
            delta.requires_grad_()
            
            # Iteratively compute adversarial perturbation
            for _ in range(self.adv_steps):
                perturbed_embedding = embedding + delta
                
                # Replace original embedding with perturbed one
                perturbed_inputs = inputs.copy()
                if emb_name == 'word_embeddings':
                    # Bypass input_ids by directly setting the embeddings
                    # Implementation varies by model architecture
                    temp_inputs = perturbed_inputs.copy()
                    if 'input_ids' in temp_inputs:
                        del temp_inputs['input_ids']
                    temp_inputs['inputs_embeds'] = perturbed_embedding
                    perturbed_outputs = self.model(**temp_inputs)
                else:
                    # Handle other embedding types
                    continue
                
                perturbed_logits = perturbed_outputs.logits
                
                # Calculate KL divergence
                adv_loss = self._compute_kl_loss(clean_logits, perturbed_logits)
                
                # Compute gradient w.r.t. to delta
                adv_loss.backward(retain_graph=True)
                
                with torch.no_grad():
                    # Normalize gradients
                    delta_grad = delta.grad.clone()
                    grad_norm = torch.norm(delta_grad)
                    if grad_norm > 0:
                        delta_grad = delta_grad / grad_norm
                    
                    # Update delta with normalized gradients
                    delta = delta + self.perturb_step_size * delta_grad
                    
                    # Project delta to epsilon ball
                    delta_norm = torch.norm(delta)
                    if delta_norm > self.epsilon:
                        delta = delta * self.epsilon / delta_norm
                
                delta.grad.zero_()
            
            # Final perturbed embedding
            perturbed_embeddings[emb_name] = embedding + delta.detach()
        
        return perturbed_embeddings
    
    def _compute_kl_loss(self, clean_logits: torch.Tensor, perturbed_logits: torch.Tensor) -> torch.Tensor:
        """
        Calculate KL divergence between clean and perturbed logits
        """
        # Convert logits to log-probabilities
        clean_log_probs = F.log_softmax(clean_logits, dim=-1)
        perturbed_log_probs = F.log_softmax(perturbed_logits, dim=-1)
        
        # Calculate KL divergence
        kl_loss = F.kl_div(
            perturbed_log_probs, 
            clean_log_probs.exp(), 
            reduction='batchmean',
            log_target=False
        )
        
        # Symmetrical KL divergence if specified
        if self.symmetrical:
            reverse_kl = F.kl_div(
                clean_log_probs, 
                perturbed_log_probs.exp(), 
                reduction='batchmean',
                log_target=False
            )
            kl_loss = 0.5 * (kl_loss + reverse_kl)
        
        return kl_loss
    
    def _compute_bregman_loss(self) -> torch.Tensor:
        """
        Calculate Bregman proximal point regularization
        """
        bregman_loss = 0.0
        
        # Sum squared differences between current and previous parameters
        for name, param in self.model.named_parameters():
            if name in self.previous_params:
                diff = param - self.previous_params[name]
                bregman_loss += torch.sum(diff.pow(2))
        
        return bregman_loss
    
    def _update_previous_params(self):
        """
        Update stored parameters for the next iteration
        """
        self.previous_params = {name: param.clone().detach()
                               for name, param in self.model.named_parameters()}


class SMARTTrainer:
    """
    Trainer class that implements the SMART fine-tuning approach
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        task_loss_fn: callable,
        optimizer: torch.optim.Optimizer,
        epsilon: float = 1e-3,
        perturb_step_size: float = 1e-3,
        perturb_noise_magnitude: float = 1e-5,
        adv_steps: int = 1,
        symmetrical: bool = False,
        kl_control: float = 0.5,
        bregman_control: float = 0.5,
    ):
        self.model = model
        self.task_loss_fn = task_loss_fn
        self.optimizer = optimizer
        
        # Initialize SMART loss
        self.smart_loss = SMARTLoss(
            model=model,
            epsilon=epsilon,
            perturb_step_size=perturb_step_size,
            perturb_noise_magnitude=perturb_noise_magnitude,
            adv_steps=adv_steps,
            symmetrical=symmetrical,
            kl_control=kl_control,
            bregman_control=bregman_control,
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step with SMART regularization
        
        Args:
            batch: Dictionary containing the input batch
            
        Returns:
            Dictionary with loss metrics
        """
        self.optimizer.zero_grad()
        
        # Calculate combined loss using SMART
        total_loss, task_loss, smart_loss = self.smart_loss(
            inputs=batch,
            task_loss_fn=self.task_loss_fn,
        )
        
        # Backward and optimize
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'smart_loss': smart_loss.item(),
        }


# Example usage
def example_fine_tuning():
    """Example of using SMART for fine-tuning a BERT model on a classification task"""
    from transformers import BertForSequenceClassification, BertTokenizer
    import torch.optim as optim
    
    # Load pre-trained model and tokenizer
    model_name = "bert-base-uncased"
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Define loss function
    def task_loss_fn(outputs):
        # Example: Cross-entropy loss for classification
        # In real scenarios, you would use actual labels from your dataset
        batch_size = outputs.logits.shape[0]
        dummy_labels = torch.zeros(batch_size, dtype=torch.long)
        return F.cross_entropy(outputs.logits, dummy_labels)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Initialize SMART trainer
    trainer = SMARTTrainer(
        model=model,
        task_loss_fn=task_loss_fn,
        optimizer=optimizer,
        epsilon=1e-3,
        perturb_step_size=1e-3,
        kl_control=0.5,
        bregman_control=0.5,
    )
    
    # Example batch (in real scenarios, this would come from your dataloader)
    example_text = ["This is an example sentence", "Another example"]
    inputs = tokenizer(example_text, padding=True, truncation=True, return_tensors="pt")
    
    # Training step
    loss_metrics = trainer.train_step(inputs)
    
    print(f"Total Loss: {loss_metrics['total_loss']:.4f}")
    print(f"Task Loss: {loss_metrics['task_loss']:.4f}")
    print(f"SMART Loss: {loss_metrics['smart_loss']:.4f}")


if __name__ == "__main__":
    example_fine_tuning()
