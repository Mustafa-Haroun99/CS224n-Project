import torch
import torch.nn as nn
import torch.optim as optim

class BregmanProximalOptimizer:
    def __init__(self, model, base_optimizer, mu=0.1, divergence_type="squared"):
        """
        Implements Bregman Proximal Point Optimization.
        
        Args:
            model: PyTorch model.
            base_optimizer: Base optimizer (e.g., Adam, SGD).
            mu: Regularization parameter for Bregman divergence.
            divergence_type: Type of Bregman divergence ("squared" for Euclidean distance).
        """
        self.model = model
        self.base_optimizer = base_optimizer
        self.mu = mu
        self.divergence_type = divergence_type
        # Create a copy of model parameters as θ_t
        self.prev_model_state = {name: param.clone().detach() for name, param in model.named_parameters()}
    
    def step(self, loss_fn, *args):
        """
        Perform one optimization step with Bregman Proximal Regularization.
        
        Args:
            loss_fn: Loss function to be called with *args
            *args: Arguments to be passed to the loss function
        """
        # Store current parameters for Bregman term calculation
        current_params = {name: param.clone().detach() for name, param in self.model.named_parameters() if param.requires_grad}
        
        # Compute standard loss
        self.base_optimizer.zero_grad()
        loss = loss_fn(*args)
        
        # Add regularization for Bregman proximal term
        bregman_term = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                prev_param = self.prev_model_state[name]
                if self.divergence_type == "squared":
                    # Squared Euclidean distance
                    bregman_term += ((param - prev_param) ** 2).sum()
                # Could add other divergence types here
        
        # Total loss with Bregman regularization
        total_loss = loss + (self.mu / 2) * bregman_term
        total_loss.backward()
        
        # Update parameters
        self.base_optimizer.step()
        
        # Update stored θ_t after optimization
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.prev_model_state[name] = param.clone().detach()
                
        return loss.item()