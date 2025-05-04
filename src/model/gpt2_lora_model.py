from torch import nn
import torch.nn.init as init
import torch

from src.model.gpt2_adapter_model_base import GPT2AdapterBase
from transformers.models.gpt2.modeling_gpt2 import Conv1D

class LoRA(nn.Module):
    """
    Low-Rank Adaptation module for linear layers.
    
    Implements the LoRA method (Hu et al., 2021) which adapts pre-trained models
    by injecting trainable low-rank matrices into the attention layers.
    """
    def __init__(self, orig_module, rank, alpha, dropout_p, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        
        # Handle Conv1D layers from GPT2 which are used instead of nn.Linear
        if isinstance(orig_module, Conv1D):
            in_features = orig_module.nx  # input dim
            out_features = orig_module.nf  # output dim
        elif isinstance(orig_module, nn.Linear):
            in_features = orig_module.in_features
            out_features = orig_module.out_features
        else:
            raise ValueError(f"Unsupported module type: {type(orig_module)}")
            
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        init.normal_(self.lora_down.weight, mean=0, std=0.01)  # Init with N(0, a^2)

        self.lora_up = nn.Linear(rank, out_features, bias=False)
        init.zeros_(self.lora_up.weight)  # Init with zeros 
        self.rank = rank
        self.alpha = alpha
    
    def forward(self, x):
        """
        Forward pass for the LoRA module.
        
        Args:
            x: Input tensor
            
        Returns:
            Scaled output of low-rank adaptation
        """
        x = self.dropout(x)
        x = self.lora_down(x)
        x = self.lora_up(x)
        x = (self.alpha / self.rank) * x
        return x

class GPT2LoRA(GPT2AdapterBase):
    """
    GPT-2 model with Low-Rank Adaptation.
    
    This implementation applies LoRA to the query, key, value, and output projection
    matrices in the attention layers of GPT-2, as well as to the MLP intermediate
    and output layers if specified in the configuration.
    """
    def __init__(self, gpt2_config, lora_config, **cfg):
        super().__init__(**gpt2_config)
        
        # Freeze all parameters of the base model
        for p in self.parameters():
            p.requires_grad = False
        
        self.lora_config = lora_config
        
        # Create a ModuleList to properly register all LoRA modules
        self.lora_modules = nn.ModuleList()
        
        # Add LoRA modules to specified target layers
        # Note: GPT-2 uses Conv1D for attention layers, not nn.Linear
        for name, module in self.named_modules():
            if self.check_module(name, module):
                # Create a LoRA module for this layer
                lora_module = LoRA(module, **lora_config)
                
                # Add to ModuleList to ensure proper parameter registration
                self.lora_modules.append(lora_module)
                
                # Attach the LoRA module to the original module
                module.lora = lora_module
                
                # GPT-2's Conv1D forward passes tensors differently than Linear
                if isinstance(module, Conv1D):
                    # Save original forward method only if not already saved
                    if not hasattr(module, "original_forward"):
                        module.original_forward = module.forward
                    
                    # Define new forward for Conv1D
                    def make_conv1d_forward(mod):
                        def forward(x):
                            # Original Conv1D forward reshapes x
                            size_out = x.size()[:-1] + (mod.nf,)
                            original_output = mod.original_forward(x)
                            
                            # For LoRA, we need to handle the right shape
                            # If input is [batch, seq_len, input_dim]
                            # We need [batch, seq_len, output_dim] from LoRA
                            lora_output = mod.lora(x)
                            return original_output + lora_output
                        return forward
                    
                    # Set the new forward method
                    module.forward = make_conv1d_forward(module)
                    
                else:  # Standard nn.Linear
                    # Save original forward method only if not already saved
                    if not hasattr(module, "original_forward"):
                        module.original_forward = module.forward
                    
                    # Define a new forward method that adds LoRA output
                    def make_forward(mod):
                        def forward(x):
                            return mod.original_forward(x) + mod.lora(x)
                        return forward
                    
                    # Set the new forward method
                    module.forward = make_forward(module)
        
        # Print number of trainable parameters to verify setup
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"GPT2LoRA initialized with {trainable_params} trainable parameters")
        
    def check_module(self, name, module):
        """
        Check if a module should have LoRA applied to it.
        
        In GPT-2, we target:
        - Query, key, value projections in attention (c_attn) - these are Conv1D
        - Output projection in attention (c_proj) - these are Conv1D
        - MLP intermediate and output layers (c_fc, c_proj) - these are Conv1D
        
        Args:
            name: Full name of the module
            module: The module itself
            
        Returns:
            Boolean indicating whether this module should have LoRA
        """
        # Check if it's a Conv1D or Linear layer and in target layers list
        return (
            (isinstance(module, Conv1D) or isinstance(module, nn.Linear)) and
            name.split('.')[-1] in self.lora_config['target_layers']
        ) 