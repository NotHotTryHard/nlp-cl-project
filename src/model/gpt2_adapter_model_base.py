import torch
from torch import nn

from src.model.gpt2model import GPT2forGeneration

class GPT2AdapterBase(GPT2forGeneration):
    """
    Base class for GPT-2 models with adapter functionality.
    
    This class provides the foundation for adapter-based modifications to GPT-2,
    similar to how T5AdapterBase provides for T5. It implements methods to add
    and remove adapter modules from the forward pass.
    
    Unlike T5 which has separate encoder and decoder, GPT-2 is decoder-only,
    so adapters are applied only to the decoder stack.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def check_module(self, name, module):
        """
        Method to check if a module should have an adapter attached.
        Must be implemented by subclasses.
        """
        pass
        
    @staticmethod 
    def add_lora_forward(module):
        """
        Add Low-Rank Adaptation (LoRA) to a module's forward pass.
        
        Args:
            module: The PyTorch module to modify
            
        Returns:
            A new forward function that adds the LoRA result to the original
        """
        def new_forward(x):
            return module.original_forward(x) + module.lora(x)
        
        if not hasattr(module, "original_forward"):
            module.original_forward = module.forward
        
        return new_forward
    
    @staticmethod
    def remove_lora_forward(module):
        """
        Remove the LoRA adaptation and restore the original forward pass.
        
        Args:
            module: The PyTorch module to restore
            
        Returns:
            The original forward function
        """
        return module.original_forward
    
    def enable_adapters(self):
        """
        Enable all adapters in the model by modifying forward passes.
        """
        for name, module in self.named_modules():
            if self.check_module(name, module):
                module.forward = self.add_lora_forward(module)
    
    def disable_adapters(self):
        """
        Disable all adapters and restore original forward passes.
        """
        for name, module in self.named_modules():
            if self.check_module(name, module):
                module.forward = self.remove_lora_forward(module) 