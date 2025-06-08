from torch import nn

from src.model.gpt2model import GPT2forGeneration

class GPT2AdapterBase(GPT2forGeneration):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def check_module(self, name, module):
        pass
        
    @staticmethod 
    def add_lora_forward(module):
        def new_forward(x):
            return module.original_forward(x) + module.lora(x)
        
        if not hasattr(module, "original_forward"):
            module.original_forward = module.forward
        
        return new_forward
    
    @staticmethod
    def remove_lora_forward(module):
        return module.original_forward
    
    def enable_adapters(self):
        for name, module in self.named_modules():
            if self.check_module(name, module):
                module.forward = self.add_lora_forward(module)
    
    def disable_adapters(self):
        for name, module in self.named_modules():
            if self.check_module(name, module):
                module.forward = self.remove_lora_forward(module) 