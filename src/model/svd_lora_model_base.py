import torch
from torch import nn

from src.model.adapter_model_base import AdapterModelBase
from src.model.svd_lora import SVDLoRA


class SVDLoRAModelBase(AdapterModelBase):
    def __init__(self, svd_lora_config, **cfg):
        AdapterModelBase.__init__(self) 
        for p in self.parameters():
            p.requires_grad = False

        self.count_adaptable_weights = 0
        self.svd_lora_config = svd_lora_config

        self.svd_loras = []
        
        for name, module in self.named_modules():
            if self.check_module(name, module):
                module.lora = SVDLoRA(module, enable_extra_loss=True, **svd_lora_config)
                module.forward = self.add_lora_forward(module)
                self.svd_loras.append(module.lora)
                self.count_adaptable_weights += 2
        
        print("Init loss:", self.calc_extra_loss())
        
    def enable_extra_loss(self):
        for svd_lora in self.svd_loras:
            svd_lora.enable_extra_loss = True
    
    def disable_extra_loss(self):
        for svd_lora in self.svd_loras:
            svd_lora.enable_extra_loss = False

    def collect_extra_loss(self):
        extra_loss = torch.tensor(0., device=self.model.device)
        for svd_lora in self.svd_loras:
            extra_loss = extra_loss + svd_lora.extra_loss
            svd_lora.clean_extra_loss()
        
        extra_loss = extra_loss / self.count_adaptable_weights
        return extra_loss

    def calc_extra_loss(self):
        extra_loss = torch.tensor(0., device=self.model.device)
        for svd_lora in self.svd_loras:
            svd_lora.calc_extra_loss()
            extra_loss = extra_loss + svd_lora.extra_loss
            svd_lora.clean_extra_loss()

        extra_loss = extra_loss / self.count_adaptable_weights
        return extra_loss

    @staticmethod
    def add_lora_forward(module):
        def new_forward(x):
            return module.lora(x) # instead of module.original_forward(x) + module.lora(x)
        
        if not hasattr(module, "original_forward"):
            module.original_forward = module.forward
        
        return new_forward
    
    def reinit_adapters(self):
        for svd_lora in self.svd_loras:
            svd_lora.reinit_self()
    
    def collect_singular_values(self, module_names):
        singular_values = {}
        for name, module in self.named_modules(): # as it's a generator
            if name in module_names:
                singular_values[name] = module.lora.s
        return singular_values
