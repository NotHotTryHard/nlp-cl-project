from torch import nn

from src.model.t5_lora_model import LoRA
from src.model.adapter_model_base import AdapterModelBase

class LoRASequentialModelBase(AdapterModelBase):
    def __init__(self, lora_config, **cfg):
        super(self, AdapterModelBase).__init__()
        self.current_adapter_idx = -10
        self.n_adapters = lora_config['n_adapters']

        for p in self.parameters():
            p.requires_grad = False

        self.lora_config = lora_config

        self.loras = nn.ModuleList([nn.ModuleDict() for _ in range(self.n_adapters)])
        
        for name, module in self.named_modules():
            if self.check_module(name, module):
                for i in range(self.n_adapters):
                    module_dict_name = name.replace('.', '_')
                    self.loras[i][module_dict_name] = LoRA(module, **lora_config)
        
        self.disabled_adapters = False
        self.update_adapters(adapter_idx=0)
    
    def update_adapters(self, adapter_idx):
        if self.current_adapter_idx != adapter_idx:
            self.current_adapter_idx = adapter_idx

            if adapter_idx == -1:
                print("Working without adapters!")
                self.disable_adapters()
                self.disabled_adapters = True
                return
            
            if self.disabled_adapters:
                self.enable_adapters()
                self.disabled_adapters = False
            
            print(f"Changing to {adapter_idx+1}-th adapter!")
            for name, module in self.named_modules():
                if self.check_module(name, module):
                    module_dict_name = name.replace('.', '_')
                    module.lora = self.loras[adapter_idx][module_dict_name]
                    module.forward = self.add_lora_forward(module)
