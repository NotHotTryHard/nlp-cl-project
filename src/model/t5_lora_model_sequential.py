from torch import nn

from src.model.t5_lora_model import LoRA
from src.model.t5_adapter_model_base import T5AdapterBase

class T5LoRASequential(T5AdapterBase):
    def __init__(self, t5_config, lora_config, **cfg):
        super().__init__(**t5_config)

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
    
    def check_module(self, name, module):
        return isinstance(module, nn.Linear) and name.split('.')[-1] in self.lora_config['target_layers']
        
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
