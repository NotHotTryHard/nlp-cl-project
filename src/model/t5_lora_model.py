from torch import nn

from src.model.t5model import T5forSummarization

class LoRA(nn.Module):
    def __init__(self, orig_module, rank, alpha, dropout_p, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.lora_down = nn.Linear(orig_module.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, orig_module.out_features, bias=False)
        self.rank = rank
        self.alpha = alpha
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.lora_down(x)
        x = self.lora_up(x)
        x = (self.alpha / self.rank) * x
        return x


class T5LoRA(T5forSummarization):
    def __init__(self, t5_config, lora_config, **cfg):
        super().__init__(**t5_config)
        
        for p in self.parameters():
            p.requires_grad = False
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and name.split('.')[-1] in lora_config['target_layers']:
                module.lora = LoRA(module, **lora_config)
                module.forward = self.add_lora_forward(module)
        
    @staticmethod
    def add_lora_forward(module):
        def new_forward(x):
            return module.original_forward(x) + module.lora(x)
        module.original_forward = module.forward
        return new_forward 


class T5LoRASequential(T5forSummarization):
    def __init__(self, t5_config, lora_config, **cfg):
        super().__init__(**t5_config)

        self.current_adapter_idx = 0
        self.n_adapters = lora_config['n_adapters']
        self.lora_config = lora_config
        
        for p in self.parameters():
            p.requires_grad = False

        self.loras = [{} for _ in range(self.n_adapters)]
        
        for name, module in self.named_modules():
            if self.check_module(name, module):
                for i in range(self.n_adapters):
                    self.loras[i][name] = LoRA(module, **lora_config)
        
        self.update_adapter(adapter_index=0)
    
    def update_adapter(self, adapter_index):
        for name, module in self.named_modules():
            if self.check_module(name, module):
                module.lora = self.loras[adapter_index][name]
                module.forward = self.add_lora_forward(module)

    def check_module(self, name, module):
        return isinstance(module, nn.Linear) and name.split('.')[-1] in self.lora_config['target_layers']

    def update_adapter_index(self, adapter_idx):
        if self.current_adapter_idx != adapter_idx:
            print(f"Changing to {adapter_idx+1}-th adapter!")
            self.update_adapter(adapter_idx)
        self.current_adapter_idx = adapter_idx
        
    @staticmethod
    def add_lora_forward(module):
        def new_forward(x):
            return module.original_forward(x) + module.lora(x)
        module.original_forward = module.forward
        return new_forward 
