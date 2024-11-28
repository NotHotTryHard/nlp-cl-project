from torch import nn

from src.model.t5model import T5forSummarization

class LoRA(nn.Module):
    def __init__(self, orig_module, rank, alpha, dropout_p):
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
        
        for name, module in model.named_modules():
            module.requires_grad = False
            if 'hui' in name:
                module.lora = LoRA(module, **lora_config)
                module.forward = self.add_lora_forward(module)
                module.lora.requires_grad = True
        
        
    def add_lora_forward(self, module):
        def new_forward(x):
            return module.original_forward(x) + module.lora(x)
        module.original_forward = module.forward
        return new_forward 