import torch
from torch import nn

from src.model.t5model import T5forSummarization

class SVDLoRA(nn.Module):
    def __init__(self, orig_module, rank, **kwargs):
        super().__init__()
        # self.dropout = nn.Dropout(dropout_p)
        # self.lora_down = nn.Linear(orig_module.in_features, rank, bias=False)
        # self.lora_up = nn.Linear(rank, orig_module.out_features, bias=False)
        # self.rank = rank
        # self.alpha = alpha
        
        # weight - (out_features, in_features)
        self.u, self.s, self.vt = torch.linalg.svd(orig_module.weight.T, full_matrices=False) 
        # u - out_features, k
        # s - k
        # vt - k, in_features
        self.in_features = orig_module.in_features
        self.out_features = orig_module.out_features
        self.k = self.s.shape[0]
        
        self.lora_u = nn.Parameter(self.u[:, -rank:], requires_grad=True) # out_features, rank
        self.lora_s = nn.Parameter(self.s[-rank:], requires_grad=True)  # rank, 
        self.lora_vt = nn.Parameter(self.vt[-rank:, :], requires_grad=True) # rank, in_features
        
        self.u = nn.Parameter(self.u[:, :-rank], requires_grad=False)
        self.s = nn.Parameter(self.s[:-rank], requires_grad=False)
        self.vt = nn.Parameter(self.vt[:-rank, :], requires_grad=False)
    
    def forward(self, x):
        # print('=='*10)
        # print("x.shape", x.shape)
        # print("self.u", self.u.shape)
        # print("self.s", self.s.shape)
        # print("self.vt", self.vt.shape)
        orig_pass = x @ self.u @ torch.diag(self.s) @ self.vt
        lora_pass = x @ self.lora_u @ torch.diag(self.lora_s) @ self.lora_vt
        return orig_pass + lora_pass


class T5SVDLoRA(T5forSummarization):
    def __init__(self, t5_config, svd_lora_config, **cfg):
        super().__init__(**t5_config)
        
        for p in self.parameters():
            p.requires_grad = False
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and name.split('.')[-1] in svd_lora_config['target_layers']:
                module.lora = SVDLoRA(module, **svd_lora_config)
                module.forward = self.add_lora_forward(module)
        
        print("Init loss:", self.calc_extra_loss())
                
        
        
    def add_lora_forward(self, module):
        def new_forward(x):
            return module.lora(x)
        module.original_forward = module.forward
        return new_forward 
    
    
    def calc_extra_loss(self, ):
        res = 0
        cnt = 0
        for name, module in self.named_modules():
            if hasattr(module, 'lora'):
                mod = getattr(module, 'lora')
                u_cat = torch.cat([mod.u, mod.lora_u], 1)
                vt_cat = torch.cat([mod.vt, mod.lora_vt], 0)
                u_norm = torch.norm(u_cat.T @ u_cat - torch.eye(mod.k, device=u_cat.device)) 
                vt_norm = torch.norm(vt_cat @ vt_cat.T - torch.eye(mod.k, device=u_cat.device)) 
                res += (u_norm + vt_norm)
                cnt += 2
        return (res / cnt)