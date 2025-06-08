import torch
from torch import nn
from transformers.modeling_utils import Conv1D

from src.model.gpt2_adapter_model_base import GPT2AdapterBase

class SVDLoRA(nn.Module):
    def __init__(
            self,
            orig_module,
            enable_extra_loss,
            rank,
            reinit_lora=False,
            reinit_singular_values=False,
            reinit_std=1.0,
            reinit_singular_values_std=1.0,
            reinit_use_qr=True,
            reinit_ortho_project=True,
            **kwargs
    ):
        super().__init__()
        self.in_features = orig_module.weight.shape[0]
        self.out_features = orig_module.weight.shape[1]

        self.rank = rank
        self.reinit_lora = reinit_lora
        self.reinit_singular_values = reinit_singular_values
        self.reinit_std = reinit_std
        self.reinit_singular_values_std = reinit_singular_values_std
        self.reinit_use_qr = reinit_use_qr
        self.reinit_ortho_project = reinit_ortho_project

        self.init_with_weight(orig_module.weight)
        
        self.extra_loss = torch.tensor(0., device=self.u.device)
        self.enable_extra_loss = enable_extra_loss
    
    def calc_extra_loss(self):
        u_cat = torch.cat([self.u, self.lora_u], 1)
        vt_cat = torch.cat([self.vt, self.lora_vt], 0)
        u_norm = torch.norm(u_cat.T @ u_cat - torch.eye(self.k, device=self.u.device, requires_grad=False))
        vt_norm = torch.norm(vt_cat @ vt_cat.T - torch.eye(self.k, device=self.u.device, requires_grad=False))
        self.extra_loss = u_norm + vt_norm
    
    def clean_extra_loss(self):
        self.extra_loss = torch.tensor(0., device=self.u.device)
    
    def lora_orthogonal_reinit(self):
        with torch.no_grad():
            nn.init.normal_(self.lora_u, mean=0.0, std=self.reinit_std)
            nn.init.normal_(self.lora_vt, mean=0.0, std=self.reinit_std)
        
        if self.reinit_singular_values:
            nn.init.normal_(self.lora_s, mean=0.0, std=self.reinit_singular_values_std)
        
        lora_u = self.lora_u
        lora_v = self.lora_vt.T
        
        if self.reinit_ortho_project:
            Iu = torch.eye(self.u.shape[0], dtype=self.u.dtype, device=self.u.device)
            Iv = torch.eye(self.vt.shape[1], dtype=self.vt.dtype, device=self.vt.device)
            lora_u = (Iu - self.u @ self.u.T) @ self.lora_u
            lora_v = (Iv - self.vt.T @ self.vt) @ self.lora_vt.T
        
        if self.reinit_use_qr:
            lora_u, _ = torch.linalg.qr(lora_u)
            lora_v, _ = torch.linalg.qr(lora_v)

        with torch.no_grad():
            self.lora_u.copy_(lora_u)
            self.lora_vt.copy_(lora_v.T)

    def init_with_weight(self, prev_weight):
        self.u, self.s, self.vt = torch.linalg.svd(prev_weight.data, full_matrices=False) 
        self.k = self.s.shape[0]

        self.lora_u = nn.Parameter(self.u[:, -self.rank:], requires_grad=True)
        self.lora_s = nn.Parameter(self.s[-self.rank:], requires_grad=True)
        self.lora_vt = nn.Parameter(self.vt[-self.rank:, :], requires_grad=True)
        
        self.u = nn.Parameter(self.u[:, :-self.rank], requires_grad=False)
        self.s = nn.Parameter(self.s[:-self.rank], requires_grad=False)
        self.vt = nn.Parameter(self.vt[:-self.rank, :], requires_grad=False)

        self.register_buffer('orig_weight', self.u @ torch.diag(self.s) @ self.vt)

        if self.reinit_lora:
            self.lora_orthogonal_reinit()

    def reinit_self(self):
        weight = self.orig_weight + self.lora_u @ torch.diag(self.lora_s) @ self.lora_vt
        self.init_with_weight(weight)
        self.extra_loss = torch.tensor(0., device=self.u.device)
    
    def forward(self, x):
        orig_pass = x @ self.orig_weight
        lora_pass = x @ (self.lora_u @ torch.diag(self.lora_s) @ self.lora_vt)

        if self.enable_extra_loss:
            self.calc_extra_loss()
        
        return orig_pass + lora_pass

class GPT2SVDLoRA(GPT2AdapterBase):
    def __init__(self, gpt2_config, svd_lora_config, **cfg):
        super().__init__(**gpt2_config)
        
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
        
    def check_module(self, name, module):
        return isinstance(module, Conv1D) and name.split('.')[-1] in self.svd_lora_config['target_layers']

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
        
        return extra_loss / self.count_adaptable_weights if self.count_adaptable_weights > 0 else 0.0

    def calc_extra_loss(self):
        extra_loss = torch.tensor(0., device=self.model.device)
        for svd_lora in self.svd_loras:
            svd_lora.calc_extra_loss()
            extra_loss = extra_loss + svd_lora.extra_loss
            svd_lora.clean_extra_loss()

        return extra_loss / self.count_adaptable_weights if self.count_adaptable_weights > 0 else 0.0

    @staticmethod 
    def add_lora_forward(module):
        def new_forward(x):
            return module.lora(x)
        
        if not hasattr(module, "original_forward"):
            module.original_forward = module.forward
        
        return new_forward
    
    def reinit_adapters(self):
        for svd_lora in self.svd_loras:
            svd_lora.reinit_self()
    
    def collect_singular_values(self, module_names):
        singular_values = {}
        for name, module in self.named_modules():
            if name in module_names:
                singular_values[name] = module.lora.s
        return singular_values 