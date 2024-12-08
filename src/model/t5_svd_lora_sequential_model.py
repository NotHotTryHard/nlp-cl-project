import torch
from torch import nn

from src.model.t5_adapter_model_base import T5AdapterBase

class SVDLoRASequential(nn.Module):
    def __init__(self, orig_module, enable_extra_loss, rank, n_adapters, **kwargs):
        super().__init__()
        self.u, self.s, self.vt = torch.linalg.svd(orig_module.weight.T, full_matrices=False)
        self.in_features = orig_module.in_features
        self.out_features = orig_module.out_features

        self.k = self.s.shape[0]
        self.n_adapters = n_adapters

        self.loras_u = nn.ParameterList()
        self.loras_s = nn.ParameterList()
        self.loras_vt = nn.ParameterList()

        for i in range(n_adapters):
            start_index = -rank * (i + 1)
            end_index = -rank * i if i else self.u.shape[1]
            self.loras_u.append(nn.Parameter(self.u[:, start_index:end_index], requires_grad=False)) # out_features, rank
            self.loras_s.append(nn.Parameter(self.s[start_index:end_index], requires_grad=False))  # rank, 
            self.loras_vt.append(nn.Parameter(self.vt[start_index:end_index, :], requires_grad=False)) # rank, in_features
        
        self.u = nn.Parameter(self.u[:, :-rank * n_adapters], requires_grad=False)
        self.s = nn.Parameter(self.s[:-rank * n_adapters], requires_grad=False)
        self.vt = nn.Parameter(self.vt[:-rank * n_adapters, :], requires_grad=False)

        self.register_buffer('orig_weight', self.u @ torch.diag(self.s) @ self.vt)

        self.extra_loss = torch.tensor(0., device=self.u.device)
        self.enable_extra_loss = enable_extra_loss
    
        self.current_adapter_index = 0
        self.update_adapter(0)
    
    def update_adapter(self, current_adapter_index):
        self.current_adapter_index = current_adapter_index
        for i in range(self.n_adapters):
            requires_grad = (i == self.current_adapter_index)
            self.loras_u[i].requires_grad = requires_grad
            self.loras_s[i].requires_grad = requires_grad
            self.loras_vt[i].requires_grad = requires_grad
        
    def calc_extra_loss(self):
        u_cat = torch.cat([self.u, *[lora_u for lora_u in self.loras_u]], 1)
        vt_cat = torch.cat([self.vt, *[lora_vt for lora_vt in self.loras_vt]], 0)
        u_norm = torch.norm(u_cat.T @ u_cat - torch.eye(self.k, device=self.u.device, requires_grad=False))
        vt_norm = torch.norm(vt_cat @ vt_cat.T - torch.eye(self.k, device=self.u.device, requires_grad=False))
        self.extra_loss = u_norm + vt_norm
    
    def clean_extra_loss(self):
        self.extra_loss = torch.tensor(0., device=self.u.device)
    
    def forward(self, x):
        output = x @ self.orig_weight
        for i in range(self.n_adapters):
            lora_output = x @ self.loras_u[i] @ torch.diag(self.loras_s[i]) @ self.loras_vt[i]
            output = output + lora_output

        if self.enable_extra_loss:
            self.calc_extra_loss()
        
        return output


class T5SVDLoRASequential(T5AdapterBase):
    def __init__(self, t5_config, svd_lora_config, **cfg):
        super().__init__(**t5_config)

        self.current_adapter_idx = -10
        self.n_adapters = svd_lora_config['n_adapters']

        for p in self.parameters():
            p.requires_grad = False

        self.count_adaptable_weights = 0
        self.svd_lora_config = svd_lora_config

        self.svd_loras = nn.ModuleList()
        
        for name, module in self.named_modules():
            if self.check_module(name, module):
                for i in range(self.n_adapters):
                    module.lora = SVDLoRASequential(module, enable_extra_loss=True, **svd_lora_config)
                    module.forward = self.add_lora_forward(module)
                    self.svd_loras.append(module.lora)
                    self.count_adaptable_weights += 2
        
        self.disabled_adapters = False
        self.update_adapters(adapter_idx=0)
        print("Init loss:", self.calc_extra_loss())

    def check_module(self, name, module):
        return isinstance(module, nn.Linear) and name.split('.')[-1] in self.svd_lora_config['target_layers']

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
            for svd_lora in self.svd_loras:
                svd_lora.update_adapter(adapter_idx)
    
    def enable_extra_loss(self):
        for name, module in self.named_modules():
            if self.check_module(name, module):
                module.lora.enable_extra_loss = True
    
    def disable_extra_loss(self):
        for name, module in self.named_modules():
            if self.check_module(name, module):
                module.lora.enable_extra_loss = False
            
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
