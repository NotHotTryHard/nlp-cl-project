import torch
from torch import nn

class SVDLoRASequentialNested(nn.Module):
    def __init__(self, orig_module, enable_extra_loss, rank, n_adapters, **kwargs):
        """
        This implementation is nested, for instance:
        Given rank = 4, n_adapters = 3:
        First adapter trains last 12 singular vectors / values
        Second adapter trains last 8 singular vectors / values
        Third adapter trains last 4 singular vectors / values
        """
        super().__init__()
        self.u, self.s, self.vt = torch.linalg.svd(orig_module.weight.T, full_matrices=False)
        self.in_features = orig_module.in_features
        self.out_features = orig_module.out_features

        self.rank = rank
        self.k = self.s.shape[0]
        self.n_adapters = n_adapters

        self.register_buffer('orig_weight', self.u @ torch.diag(self.s) @ self.vt)

        self.u = nn.Parameter(self.u, requires_grad=False)
        self.s = nn.Parameter(self.s, requires_grad=False)
        self.vt = nn.Parameter(self.vt, requires_grad=False)

        self.lora_u = None
        self.lora_s = None
        self.lora_vt = None

        self.extra_loss = torch.tensor(0., device=self.u.device)
        self.enable_extra_loss = enable_extra_loss
    
        self.cur_idx = -1
        self.update_adapter(0)
    
    def update_adapter(self, new_idx):
        if new_idx != self.cur_idx:
            if self.cur_idx != -1:                
                for lora_part in [self.lora_u, self.lora_s, self.lora_vt]:
                    lora_part.requires_grad = False
                    
                prev_lora_weight = (self.lora_u @ torch.diag(self.lora_s) @ self.lora_vt).detach().clone()
                self.orig_weight.data = self.orig_weight + prev_lora_weight

            start_index = -self.rank * (new_idx + 1) #(self.n_adapters - new_idx)
            self.lora_u = nn.Parameter(self.u[:, start_index:], requires_grad=False)
            self.lora_s = nn.Parameter(self.s[start_index:], requires_grad=False)
            self.lora_vt = nn.Parameter(self.vt[start_index:, :], requires_grad=False)

            new_lora_weight = (self.lora_u @ torch.diag(self.lora_s) @ self.lora_vt).detach().clone()
            self.orig_weight.data = self.orig_weight - new_lora_weight
            
            for lora_part in [self.lora_u, self.lora_s, self.lora_vt]:
                lora_part.requires_grad = True
            
            self.cur_idx = new_idx
        
    def calc_extra_loss(self):
        u_cat = torch.cat([self.u[:, :-self.rank * (self.cur_idx + 1)], self.lora_u], 1)
        vt_cat = torch.cat([self.vt[:-self.rank * (self.cur_idx + 1), :], self.lora_vt], 0)
        u_norm = torch.norm(u_cat.T @ u_cat - torch.eye(self.k, device=self.u.device, requires_grad=False))
        vt_norm = torch.norm(vt_cat @ vt_cat.T - torch.eye(self.k, device=self.u.device, requires_grad=False))
        self.extra_loss = u_norm + vt_norm
    
    def clean_extra_loss(self):
        self.extra_loss = torch.tensor(0., device=self.u.device)
    
    def forward(self, x):        
        output = x @ (self.orig_weight + self.lora_u @ torch.diag(self.lora_s) @ self.lora_vt)

        if self.enable_extra_loss:
            self.calc_extra_loss()
        
        return output

    def to(self, device, **kwargs):
        self.u.data = self.u.data.to(device, **kwargs)
        self.s.data = self.s.data.to(device, **kwargs)
        self.vt.data = self.vt.data.to(device, **kwargs)
        self.orig_weight = self.orig_weight.to(device, **kwargs)
        
        for lora_submodule in [self.lora_u, self.lora_s, self.lora_vt]:
            lora_submodule.data = lora_submodule.data.to(device, **kwargs)
        
        return self
