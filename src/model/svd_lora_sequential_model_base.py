from torch import nn

from src.model.adapter_model_base import AdapterModelBase
from src.model.svd_lora_sequential import SVDLoRASequential
from src.model.svd_lora_sequential_nested import SVDLoRASequentialNested


class SVDLoRASequentialModelBase(AdapterModelBase):
    def init_svd_lora_sequential(self, svd_lora_config, **cfg):
        self.current_adapter_idx = -10
        self.n_adapters = svd_lora_config['n_adapters']

        for p in self.parameters():
            p.requires_grad = False

        self.count_adaptable_weights = 0
        self.svd_lora_config = svd_lora_config

        if self.svd_lora_config.get("nested", False):
            SVDLoRAClass = SVDLoRASequentialNested
        else:
            SVDLoRAClass = SVDLoRASequential

        self.svd_loras = nn.ModuleList()
        
        for name, module in self.named_modules():
            if self.check_module(name, module):
                module.lora = SVDLoRAClass(module, enable_extra_loss=True, **svd_lora_config)
                module.forward = self.add_lora_forward(module)
                self.svd_loras.append(module.lora)
                self.count_adaptable_weights += 2
        
        self.disabled_adapters = False
        self.update_adapters(adapter_idx=0)
        # print("Init loss:", self.calc_extra_loss())

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

    def to(self, device, **kwargs):
        for name, param in self.named_parameters():
            if not "lora" in name:
                if param.data.device != device:
                    param.data = param.data.to(device, **kwargs)
                    if param.grad is not None:
                        param.grad.data = param.grad.data.to(device, **kwargs)

        for buffer_name, buffer in self.named_buffers():
            if buffer.device != device:
                print(buffer_name)
                setattr(self, buffer_name, buffer.to(device, **kwargs))

        for svd_lora in self.svd_loras:
            svd_lora.to(device, **kwargs)
        
        return self
