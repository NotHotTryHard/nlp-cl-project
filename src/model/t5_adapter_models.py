from torch import nn

from src.model.lora_model_base import LoRAModelBase
from src.model.lora_sequential_model_base import LoRASequentialModelBase
from src.model.svd_lora_model_base import SVDLoRAModelBase
from src.model.svd_lora_sequential_model_base import SVDLoRASequentialModelBase
from src.model.ignoretopk_model_base import IgnoreTopKModelBase
from src.model.t5model import T5forSummarization


class T5LoRA(T5forSummarization, LoRAModelBase):
    def __init__(self, t5_config, lora_config, **cfg):
        super.__init__(**t5_config)
        self.init_lora(lora_config, **cfg)
        
    def check_module(self, name, module):
        return isinstance(module, nn.Linear) and name.split('.')[-1] in self.lora_config['target_layers']


class T5LoRASequential(T5forSummarization, LoRASequentialModelBase):
    def __init__(self, t5_config, lora_config, **cfg):
        super().__init__(**t5_config)
        self.init_lora_sequential(self, lora_config, **cfg)

    def check_module(self, name, module):
        return isinstance(module, nn.Linear) and name.split('.')[-1] in self.lora_config['target_layers']


class T5SVDLoRA(T5forSummarization, SVDLoRAModelBase):
    def __init__(self, t5_config, svd_lora_config, **cfg):
        super().__init__(**t5_config)
        self.init_svd_lora(self, svd_lora_config, **cfg)

    def check_module(self, name, module):
        return isinstance(module, nn.Linear) and name.split('.')[-1] in self.svd_lora_config['target_layers']


class T5SVDLoRASequential(T5forSummarization, SVDLoRASequentialModelBase):
    def __init__(self, t5_config, svd_lora_config, **cfg):
        super().__init__(**t5_config)
        self.init_svd_lora_sequential(self, svd_lora_config, **cfg)

    def check_module(self, name, module):
        return isinstance(module, nn.Linear) and name.split('.')[-1] in self.svd_lora_config['target_layers']


class T5IgnoreTopK(T5forSummarization, IgnoreTopKModelBase):
    def __init__(self, t5_config, ignoretopk_config, output_hidden_states=True, **cfg):
        super().__init__(output_hidden_states=output_hidden_states, **t5_config)
        self.init_ignoretopk(self, **ignoretopk_config, **cfg)
        
    def forward(self, batch):
        outputs = T5forSummarization.forward(self, batch)
        return self.ignoretopk_on_forward(outputs)
