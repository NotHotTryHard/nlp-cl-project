from torch import nn

from src.model.lora_model_base import LoRAModelBase
from src.model.lora_sequential_model_base import LoRASequentialModelBase
from src.model.svd_lora_model_base import SVDLoRAModelBase
from src.model.svd_lora_sequential_model_base import SVDLoRASequentialModelBase
from src.model.ignoretopk_model_base import IgnoreTopKModelBase
from src.model.t5model import T5forSummarization


class T5LoRA(LoRAModelBase, T5forSummarization):
    def __init__(self, t5_config, lora_config, **cfg):
        T5forSummarization.__init__(self, **t5_config)
        LoRAModelBase.__init__(self, lora_config, **cfg)
        
    def check_module(self, name, module):
        return isinstance(module, nn.Linear) and name.split('.')[-1] in self.lora_config['target_layers']


class T5LoRASequential(LoRASequentialModelBase, T5forSummarization):
    def __init__(self, t5_config, lora_config, **cfg):
        T5forSummarization.__init__(self, **t5_config)
        LoRASequentialModelBase.__init__(self, lora_config, **cfg)

    def check_module(self, name, module):
        return isinstance(module, nn.Linear) and name.split('.')[-1] in self.lora_config['target_layers']


class T5SVDLoRA(SVDLoRAModelBase, T5forSummarization):
    def __init__(self, t5_config, svd_lora_config, **cfg):
        T5forSummarization.__init__(self, **t5_config)
        SVDLoRAModelBase.__init__(self, svd_lora_config, **cfg)

    def check_module(self, name, module):
        return isinstance(module, nn.Linear) and name.split('.')[-1] in self.svd_lora_config['target_layers']


class T5SVDLoRASequential(SVDLoRASequentialModelBase, T5forSummarization):
    def __init__(self, t5_config, svd_lora_config, **cfg):
        T5forSummarization.__init__(self, **t5_config)
        SVDLoRASequentialModelBase.__init__(self, svd_lora_config, **cfg)

    def check_module(self, name, module):
        return isinstance(module, nn.Linear) and name.split('.')[-1] in self.svd_lora_config['target_layers']


class T5IgnoreTopK(IgnoreTopKModelBase, T5forSummarization):
    def __init__(self, t5_config, ignoretopk_config, output_hidden_states=True, **cfg):
        T5forSummarization.__init__(self, output_hidden_states=output_hidden_states, **t5_config)
        IgnoreTopKModelBase.__init__(self, **ignoretopk_config, **cfg)
        
    def forward(self, batch):
        outputs = T5forSummarization.forward(self, batch)
        return self.ignoretopk_on_forward(outputs)
