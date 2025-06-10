from transformers.modeling_utils import Conv1D

from src.model.lora_model_base import LoRAModelBase
from src.model.lora_sequential_model_base import LoRASequentialModelBase
from src.model.svd_lora_model_base import SVDLoRAModelBase
from src.model.svd_lora_sequential_model_base import SVDLoRASequentialModelBase
from src.model.ignoretopk_model_base import IgnoreTopKModelBase
from src.model.gpt2model import GPT2forGeneration


class GPT2LoRA(LoRAModelBase, GPT2forGeneration):
    def __init__(self, gpt2_config, lora_config, **cfg):
        GPT2forGeneration.__init__(self, **gpt2_config)
        LoRAModelBase.__init__(self, lora_config, **cfg)
        
    def check_module(self, name, module):
        return isinstance(module, Conv1D) and name.split('.')[-1] in self.lora_config['target_layers']


class GPT2LoRASequential(LoRASequentialModelBase, GPT2forGeneration):
    def __init__(self, gpt2_config, lora_config, **cfg):
        GPT2forGeneration.__init__(self, **gpt2_config)
        LoRASequentialModelBase.__init__(self, lora_config, **cfg)

    def check_module(self, name, module):
        return isinstance(module, Conv1D) and name.split('.')[-1] in self.lora_config['target_layers']


class GPT2SVDLoRA(SVDLoRAModelBase, GPT2forGeneration):
    def __init__(self, gpt2_config, svd_lora_config, **cfg):
        GPT2forGeneration.__init__(self, **gpt2_config)
        SVDLoRAModelBase.__init__(self, svd_lora_config, **cfg)

    def check_module(self, name, module):
        return isinstance(module, Conv1D) and name.split('.')[-1] in self.lora_config['target_layers']


class GPT2SVDLoRASequential(SVDLoRASequentialModelBase, GPT2forGeneration):
    def __init__(self, gpt2_config, svd_lora_config, **cfg):
        GPT2forGeneration.__init__(self, **gpt2_config)
        SVDLoRASequentialModelBase.__init__(self, svd_lora_config, **cfg)

    def check_module(self, name, module):
        return isinstance(module, Conv1D) and name.split('.')[-1] in self.lora_config['target_layers']


class GPT2IgnoreTopK(IgnoreTopKModelBase, GPT2forGeneration):
    def __init__(self, gpt2_config, ignoretopk_config, output_hidden_states=True, **cfg):
        GPT2forGeneration.__init__(self, output_hidden_states=output_hidden_states, **gpt2_config)
        IgnoreTopKModelBase.__init__(self, **ignoretopk_config, **cfg)
        
    def forward(self, batch):
        outputs = GPT2forGeneration.forward(self, batch)
        return self.ignoretopk_on_forward(outputs)
