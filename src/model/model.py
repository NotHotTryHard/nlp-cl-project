import torch

from torch import nn
from torch.distributions.categorical import Categorical

from src.model.decoder import Decoder

class DecoderModel(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()

        self.model_config = config["model"]["args"]
        
        self.dataset = dataset
        self.max_length = dataset.max_length
        
        self.decoder = Decoder(
            num_layers=self.model_config["num_layers"],
            num_heads=self.model_config["num_heads"],
            embed_dim=self.model_config["embed_dim"],
            feedforward_dim=self.model_config["feedforward_dim"],
            attn_dropout=self.model_config["attn_dropout"],
            ff_dropout=self.model_config["ff_dropout"],
            use_flash_attention=self.model_config["use_flash_attention"],
            vocab_size=dataset.vocab_size,
            max_length=dataset.max_length
        )
        
    def forward(self, x):
        output = self.decoder(x)
        return output

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
       
        self.eval()
        device = next(self.parameters()).device
        tokens = torch.tensor([[self.dataset.bos_id] + self.dataset.text2ids(prefix)]).to(device)

        # 2 stopping conditions: reaching max len or getting <eos> token
        while tokens.shape[1] < self.max_length:
            logits = self.forward(tokens) / temp
            new_token = Categorical(logits=logits[:, -1:]).sample()

            if new_token.item() == self.dataset.eos_id:
                break

            tokens = torch.cat([tokens, new_token], dim=1)

        return self.dataset.ids2text(tokens[:, 1:].squeeze())
