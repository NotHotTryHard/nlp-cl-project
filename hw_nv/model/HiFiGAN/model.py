import torch

from torch import nn

from hw_nv.model.HiFiGAN.generator import Generator
from hw_nv.model.HiFiGAN.period_discriminator import MultiPeriodDiscriminator
from hw_nv.model.HiFiGAN.scale_discriminator import MultiScaleDiscriminator


class HiFiGANModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = config["model"]

        self.generator = Generator(self.model_config)
        self.MPD = MultiPeriodDiscriminator(self.model_config)
        self.MSD = MultiScaleDiscriminator(self.model_config)
    
    def forward(self, **batch):
        return self.generator(batch["mel"])

    def discriminate(self, **batch):
        result = {"true": {}, "gen": {}}

        for input_type, input_name in zip(["true", "gen"], ["wav", "wav_gen"]):
            for D_name, D in zip(["MPD", "MSD"], [self.MPD, self.MSD]):
                outputs, layer_outputs = D(batch[input_name])
                result[input_type][f"{D_name}_outputs"] = outputs
                result[input_type][f"{D_name}_layer_outputs"] = layer_outputs

        return result
