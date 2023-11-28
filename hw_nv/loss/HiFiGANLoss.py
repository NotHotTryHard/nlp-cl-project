import torch
import torch.nn.functional as F

from itertools import chain
from torch import nn


class HiFiGANGeneratorLoss(nn.Module):
    def __init__(self, mel_generator, alpha_fm_loss=2, alpha_mel_loss=45):
        super().__init__()
        self.mel_generator = mel_generator
        self.mel_loss = nn.L1Loss()
        self.fm_loss = nn.L1Loss()

        self.alpha_fm_loss = alpha_fm_loss
        self.alpha_mel_loss = alpha_mel_loss
    
    def forward(self, wav_gen, mel, length, D_outputs, **batch):
        mpd_true_layer_outputs = D_outputs["true"]["MPD_layer_outputs"]
        msd_true_layer_outputs = D_outputs["true"]["MSD_layer_outputs"]
        
        mpd_gen_outputs = D_outputs["gen"]["MPD_outputs"]
        msd_gen_outputs = D_outputs["gen"]["MSD_outputs"]
        mpd_gen_layer_outputs = D_outputs["gen"]["MPD_layer_outputs"]
        msd_gen_layer_outputs = D_outputs["gen"]["MSD_layer_outputs"]
        
        GAN_loss = sum(
            torch.mean((output - 1) ** 2)
            for output in chain(mpd_gen_outputs, msd_gen_outputs)
        )

        mel_generated = self.mel_generator(wav_gen)
        generated_length = mel_generated.shape[-1]
        mel, mel_generated = self.pad_mels(mel, mel_generated, length, generated_length)
        mel_loss = self.alpha_mel_loss * self.mel_loss(mel, mel_generated)

        pmd_fm_loss = sum(
            self.fm_loss(true_feat, true_feat)
            for true_feat, true_feat in zip(mpd_true_layer_outputs, mpd_gen_layer_outputs)
        )
        psd_fm_loss = sum(
            self.fm_loss(true_feat, true_feat)
            for true_feat, true_feat in zip(msd_true_layer_outputs, msd_gen_layer_outputs)
        )
        fm_loss = self.alpha_fm_loss * (pmd_fm_loss + psd_fm_loss)

        return GAN_loss + mel_loss + fm_loss, GAN_loss, mel_loss, fm_loss


    def pad_mels(self, mel1, mel2, length1, length2):
        length_diff = length2 - length1

        silence_value = self.mel_generator.config.pad_value
        mel1 = F.pad(mel1, (0, 0, max(0, length_diff)), 'constant', silence_value)
        mel2 = F.pad(mel2, (0, 0, max(0, -length_diff)), 'constant', silence_value)

        return mel1, mel2


class HiFiGANDiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, D_outputs, **batch):
        mpd_true_outputs = D_outputs["true"]["MPD_outputs"]
        msd_true_outputs = D_outputs["true"]["MSD_outputs"]
        
        mpd_gen_outputs = D_outputs["gen"]["MPD_outputs"]
        msd_gen_outputs = D_outputs["gen"]["MSD_outputs"]

        mpd_loss = sum(
            torch.mean((true_prob - 1) ** 2) + torch.mean((gen_prob - 1) ** 2)
            for true_prob, gen_prob in zip(mpd_true_outputs, mpd_gen_outputs)
        )
        msd_loss = sum(
            torch.mean((true_prob - 1) ** 2) + torch.mean((gen_prob - 1) ** 2)
            for true_prob, gen_prob in zip(msd_true_outputs, msd_gen_outputs)
        )

        return mpd_loss + msd_loss, mpd_loss, msd_loss

