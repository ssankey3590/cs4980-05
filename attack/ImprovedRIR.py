import torch
import torchaudio as ta
import torch.nn as nn
import torch.nn.functional as F
from .attack import Attacker
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pystoi.stoi import stoi


class ImprovedRIR(Attacker):
    def __init__(self, task, config) -> None:
        super(ImprovedRIR, self).__init__(task, config)
        rir, _ = ta.load(config.rir_path)
        rir = rir[0]
        rir = rir / (torch.norm(rir, p=2) + 1e-10)
        rir = rir[rir.argmax():rir.argmax() + self.config.rir_len]
        self.rir = rir.flip(dims=(0,))
        self.window = torch.tukey_window(self.config.overlap * 2, alpha=self.config.tukey_alpha)

    def initialize(self, wav, interval=None):
        assert interval is not None
        self.interval = list(filter(lambda x: x[2] != '' and x[1] - x[0] >= self.config.rir_len, interval))
        self.rir = self.rir.to(wav.device)
        if self.config.mode == 'local':
            self.perturb = [nn.Parameter(self.rir.clone()) for _ in range(len(self.interval))]
            self.optimizer = torch.optim.Adam(self.perturb, lr=self.config.lr)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        else:
            self.perturb = nn.Parameter(self.rir.clone())
            self.optimizer = torch.optim.Adam([self.perturb], lr=self.config.lr)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        self.window = self.window.to(wav.device)
        # Compute phoneme weights
        self.phoneme_weights = []
        for i, (s, e, phoneme) in enumerate(self.interval):
            weight = self.get_phoneme_weight(phoneme)
            self.phoneme_weights.append(weight)

    def normalize(self, x, vmin, vmax):
        return (x - x.min()) / (x.max() - x.min()) * (vmax - vmin) + vmin

    def convolution(self, wav, s, e, rir):
        # Phoneme-level convolution
        s -= self.config.rir_len - 1 + self.config.overlap
        if s < 0:
            seg = F.pad(wav[:, :e], (abs(s), 0))
        else:
            seg = wav[:, s:e]
        return F.conv1d(seg.unsqueeze(0), rir.unsqueeze(0).unsqueeze(0))

    def concatenation(self, segs):
        # Overlap-add concatenation with Tukey window
        new_segs = []
        for k in range(0, len(segs) - 1):
            segs[k][:, :, -self.config.overlap:] = segs[k][:, :, -self.config.overlap:] * self.window[
                                                                                          self.config.overlap:] + \
                                                   segs[k + 1][:, :, :self.config.overlap] * self.window[
                                                                                             :self.config.overlap]
            new_segs.append(segs[k][:, :, self.config.overlap:])
        new_segs.append(segs[-1][:, :, self.config.overlap:])
        if new_segs[0].shape[-1] >= self.config.overlap:
            new_segs[0][:, :, :self.config.overlap] *= self.window[:self.config.overlap]
        if new_segs[-1].shape[-1] >= self.config.overlap:
            new_segs[-1][:, :, -self.config.overlap:] *= self.window[self.config.overlap:]
        return torch.concat(new_segs, dim=-1)

    def generate(self, wav):
        if self.config.mode == 'local':
            wav_ = []
            p = 0
            for i, perturb_i in enumerate(self.perturb):
                s, e = self.interval[i][0], self.interval[i][1]
                if s > p:
                    wav_.append(self.convolution(wav, p, s, self.rir))
                wav_.append(self.convolution(wav, s, e, perturb_i))
                p = e
            if p < wav.shape[-1]:
                wav_.append(self.convolution(wav, p, wav.shape[-1], self.rir))
            wav_ = self.concatenation(wav_).squeeze(0)
        else:
            wav = F.pad(wav, (self.config.rir_len - 1, 0)).unsqueeze(0)
            wav_ = F.conv1d(wav, self.perturb.unsqueeze(0).unsqueeze(0)).squeeze(0)
        wav_ = self.normalize(wav_, wav.min(), wav.max())
        return wav_

    def get_phoneme_weight(self, phoneme):
        less_perceptible = ['k', 't', 'p', 's']  # Example phonemes
        if phoneme in less_perceptible:
            return 1.0  # Higher weight for less perceptible phonemes
        else:
            return 0.5  # Lower weight for more perceptible phonemes

    def perceptual_loss(self, wav, wav_, fs=16000):
        stoi_loss = -stoi(wav.cpu().numpy(), wav_.cpu().detach().numpy(), fs, extended=False)
        return torch.tensor(stoi_loss, device=wav.device)

    def penalty(self, wav, wav_):
        if self.config.mode == 'local':
            rir_loss = 0
            for i, perturb_i in enumerate(self.perturb):
                weight = self.phoneme_weights[i]
                rir_loss += weight * F.mse_loss(self.rir, perturb_i, reduction='sum')
            rir_loss /= len(self.perturb)
        else:
            rir_loss = F.mse_loss(self.rir, self.perturb, reduction='sum')

        perceptual_loss = self.perceptual_loss(wav, wav_)

        # Adaptive weighting example
        if perceptual_loss.item() < self.config.perceptual_threshold:
            self.config.rir_loss_weight *= self.config.weight_decay_factor
        else:
            self.config.rir_loss_weight *= self.config.weight_increase_factor
        self.config.rir_loss_weight = min(max(self.config.rir_loss_weight, self.config.rir_loss_min),
                                          self.config.rir_loss_max)

        total_loss = self.config.wav_loss_weight * perceptual_loss + self.config.rir_loss_weight * rir_loss

        # Update the scheduler
        self.scheduler.step(total_loss)

        return total_loss

    def __str__(self) -> str:
        return f'RIR-{self.config.rir_type}-{self.config.rir_len}-' + str(self.config.alpha)

    def get_perturb(self):
        if self.config.mode == 'local':
            perturbs = []
            for perturb_i in self.perturb:
                perturbs.append(perturb_i.detach().cpu().flip(dims=(0,)))
            return perturbs
        else:
            return self.perturb.detach().cpu().flip(dims=(0,))