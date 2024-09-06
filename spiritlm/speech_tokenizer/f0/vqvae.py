# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# VQ-VAE model, adapted from:
# - https://github.com/openai/jukebox/blob/master/jukebox/vqvae/
# - https://github.com/facebookresearch/speech-resynthesis/blob/main/modules/vq.py


import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf


def load_vqvae(checkpoint):
    config = Path(checkpoint).parent / "config.yaml"
    cfg = OmegaConf.load(config)
    model = VQVAE(cfg)
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    model.eval()
    print(f"VQVAE model loaded from '{checkpoint}'!")
    return model, cfg


class VQVAE(nn.Module):
    def __init__(self, h):
        super().__init__()

        self.encoder = Encoder(**h.encoder_params)
        self.vq = Bottleneck(**h.vq_params)
        self.decoder = Decoder(**h.decoder_params)

    def forward(self, x, compute_pred=False):
        with torch.no_grad():
            z = self.encoder(x)
            codes, z_q, commit_losses, metrics = self.vq(z)

            if not compute_pred:
                return codes
            x_hat = self.decoder(z_q)

            return codes, x_hat


class BottleneckBlock(nn.Module):
    def __init__(self, k_bins, emb_width, mu):
        super().__init__()
        self.k_bins = k_bins
        self.emb_width = emb_width
        self.mu = mu
        self.reset_k()
        self.threshold = 1.0

    def reset_k(self):
        self.init = False
        self.k_sum = None
        self.k_elem = None
        self.register_buffer("k", torch.zeros(self.k_bins, self.emb_width))

    def _tile(self, x):
        d, ew = x.shape
        if d < self.k_bins:
            n_repeats = (self.k_bins + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def init_k(self, x):
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
        self.init = True
        # init k_w using random vectors from x
        y = self._tile(x)
        _k_rand = y[torch.randperm(y.shape[0])][:k_bins]
        dist.broadcast(_k_rand, 0)
        self.k = _k_rand
        assert self.k.shape == (k_bins, emb_width)
        self.k_sum = self.k
        self.k_elem = torch.ones(k_bins, device=self.k.device)

    def restore_k(self, num_tokens=None, threshold=1.0):
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
        self.init = True
        assert self.k.shape == (k_bins, emb_width)
        self.k_sum = self.k.clone()
        self.k_elem = torch.ones(k_bins, device=self.k.device)
        if num_tokens is not None:
            expected_usage = num_tokens / k_bins
            self.k_elem.data.mul_(expected_usage)
            self.k_sum.data.mul_(expected_usage)
        self.threshold = threshold

    def update_k(self, x, x_l):
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
        with torch.no_grad():
            # Calculate new centres
            x_l_onehot = torch.zeros(
                k_bins, x.shape[0], device=x.device
            )  # k_bins, N * L
            x_l_onehot.scatter_(0, x_l.view(1, x.shape[0]), 1)

            _k_sum = torch.matmul(x_l_onehot, x)  # k_bins, w
            _k_elem = x_l_onehot.sum(dim=-1)  # k_bins
            y = self._tile(x)
            _k_rand = y[torch.randperm(y.shape[0])][:k_bins]

            dist.broadcast(_k_rand, 0)
            dist.all_reduce(_k_sum)
            dist.all_reduce(_k_elem)

            # Update centres
            old_k = self.k
            self.k_sum = mu * self.k_sum + (1.0 - mu) * _k_sum  # w, k_bins
            self.k_elem = mu * self.k_elem + (1.0 - mu) * _k_elem  # k_bins
            usage = (self.k_elem.view(k_bins, 1) >= self.threshold).float()
            self.k = (
                usage
                * (self.k_sum.view(k_bins, emb_width) / self.k_elem.view(k_bins, 1))
                + (1 - usage) * _k_rand
            )
            _k_prob = _k_elem / torch.sum(
                _k_elem
            )  # x_l_onehot.mean(dim=-1)  # prob of each bin
            entropy = -torch.sum(
                _k_prob * torch.log(_k_prob + 1e-8)
            )  # entropy ie how diverse
            used_curr = (_k_elem >= self.threshold).sum()
            usage = torch.sum(usage)
            dk = torch.norm(self.k - old_k) / np.sqrt(np.prod(old_k.shape))
        return dict(entropy=entropy, used_curr=used_curr, usage=usage, dk=dk)

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  # x_en = (N * L, w), k_j = (w, k_bins)

        if x.shape[-1] == self.emb_width:
            prenorm = torch.norm(x - torch.mean(x)) / np.sqrt(np.prod(x.shape))
        elif x.shape[-1] == 2 * self.emb_width:
            x1, x2 = x[..., : self.emb_width], x[..., self.emb_width :]
            prenorm = (torch.norm(x1 - torch.mean(x1)) / np.sqrt(np.prod(x1.shape))) + (
                torch.norm(x2 - torch.mean(x2)) / np.sqrt(np.prod(x2.shape))
            )

            # Normalise
            x = x1 + x2
        else:
            assert False, f"Expected {x.shape[-1]} to be (1 or 2) * {self.emb_width}"
        return x, prenorm

    def postprocess(self, x_l, x_d, x_shape):
        # [NT, C] -> NTC -> NCT
        N, T = x_shape
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        x_l = x_l.view(N, T)
        return x_l, x_d

    def quantise(self, x):
        # Calculate latent code x_l
        k_w = self.k.t()
        distance = (
            torch.sum(x**2, dim=-1, keepdim=True)
            - 2 * torch.matmul(x, k_w)
            + torch.sum(k_w**2, dim=0, keepdim=True)
        )  # (N * L, b)
        min_distance, x_l = torch.min(distance, dim=-1)
        fit = torch.mean(min_distance)
        return x_l, fit

    def dequantise(self, x_l):
        x = F.embedding(x_l, self.k)
        return x

    def encode(self, x):
        N, width, T = x.shape

        # Preprocess.
        x, prenorm = self.preprocess(x)

        # Quantise
        x_l, fit = self.quantise(x)

        # Postprocess.
        x_l = x_l.view(N, T)
        return x_l

    def decode(self, x_l):
        N, T = x_l.shape
        width = self.emb_width

        # Dequantise
        x_d = self.dequantise(x_l)

        # Postprocess
        x_d = x_d.view(N, T, width).permute(0, 2, 1).contiguous()
        return x_d

    def forward(self, x, update_k=True):
        N, width, T = x.shape

        # Preprocess
        x, prenorm = self.preprocess(x)

        # Init k if not inited
        if update_k and not self.init:
            self.init_k(x)

        # Quantise and dequantise through bottleneck
        x_l, fit = self.quantise(x)
        x_d = self.dequantise(x_l)

        # Update embeddings
        if update_k and self.training:
            update_metrics = self.update_k(x, x_l)
        else:
            update_metrics = {}

        # Loss
        commit_loss = torch.norm(x_d.detach() - x) ** 2 / np.prod(x.shape)

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_l, x_d = self.postprocess(x_l, x_d, (N, T))
        return x_l, x_d, commit_loss, dict(fit=fit, pn=prenorm, **update_metrics)


class Bottleneck(nn.Module):
    def __init__(self, l_bins, emb_width, mu, levels):
        super().__init__()
        self.levels = levels
        level_block = lambda level: BottleneckBlock(l_bins, emb_width, mu)
        self.level_blocks = nn.ModuleList()
        for level in range(self.levels):
            self.level_blocks.append(level_block(level))

    def encode(self, xs):
        zs = [level_block.encode(x) for (level_block, x) in zip(self.level_blocks, xs)]
        return zs

    def decode(self, zs, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        xs_quantised = [
            level_block.decode(z)
            for (level_block, z) in zip(self.level_blocks[start_level:end_level], zs)
        ]
        return xs_quantised

    def forward(self, xs):
        zs, xs_quantised, commit_losses, metrics = [], [], [], []
        for level in range(self.levels):
            level_block = self.level_blocks[level]
            x = xs[level]
            z, x_quantised, commit_loss, metric = level_block(x, update_k=self.training)
            zs.append(z)
            if not self.training:
                # Be extra paranoid and make sure the encoder weights can't
                # change from straight-through estimator
                x_quantised = x_quantised.detach()
            xs_quantised.append(x_quantised)
            commit_losses.append(commit_loss)
            if self.training:
                metrics.append(metric)
        return zs, xs_quantised, commit_losses, metrics


class ResConvBlock(nn.Module):
    def __init__(self, n_in, n_state):
        super().__init__()
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(n_in, n_state, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(n_state, n_in, 1, 1, 0),
        )

    def forward(self, x):
        return x + self.model(x)


class Resnet(nn.Module):
    def __init__(self, n_in, n_depth, m_conv=1.0):
        super().__init__()
        self.model = nn.Sequential(
            *[ResConvBlock(n_in, int(m_conv * n_in)) for _ in range(n_depth)]
        )

    def forward(self, x):
        return self.model(x)


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, zero_out=False, res_scale=1.0):
        super().__init__()
        padding = dilation
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(n_in, n_state, 3, 1, padding, dilation),
            nn.ReLU(),
            nn.Conv1d(n_state, n_in, 1, 1, 0),
        )
        if zero_out:
            out = self.model[-1]
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.model(x)


class Resnet1D(nn.Module):
    def __init__(
        self,
        n_in,
        n_depth,
        m_conv=1.0,
        dilation_growth_rate=1,
        dilation_cycle=None,
        zero_out=False,
        res_scale=False,
        reverse_dilation=False,
        checkpoint_res=False,
    ):
        super().__init__()

        def _get_depth(depth):
            if dilation_cycle is None:
                return depth
            else:
                return depth % dilation_cycle

        blocks = [
            ResConv1DBlock(
                n_in,
                int(m_conv * n_in),
                dilation=dilation_growth_rate ** _get_depth(depth),
                zero_out=zero_out,
                res_scale=1.0 if not res_scale else 1.0 / math.sqrt(n_depth),
            )
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.checkpoint_res = checkpoint_res
        if self.checkpoint_res == 1:
            if dist.get_rank() == 0:
                print("Checkpointing convs")
            self.blocks = nn.ModuleList(blocks)
        else:
            self.model = nn.Sequential(*blocks)

    def forward(self, x):
        if self.checkpoint_res == 1:
            raise NotImplementedError("Checkpoint not implemented")
        else:
            return self.model(x)


def assert_shape(x, exp_shape):
    assert x.shape == exp_shape, f"Expected {exp_shape} got {x.shape}"


class EncoderConvBlock(nn.Module):
    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        down_t,
        stride_t,
        width,
        depth,
        m_conv,
        dilation_growth_rate=1,
        dilation_cycle=None,
        zero_out=False,
        res_scale=False,
    ):
        super().__init__()
        blocks = []
        if type(stride_t) is tuple or type(stride_t) is list:
            start = True
            for s_t, d_t in zip(stride_t, down_t):
                if s_t % 2 == 0:
                    filter_t, pad_t = s_t * 2, s_t // 2
                else:
                    filter_t, pad_t = s_t * 2 + 1, s_t // 2 + 1
                if d_t > 0:
                    for i in range(d_t):
                        block = nn.Sequential(
                            nn.Conv1d(
                                input_emb_width if i == 0 and start else width,
                                width,
                                filter_t,
                                s_t,
                                pad_t,
                            ),
                            Resnet1D(
                                width,
                                depth,
                                m_conv,
                                dilation_growth_rate,
                                dilation_cycle,
                                zero_out,
                                res_scale,
                            ),
                        )
                        blocks.append(block)
                        start = False
            block = nn.Conv1d(width, output_emb_width, 3, 1, 1)
            blocks.append(block)
        else:
            filter_t, pad_t = stride_t * 2, stride_t // 2
            if down_t > 0:
                for i in range(down_t):
                    block = nn.Sequential(
                        nn.Conv1d(
                            input_emb_width if i == 0 else width,
                            width,
                            filter_t,
                            stride_t,
                            pad_t,
                        ),
                        Resnet1D(
                            width,
                            depth,
                            m_conv,
                            dilation_growth_rate,
                            dilation_cycle,
                            zero_out,
                            res_scale,
                        ),
                    )
                    blocks.append(block)
                block = nn.Conv1d(width, output_emb_width, 3, 1, 1)
                blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class DecoderConvBock(nn.Module):
    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        down_t,
        stride_t,
        width,
        depth,
        m_conv,
        dilation_growth_rate=1,
        dilation_cycle=None,
        zero_out=False,
        res_scale=False,
        reverse_decoder_dilation=False,
        checkpoint_res=False,
    ):
        super().__init__()
        blocks = []

        if type(stride_t) is tuple or type(stride_t) is list:
            block = nn.Conv1d(output_emb_width, width, 3, 1, 1)
            blocks.append(block)
            for k, (s_t, d_t) in enumerate(zip(stride_t, down_t)):
                if d_t > 0:
                    if s_t % 2 == 0:
                        filter_t, pad_t = s_t * 2, s_t // 2
                    else:
                        filter_t, pad_t = s_t * 2 + 1, s_t // 2 + 1
                    end = k == len(stride_t) - 1
                    for i in range(d_t):
                        block = nn.Sequential(
                            Resnet1D(
                                width,
                                depth,
                                m_conv,
                                dilation_growth_rate,
                                dilation_cycle,
                                zero_out=zero_out,
                                res_scale=res_scale,
                                reverse_dilation=reverse_decoder_dilation,
                                checkpoint_res=checkpoint_res,
                            ),
                            nn.ConvTranspose1d(
                                width,
                                input_emb_width if i == (d_t - 1) and end else width,
                                filter_t,
                                s_t,
                                pad_t,
                            ),
                        )
                        blocks.append(block)
        else:
            if down_t > 0:
                filter_t, pad_t = stride_t * 2, stride_t // 2
                block = nn.Conv1d(output_emb_width, width, 3, 1, 1)
                blocks.append(block)
                for i in range(down_t):
                    block = nn.Sequential(
                        Resnet1D(
                            width,
                            depth,
                            m_conv,
                            dilation_growth_rate,
                            dilation_cycle,
                            zero_out=zero_out,
                            res_scale=res_scale,
                            reverse_dilation=reverse_decoder_dilation,
                            checkpoint_res=checkpoint_res,
                        ),
                        nn.ConvTranspose1d(
                            width,
                            input_emb_width if i == (down_t - 1) else width,
                            filter_t,
                            stride_t,
                            pad_t,
                        ),
                    )
                    blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        levels,
        downs_t,
        strides_t,
        **block_kwargs,
    ):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t

        block_kwargs_copy = dict(**block_kwargs)
        if "reverse_decoder_dilation" in block_kwargs_copy:
            del block_kwargs_copy["reverse_decoder_dilation"]
        level_block = lambda level, down_t, stride_t: EncoderConvBlock(
            input_emb_width if level == 0 else output_emb_width,
            output_emb_width,
            down_t,
            stride_t,
            **block_kwargs_copy,
        )
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

    def forward(self, x):
        N, T = x.shape[0], x.shape[-1]
        emb = self.input_emb_width
        assert_shape(x, (N, emb, T))
        xs = []

        # 64, 32, ...
        iterator = zip(list(range(self.levels)), self.downs_t, self.strides_t)
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            if type(stride_t) is tuple or type(stride_t) is list:
                emb, T = self.output_emb_width, T // np.prod(
                    [s**d for s, d in zip(stride_t, down_t)]
                )
            else:
                emb, T = self.output_emb_width, T // (stride_t**down_t)
            assert_shape(x, (N, emb, T))
            xs.append(x)

        return xs


class Decoder(nn.Module):
    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        levels,
        downs_t,
        strides_t,
        **block_kwargs,
    ):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels

        self.downs_t = downs_t

        self.strides_t = strides_t

        level_block = lambda level, down_t, stride_t: DecoderConvBock(
            output_emb_width, output_emb_width, down_t, stride_t, **block_kwargs
        )
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

        self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1)

    def forward(self, xs, all_levels=True):
        if all_levels:
            assert len(xs) == self.levels
        else:
            assert len(xs) == 1
        x = xs[-1]
        N, T = x.shape[0], x.shape[-1]
        emb = self.output_emb_width
        assert_shape(x, (N, emb, T))

        # 32, 64 ...
        iterator = reversed(
            list(zip(list(range(self.levels)), self.downs_t, self.strides_t))
        )
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            if type(stride_t) is tuple or type(stride_t) is list:
                emb, T = self.output_emb_width, T * np.prod(
                    [s**d for s, d in zip(stride_t, down_t)]
                )
            else:
                emb, T = self.output_emb_width, T * (stride_t**down_t)
            assert_shape(x, (N, emb, T))
            if level != 0 and all_levels:
                x = x + xs[level - 1]

        x = self.out(x)
        return x
