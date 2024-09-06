# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class LinearQuantizerModel(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        state_dict = torch.load(ckpt_path, map_location="cpu")

        self.vocab_size = state_dict["model_cfg"]["vocab_size"]
        dim = state_dict["model_cfg"]["dim"]
        upstream_dim = state_dict["model_cfg"]["upstream_dim"]

        out_dim = self.vocab_size + 1  # vocab_size + 1 for blank in CTC
        mid_dim = upstream_dim - out_dim

        self.encoder = nn.Sequential(
            *[
                nn.Linear(dim, dim - mid_dim // 4),
                nn.LeakyReLU(),
                nn.Linear(dim - mid_dim // 4, dim - mid_dim // 2),
                nn.LeakyReLU(),
                nn.Linear(dim - mid_dim // 2, self.vocab_size + 1),
            ]
        )

        self.encoder.load_state_dict(state_dict["model_weight"])

    def forward(self, x):
        logits = self.encoder(x)
        logits = torch.nn.functional.log_softmax(logits, dim=-1)
        code = logits.argmax(dim=-1)

        # post-process units: replace BLANK with most-left non-BLANK units
        non_stop_counter = 0
        while (code == self.vocab_size).any():
            non_stop_counter += 1
            code[code == self.vocab_size] = torch.roll(code, 1)[code == self.vocab_size]
            if non_stop_counter == 10000:
                break

        return code


class KmeansModel(nn.Module):
    def __init__(self, km_path):
        super().__init__()
        states = torch.load(km_path, map_location="cpu")
        assert (
            "cluster_centers" in states and "n_clusters" in states
        ), "Not a valid kmeans checkpoint."
        C_np = states["cluster_centers"].transpose()  # [d_feats, K]
        Cnorm_np = (C_np**2).sum(0, keepdims=True)  # [K,]
        self.K = states["n_clusters"]
        assert self.K == C_np.shape[-1]

        self.C = nn.Parameter(torch.from_numpy(C_np), requires_grad=False)
        self.Cnorm = nn.Parameter(torch.from_numpy(Cnorm_np), requires_grad=False)

    def forward(self, x):
        batched = False
        if len(x.shape) == 3:  # [B, T, d]
            batched = True
            B, T, d = x.shape
            x = x.view(-1, d)

        # x: [T, d]; C: [d, K]; Cnorm: [K,]
        dist = x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
        assigned_clusters = dist.argmin(dim=1)  # [T,]

        if batched:
            assigned_clusters = assigned_clusters.view(B, T)

        return assigned_clusters


def load_quantizer_model(ckpt_path, is_linear_quantizer):
    if is_linear_quantizer:
        model = LinearQuantizerModel(ckpt_path)
        print(f"Loaded LinearQuantizer from '{ckpt_path}'")
    else:
        model = KmeansModel(ckpt_path)
        print(f"Loaded KmeansModel from '{ckpt_path}'")
    return model
