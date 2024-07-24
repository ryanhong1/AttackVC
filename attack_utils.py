import torch
import torch.nn as nn
from torch import Tensor
from tqdm import trange
import numpy as np

def e2e_attack(
    model: nn.Module,
    vc_src: Tensor,
    vc_tgt: Tensor,
    adv_tgt: Tensor,
    eps: float,
    n_iters,
) -> Tensor:
    ptb = torch.zeros_like(vc_tgt).normal_(0, 1).requires_grad_(True)
    opt = torch.optim.Adam([ptb])
    criterion = nn.MSELoss()
    pbar = trange(n_iters)

    with torch.no_grad():
        org_out = model.inference(vc_src, vc_tgt)
        tgt_out = model.inference(vc_src, adv_tgt)

    for _ in pbar:
        adv_inp = vc_tgt + eps * ptb.tanh()
        adv_out = model.inference(vc_src, adv_inp)
        loss = criterion(adv_out, tgt_out) - 0.1 * criterion(adv_out, org_out)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return vc_tgt + eps * ptb.tanh()


def emb_attack(
    model: nn.Module, vc_tgt: Tensor, adv_tgt: Tensor, eps: float, n_iters: int
) -> Tensor:
    ptb = torch.zeros_like(vc_tgt).normal_(0, 1).requires_grad_(True)
    opt = torch.optim.Adam([ptb])
    criterion = nn.MSELoss()
    pbar = trange(n_iters)

    with torch.no_grad():
        org_emb = model.speaker_encoder(vc_tgt)
        tgt_emb = model.speaker_encoder(adv_tgt)

    for _ in pbar:
        adv_inp = vc_tgt + eps * ptb.tanh()
        adv_emb = model.speaker_encoder(adv_inp)
        loss = criterion(adv_emb, tgt_emb) - 0.1 * criterion(adv_emb, org_emb)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return vc_tgt + eps * ptb.tanh()

def fb_attack(
    model: nn.Module, vc_tgt: Tensor, adv_tgt: Tensor, eps: float, n_iters: int, specto: Tensor
) -> Tensor:
    # Create a mask from specto
    mask = specto.float()  # Convert the binary mask to float tensor

    # Initialize the perturbation with requires_grad=True
    ptb = specto.clone().detach().requires_grad_(True)

    # Create an optimizer for the perturbation
    opt = torch.optim.Adam([ptb])
    criterion = nn.MSELoss()
    pbar = trange(n_iters)

    with torch.no_grad():
        org_emb = model.speaker_encoder(vc_tgt)
        tgt_emb = model.speaker_encoder(adv_tgt)

    for _ in pbar:
        # Apply the mask to ensure only elements where mask == 1 are optimized
        adv_inp = vc_tgt + eps * (ptb * mask)
        adv_emb = model.speaker_encoder(adv_inp)
        loss = criterion(adv_emb, tgt_emb) - 0.1 * criterion(adv_emb, org_emb)

        opt.zero_grad()
        loss.backward()

        # Apply gradients only where mask is 1
        with torch.no_grad():
            ptb.grad *= mask

        opt.step()

    return vc_tgt + eps * (ptb * mask).tanh()