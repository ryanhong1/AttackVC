import torch
import torch.nn as nn
from torch import Tensor
from tqdm import trange
import numpy as np

loaded_array = np.load('./226_001_norm.npy')

lrp = torch.tensor(loaded_array)
print("lrp:")
print(lrp.shape)



def e2e_attack(
    model: nn.Module,
    vc_src: Tensor,
    vc_tgt: Tensor,
    adv_tgt: Tensor,
    eps: float,
    n_iters,
) -> Tensor:
    ptb = torch.zeros_like(vc_tgt).normal_(0, 1).requires_grad_(True) #vc_tgt와 같은 모양의 tensor를 0과 1사이 random 채움.
    opt = torch.optim.Adam([ptb])
    criterion = nn.MSELoss()
    pbar = trange(n_iters)

    with torch.no_grad():
        org_out = model.inference(vc_src, vc_tgt)#vc_src = t는 content 제공 voice.
        tgt_out = model.inference(vc_src, adv_tgt)

    for _ in pbar:
        adv_inp = vc_tgt + eps * ptb.tanh() #tanh bounds perturbation between -1 and 1, keeping it small
        adv_out = model.inference(vc_src, adv_inp)
        loss = criterion(adv_out, tgt_out) - 0.1 * criterion(adv_out, org_out)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return vc_tgt + eps * ptb.tanh()


def emb_attack(#only speaker encoder is involved -> more efficient
    model: nn.Module, vc_tgt: Tensor, adv_tgt: Tensor, eps: float, n_iters: int
) -> Tensor:
    ptb = torch.zeros_like(vc_tgt).normal_(0, 1).requires_grad_(True) 
    opt = torch.optim.Adam([ptb])
    criterion = nn.MSELoss()
    pbar = trange(n_iters)

    with torch.no_grad():#temporarily disable gradient calculation as you won't need it
        org_emb = model.speaker_encoder(vc_tgt)#the voice you want to copy in vc(voice with vocal timbre). you gotta defend this.E(x)
        tgt_emb = model.speaker_encoder(adv_tgt)#adversarial target. E(y))
    #vc : x의 목소리와 y의 content
    

    for _ in pbar: #parameter : ptb
        adv_inp = vc_tgt + eps * ptb.tanh()#x+d
        adv_emb = model.speaker_encoder(adv_inp)#F(t,x+d)
        loss = criterion(adv_emb, tgt_emb) - 0.1 * criterion(adv_emb, org_emb) #lambda = 0.1. x+d와 y는 가까워지고 x+d와 x는 멀어짐.
            #
            #뒷부분 : x+d가 x와 달라지게 함 -> original과 달라지면서 혼란
        opt.zero_grad()
        loss.backward()
        opt.step()

    perturbation = eps*ptb.tanh()
    np.save('./ptb.npy', perturbation.detach().numpy())
    return vc_tgt + eps * ptb.tanh()

def fb_attack(#only speaker encoder is involved -> more efficient
    model: nn.Module, vc_tgt: Tensor, adv_tgt: Tensor, eps: float, n_iters: int
) -> Tensor:
    ptb = lrp
    opt = torch.optim.Adam([ptb])
    criterion = nn.MSELoss()
    pbar = trange(n_iters)

    with torch.no_grad():#temporarily disable gradient calculation as you won't need it
        org_emb = model.speaker_encoder(vc_tgt)#the voice you want to copy in vc(voice with vocal timbre). you gotta defend this.E(x)
        tgt_emb = model.speaker_encoder(adv_tgt)#adversarial target. E(y))
    #vc : x의 목소리와 y의 content
    

    for _ in pbar: #parameter : ptb
        adv_inp = vc_tgt + eps * ptb#x+d
        adv_emb = model.speaker_encoder(adv_inp)#F(t,x+d)
        loss = criterion(adv_emb, tgt_emb) - 0.1 * criterion(adv_emb, org_emb) #lambda = 0.1. x+d와 y는 가까워지고 x+d와 x는 멀어짐.
        opt.zero_grad()
        loss.backward()
        opt.step()

    perturbation = eps*ptb.tanh()
    np.save('./ptb_lrp.npy', perturbation.detach().numpy())
    return vc_tgt + eps * ptb.tanh()