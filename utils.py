import torch
from geomloss import SamplesLoss
from torch import nn

from cdca import get_cdca_term


def loss_shaham(h0, h1, features, y, treatments,p):
    mask0 = (treatments == 0)
    mask1 = (treatments == 1)
    # cirection1 = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, backend="tensorized")

    cirection2 = nn.L1Loss()
    cirection1 = nn.MSELoss()

    labels_t_s, labels_s_t, labels_s_s, labels_t_t = get_cdca_term(features[mask0], features[mask1], y[mask0], y[mask1])
    self_loss = cirection2(labels_s_s, y[mask0]) + cirection1(
        labels_t_t, y[mask1])
    loss_t0 = cirection1(h0[mask0], y[mask0])
    loss_t0_cdca = cirection2(labels_s_t, h1[mask0])

    loss_t1 = cirection1(h1[mask1], y[mask1])
    loss_t1_cdca = cirection2(labels_t_s, h0[mask1])

    # p = torch.sum(mask1) / treatments.shape[0]

    loss =(1.0 - p) * (loss_t0 + loss_t0_cdca) + (p) * (loss_t1 + loss_t1_cdca) + self_loss
    loss_value = (1.0 - p) * (loss_t0 + loss_t0_cdca) + p * (loss_t1 + loss_t1_cdca) + self_loss
    return loss, loss_value


def mmd(x, y, mmd_kernel_bandwidth=[0.1, 0.5, 1, 2]):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """

    nx = x.shape[0]
    ny = y.shape[0]
    dxx = torch.cdist(x, x) ** 2  # (x_i-x_j)^2
    dyy = torch.cdist(y, y) ** 2  # (y_i-y_j)^2
    dxy = torch.cdist(x, y) ** 2  # (x_i-y_j)^2
    device = x.device if x.is_cuda else torch.device("cpu")
    XX, YY, XY = (torch.zeros(dxx.shape).to(device),
                  torch.zeros(dyy.shape).to(device),
                  torch.zeros(dxy.shape).to(device))

    for a in mmd_kernel_bandwidth:
        XX += torch.exp(-0.5 * dxx / a)
        YY += torch.exp(-0.5 * dyy / a)
        XY += torch.exp(-0.5 * dxy / a)

    return torch.sum(XX) / (nx ** 2) + torch.sum(YY) / (ny ** 2) - 2. * torch.sum(XY) / (nx * ny)


def loss_shalit(h0, h1, features, y, treatments,p, alpha=10, blur=0.05):
    mask0 = (treatments == 0)
    mask1 = (treatments == 1)
    yf_p = torch.zeros_like(y)
    yf_p[mask0] = h0[mask0]
    yf_p[mask1] = h1[mask1]

    # yf_p,ycf_p = inverce_convert(h0, h1, treatments)
    cirection = nn.MSELoss(reduction='none')
    # loss_t0 = cirection(h0[mask0], y[mask0])
    # loss_t1 = cirection(h1[mask1], y[mask1])
    """Imbalance loss"""
    # p = torch.mean(treatments)
    w_t = treatments / (2 * p)
    w_c = (1 - treatments) / (2 * (1 - p))
    sampeles_weights = w_t + w_c
    loss_rec = torch.mean(sampeles_weights * cirection(yf_p, y))
    samples_loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur, backend="tensorized")
    phi0 = features[mask0]
    phi1 = features[mask1]
    imbalance_loss = samples_loss(phi1, phi0)

    loss = loss_rec / loss_rec.item() + alpha * imbalance_loss / imbalance_loss.item()

    loss_value = loss_rec + alpha * imbalance_loss
    return loss, loss_value
