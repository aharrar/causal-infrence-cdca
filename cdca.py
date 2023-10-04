# cdca loss
from math import sqrt

import torch


def get_one_hot_encoding(label, n_classes):
    one_hot_label = torch.zeros(len(label), n_classes).to(label.device)
    one_hot_label[range(len(label)), label.type(torch.int)] = 1

    return one_hot_label


def t_kernel(values):
    kernel_t = 1 / (1 + values)

    return kernel_t


def gaussian_kernel(values, scale):
    scale_fac_i = scale
    kernel_gaussian = torch.exp(-values / (scale_fac_i.unsqueeze(1)))

    return kernel_gaussian


def get_scale_fac(d):
    # when d=pdist(x,y), set discard_diag to True, when d=pdist(x,x), set discard_diag to False.
    [median_row, _] = torch.median(d + torch.max(d) * torch.eye(n=d.shape[0], m=d.shape[1]), dim=1)

    return median_row


def get_weight_matrix(pairwise_distances):
    kernel_t = t_kernel(pairwise_distances)
    k_kernel_matrix = kernel_t
    weight_matrix = torch.diag(1 / (torch.sum(k_kernel_matrix, dim=1) + 1e-5)) @ k_kernel_matrix

    return weight_matrix


def get_cdca_term(src_feature, tgt_feature, src_label, tgt_label
                  ):
    scale = 2 # sqrt(tgt_feature.shape[1])
    attention_s_t = torch.nn.functional.softmax(src_feature @ tgt_feature.t() / scale, dim=-1)
    attention_t_s = torch.nn.functional.softmax(tgt_feature @ src_feature.t() / scale, dim=-1)
    attention_s_s = torch.nn.functional.softmax(src_feature @ src_feature.t() / scale, dim=-1)
    attention_t_t = torch.nn.functional.softmax(tgt_feature @ tgt_feature.t() / scale, dim=-1)

    labels_t_s = attention_t_s @ src_label
    labels_s_t = attention_s_t @ tgt_label
    labels_s_s = attention_s_s @ src_label
    labels_t_t = attention_t_t @ tgt_label

    return labels_t_s, labels_s_t, labels_s_s, labels_t_t
