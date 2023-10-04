import numpy as np
import torch

from dataset import inverce_convert, convert


def evaluate_ihdp(net, data, scale_outcome=True):
    x = data['x']
    t = (data['t']).squeeze()
    yf = data['yf']
    ycf = data['ycf']
    mu0 = data['mu_0']
    mu1 = data['mu_1']
    y0_r, y1_r = convert(yf, ycf, t)
    predictions = net.predict(torch.tensor(x))
    predictions = predictions.squeeze().detach().numpy()
    y0 = predictions[:, 0]
    y1 = predictions[:, 1]

    if scale_outcome:
        y0_r = data['y0_scaler'].inverse_transform(y0_r.reshape(-1, 1))
        y1_r = data['y1_scaler'].inverse_transform(y1_r.reshape(-1, 1))

        y0 = data['y0_scaler'].inverse_transform(y0.reshape(-1, 1))
        y1 = data['y1_scaler'].inverse_transform(y1.reshape(-1, 1))

    yf_p, ycf_p = inverce_convert(y0, y1, t)
    yf, ycf = inverce_convert(y0_r, y1_r, t)
    y0,y1 = convert(yf,ycf_p,t)
    prediction_effect = y1 - y0
    real_effect = y1_r - y0_r
    effect_ate = np.abs(np.mean(real_effect) - np.mean(prediction_effect))
    effect_pehe = np.sqrt(np.mean(np.square(real_effect - prediction_effect)))

    rmse_fact = np.sqrt(np.mean(np.square(yf_p - yf)))
    rmse_cfact = np.sqrt(np.mean(np.square(ycf_p - ycf)))

    eff_pred = ycf_p - yf_p
    eff_pred[t > 0] = -eff_pred[t > 0]

    ite_pred = ycf_p - yf
    ite_pred[t > 0] = -ite_pred[t > 0]
    rmse_ite = np.sqrt(np.mean(np.square(ite_pred - real_effect)))

    ate_pred = np.mean(eff_pred)
    bias_ate = ate_pred - np.mean(real_effect)

    att_pred = np.mean(eff_pred[t > 0])
    bias_att = att_pred - np.mean(real_effect[t > 0])

    atc_pred = np.mean(eff_pred[t < 1])
    bias_atc = atc_pred - np.mean(real_effect[t < 1])

    pehe = np.sqrt(np.mean(np.square(eff_pred - real_effect)))

    return {'ate_pred': ate_pred, 'att_pred': att_pred,
            'atc_pred': atc_pred, 'bias_ate': bias_ate,
            'bias_att': bias_att, 'bias_atc': bias_atc,
            'rmse_fact': rmse_fact, 'rmse_cfact': rmse_cfact,
            "effect_pehe": effect_pehe,
            'pehe': pehe, 'effect_ate': effect_ate, 'rmse_ite': rmse_ite}
    # 'policy_value': policy_value, 'policy_curve': policy_curve}
