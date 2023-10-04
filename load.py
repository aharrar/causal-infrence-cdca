import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from dataset import convert, inverce_convert, IhdpDataset


def get_train_and_test(data_path, i=2):
    data = load_ihdp_data(training_data=f'{data_path}/ihdp_npci_1-1000.train.npz',
                          testing_data=f'{data_path}/ihdp_npci_1-1000.test.npz', i=i)
    ihdp_dataset = IhdpDataset(data['x'], data['t'], data['yf'], data['mu_0'], data['mu_1'], data['ycf'])
    trainset, testset = torch.utils.data.random_split(ihdp_dataset,
                                                      [len(ihdp_dataset) - int(len(
                                                          ihdp_dataset) * 0.3),
                                                       int(len(ihdp_dataset) * 0.3)])

    return data, trainset, testset


def load_ihdp_data(training_data, testing_data, sacle_outcome=True, i=7):
    with open(training_data, 'rb') as trf, open(testing_data, 'rb') as tef:
        train_data = np.load(trf);
        test_data = np.load(tef)
        ycf = np.concatenate((train_data['ycf'][:, i], test_data['ycf'][:, i])).astype(
            'float32')  # most GPUs only compute 32-bit floats
        # print(test_data['ycf'])
        yf = np.concatenate((train_data['yf'][:, i], test_data['yf'][:, i])).astype(
            'float32')  # most GPUs only compute 32-bit floats
        t = np.concatenate((train_data['t'][:, i], test_data['t'][:, i])).astype('float32')
        x = np.concatenate((train_data['x'][:, :, i], test_data['x'][:, :, i]), axis=0).astype('float32')
        mu_0 = np.concatenate((train_data['mu0'][:, i], test_data['mu0'][:, i])).astype('float32')
        mu_1 = np.concatenate((train_data['mu1'][:, i], test_data['mu1'][:, i])).astype('float32')

        data = {'x': x, 't': t, 'mu_0': mu_0, 'mu_1': mu_1}
        t = t.reshape(-1, 1)  # we're just padding one dimensional vectors with an additional dimension
        yf = yf.reshape(-1, 1)
        ycf = ycf.reshape(-1, 1)
        data['y_0'], data['y_1'] = convert(yf, ycf, t)
        # rescaling y between 0 and 1 often makes training of DL regressors easier
        data['yf'] = yf
        data['ycf'] = ycf
        data['y0_scaler'] = StandardScaler().fit(yf)
        data['y1_scaler'] = StandardScaler().fit(yf)

        if sacle_outcome:
            data['y_0'] = data['y0_scaler'].transform(data['y_0'])
            data['y_1'] = data['y1_scaler'].transform(data['y_1'])
            data['yf_origin'] = data['yf']
            data['ycf_origin'] = data['ycf']
            data['yf'], data['ycf'] = inverce_convert(data['y_0'], data['y_1'], t)

    return data
