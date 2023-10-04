from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
from evaluate import evaluate_ihdp
from load import get_train_and_test
from train import fit


if __name__ == '__main__':

    """Simple config for single set of hyperparameters with suggestions for setting up the tune hyper-parameter search"""
    config = {
        "Method": 'BITES',  # or 'ITES', 'DeepSurvT', 'DeepSurv', 'CFRNet'
        "trial_name": 'RGBSG',
        "input_dim": 25,  # name of your trial
        "lr": 0.1,  # Nuber of covariates in the data
        "shared_layer": [30, 30, 30, 30, 10],  # or just tune.grid_search([<list of lists>])
        "individual_layer": [30, 30],  # or just tune.grid_search([<list of lists>])
        "dropout": 0.2,  # or tune.choice([<list values>])
        "weight_decay": 0.2,  # or tune.choice([<list values>])
        "batch_size": 128,  # or tune.choice([<list values>])
        "epochs": 200,
        "alpha": 0.2,  # or tune.grid_search([<list values>])
        "blur": 0.05,  # or tune.grid_search([<list values>]),
    }

    cfact_baseline_train_mean = []
    cfact_shalit_train_mean = []
    cfact_shaham_train_mean = []

    cfact_baseline_train_std = []
    cfact_shalit_train_std = []
    cfact_shaham_train_std = []

    ate_baseline_train_mean = []
    ate_shalit_train_mean = []
    ate_shaham_train_mean = []

    ate_baseline_train_std = []
    ate_shalit_train_std = []
    ate_shaham_train_std = []

    shalit_train = []
    shalit_test = []
    shaham_train = []
    shaham_test = []
    baseline_train = []
    baseline_test = []
    num_of_seed = 9
    data_path = Path("data")
    for i in tqdm(range(0, 100)):
        cfact_baseline_train_seed = np.zeros(num_of_seed)
        cfact_shalit_train_seed = np.zeros(num_of_seed)
        cfact_shaham_train_seed = np.zeros(num_of_seed)

        ate_baseline_train_seed = np.zeros(num_of_seed)
        ate_shalit_train_seed = np.zeros(num_of_seed)
        ate_shaham_train_seed = np.zeros(num_of_seed)

        for j in range(num_of_seed):
            torch.manual_seed(j)
            np.random.seed(j)
            random.seed(j)

            data, trainset, testset = get_train_and_test(data_path=data_path, i=i)
            ihdp_dataloader = DataLoader(dataset=trainset, batch_size=config["batch_size"], shuffle=True,
                                         drop_last=True)
            _, t, _, _, _, _ = trainset[:]
            p = torch.mean(torch.tensor(t))
            ### our loss
            net, train_loss_list = fit(config, ihdp_dataloader,p, type_loss="shaham")
            evaluate_dict_shaham = evaluate_ihdp(net, data)
            # plot1(train_loss_list,"loss")
            ###3 shalit loss
            config["alpha"] = 0.5
            net, train_loss_list = fit(config, ihdp_dataloader,p, type_loss="shalit")
            evaluate_dict_shalit = evaluate_ihdp(net, data)
            shalit_train.append(evaluate_dict_shalit)
            ## train baseline
            config["alpha"] = 0
            net, train_loss_list = fit(config, ihdp_dataloader,p, type_loss="shalit")
            evaluate_dict_baseline = evaluate_ihdp(net, data)
            baseline_train.append(evaluate_dict_baseline)

            cfact_baseline_train_seed[j] = evaluate_dict_baseline['effect_pehe']
            cfact_shalit_train_seed[j] = evaluate_dict_shalit['effect_pehe']
            cfact_shaham_train_seed[j] = evaluate_dict_shaham['effect_pehe']

            ate_baseline_train_seed[j] = evaluate_dict_baseline['effect_ate']
            ate_shalit_train_seed[j] = evaluate_dict_shalit['effect_ate']
            ate_shaham_train_seed[j] = evaluate_dict_shaham['effect_ate']

        cfact_baseline_train_mean.append(np.mean(cfact_baseline_train_seed))
        cfact_shalit_train_mean.append(np.mean(cfact_shalit_train_seed))
        cfact_shaham_train_mean.append(np.mean(cfact_shaham_train_seed))

        cfact_baseline_train_std.append(np.std(cfact_baseline_train_seed))
        cfact_shalit_train_std.append(np.std(cfact_shalit_train_seed))
        cfact_shaham_train_std.append(np.std(cfact_shaham_train_seed))

        ate_baseline_train_mean.append(np.mean(ate_baseline_train_seed))
        ate_shalit_train_mean.append(np.mean(ate_shalit_train_seed))
        ate_shaham_train_mean.append(np.mean(ate_shaham_train_seed))

        ate_baseline_train_std.append(np.std(ate_baseline_train_seed))
        ate_shalit_train_std.append(np.std(ate_shalit_train_seed))
        ate_shaham_train_std.append(np.std(ate_shaham_train_seed))

        print(f"baseline cfact {np.mean(cfact_baseline_train_seed)} ,std {np.std(cfact_baseline_train_seed)}")
        print(f"shalit cfact {np.mean(cfact_shalit_train_seed)} ,std {np.std(cfact_shalit_train_seed)}")
        print(f"shaham cfact {np.mean(cfact_shaham_train_seed)} ,std {np.std(cfact_shaham_train_seed)}")

        print(f"baseline ate {np.mean(ate_baseline_train_seed)} ,std {np.std(ate_baseline_train_seed)}")
        print(f"shalit ate {np.mean(ate_shalit_train_seed)} ,std {np.std(ate_shalit_train_seed)}")
        print(f"shaham ate {np.mean(ate_shaham_train_seed)} ,std {np.std(ate_shaham_train_seed)}")

    print("----------------RESULTS------------------")
    print(
        f"mse pehe baseline {np.mean(np.array(cfact_baseline_train_mean))} std {np.mean(np.array(cfact_baseline_train_std))}")
    print(
        f"mse pehe shalit {np.mean(np.array(cfact_shalit_train_mean))} std {np.mean(np.array(cfact_shalit_train_std))}")
    print(
        f"mse pehe shaham {np.mean(np.array(cfact_shaham_train_mean))} std {np.mean(np.array(cfact_shaham_train_std))}")

    print(f"ate baseline {np.mean(np.array(ate_baseline_train_mean))} std {np.mean(np.array(ate_baseline_train_std))}")
    print(f"ate shalit {np.mean(np.array(ate_shalit_train_mean))} std {np.mean(np.array(ate_shalit_train_std))}")
    print(f"ate shaham {np.mean(np.array(ate_shaham_train_mean))} std {np.mean(np.array(ate_shaham_train_std))}")
