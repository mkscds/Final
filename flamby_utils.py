import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from tqdm import tqdm



torch.manual_seed(42)


def evaluate_model_on_tests(
    model, test_dataloaders, metric, use_gpu=True, return_pred=False
):
    """This function takes a pytorch model and evaluate it on a list of\
    dataloaders using the provided metric function.
    Parameters
    ----------
    model: torch.nn.Module,
        A trained model that can forward the test_dataloaders outputs
    test_dataloaders: List[torch.utils.data.DataLoader]
        A list of torch dataloaders
    metric: callable,
        A function with the following signature:\
            (y_true: np.ndarray, y_pred: np.ndarray) -> scalar
    use_gpu: bool, optional,
        Whether or not to perform computations on GPU if available. \
        Defaults to True.
    Returns
    -------
    dict
        A dictionnary with keys client_test_{0} to \
        client_test_{len(test_dataloaders) - 1} and associated scalar metrics \
        as leaves.
    """
    results_dict = {}
    y_true_dict = {}
    y_pred_dict = {}
    if torch.cuda.is_available() and use_gpu:
        model = model.cuda()
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(test_dataloaders))):
            rng_state = torch.get_rng_state()
            test_dataloader_iterator = iter(test_dataloaders[i])
            # since iterating over the dataloader changes torch random state
            # we set it again to its previous value
            # https://discuss.pytorch.org/t/does-iterating-over-unshuffled-dataloader-change-random-state/173137
            torch.set_rng_state(rng_state)
            y_pred_final = []
            y_true_final = []
            for X, y in test_dataloader_iterator:
                if torch.cuda.is_available() and use_gpu:
                    X = X.cuda()
                    y = y.cuda()
                y_pred = model(X).detach().cpu()
                y = y.detach().cpu()
                y_pred_final.append(y_pred.numpy())
                y_true_final.append(y.numpy())

            y_true_final = np.concatenate(y_true_final)
            y_pred_final = np.concatenate(y_pred_final)
            results_dict[f"client_test_{i}"] = metric(y_true_final, y_pred_final)
            if return_pred:
                y_true_dict[f"client_test_{i}"] = y_true_final
                y_pred_dict[f"client_test_{i}"] = y_pred_final
    model.train()
    if return_pred:
        return results_dict, y_true_dict, y_pred_dict
    else:
        return results_dict


