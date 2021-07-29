import numpy as np
import torch

import numpy as np


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append(f"{k}: {metrics[k] / epoch_samples :.2e}")

    print(f'{phase}: {", ".join(outputs)}')


def get_model_predictions(model, dataset_name, dataloaders, device):
    idx_max = len(dataloaders[dataset_name])
    test_preds = []
    test_true = []
    for idx, (inputs_1, inputs_2, labels) in enumerate(iter(dataloaders[dataset_name])): 
        inputs_1 = inputs_1.to(device)
        inputs_2 = inputs_2.to(device)
  # Predict
        pred = model(inputs_1, inputs_2).cpu().detach().numpy()
  # The loss functions include the sigmoid function.
        test_preds.append(pred)
        labels_np = torch.argmax(labels, dim=1).cpu().numpy()
        test_true.append(labels_np)

    test_preds = np.concatenate(test_preds)
    test_true = np.concatenate(test_true)
    return test_preds, test_true