import os
from argparse import ArgumentParser

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from cnn_model import CNN_B
from dataset import Dataset
from tools import get_model_predictions


DEFAULT_DATA_DIR = 'datasets_LHC_gm_pr'
DEFAULT_SCALER_DIR = 'scalers'
DEFAULT_MODEL_DIR = 'models'
DEFAULT_CNN_MODEL_NAME = 'CNN_dict_20210726-132458.pth'
DEFAULT_LR_MODEL_NAME = 'LR_dict_20210726-132458.pkl'
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 12

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", dest="data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help=f"Path to dir containing splitted datasets. Default: {DEFAULT_DATA_DIR}")
    parser.add_argument("--scaler_dir", dest="scaler_dir", type=str, default=DEFAULT_SCALER_DIR,
                        help=f"Path to dir containing fitted scalers. Default: {DEFAULT_SCALER_DIR}")
    parser.add_argument("--model_dir", dest="model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help=f"Path to dir containing trained models. Default: {DEFAULT_MODEL_DIR}")
    parser.add_argument("--cnn_model_name", dest="cnn_model_name", type=str, default=DEFAULT_CNN_MODEL_NAME,
                        help=f"Name of the trained cnn model. Default: {DEFAULT_CNN_MODEL_NAME}")
    parser.add_argument("--lr_model_name", dest="lr_model_name", type=str, default=DEFAULT_LR_MODEL_NAME,
                        help=f"Name of the trained lr model. Default: {DEFAULT_LR_MODEL_NAME}")
    parser.add_argument("--batch_size", dest='batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", dest='num_workers', type=int, default=DEFAULT_NUM_WORKERS)
    return parser.parse_args()


def main(args):
    if not torch.cuda.is_available():
        raise Exception("GPU not availalbe. CPU training can be slow.")
    print("device name", torch.cuda.get_device_name(0))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_matrices = np.load(f'{args.data_dir}/test_matrices.npy')
    test_features = np.load(f'{args.data_dir}/test_features.npy')
    test_target = np.load(f'{args.data_dir}/test_target.npy')

    times_scaler = joblib.load(f'{args.scaler_dir}/times_scaler.pkl')
    edep_scaler = joblib.load(f'{args.scaler_dir}/edep_scaler.pkl')
    muon_scaler = joblib.load(f'{args.scaler_dir}/muon_scaler.pkl')
    features_scaler = joblib.load(f'{args.scaler_dir}/features_scaler.pkl')

    scaled_test_times = times_scaler.transform(test_matrices[..., 0].flatten().reshape(-1, 1))
    scaled_test_edep = edep_scaler.transform(np.log(test_matrices[..., 1].flatten().reshape(-1, 1) + 1))
    scaled_test_muons = muon_scaler.transform(test_matrices[..., 2].flatten().reshape(-1, 1))
    scaled_test_matrices = test_matrices.copy()
    scaled_test_matrices[..., 0] = scaled_test_times.reshape(test_matrices.shape[0], 16, 16)
    scaled_test_matrices[..., 1] = scaled_test_edep.reshape(test_matrices.shape[0], 16, 16)
    scaled_test_matrices[..., 2] = scaled_test_muons.reshape(test_matrices.shape[0], 16, 16)
    scaled_test_features = features_scaler.transform(test_features)

    features_filter = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    mat_filter = [1, 2]

    X_features_test = scaled_test_features[..., features_filter].astype('float32')
    X_mat_test = scaled_test_matrices[..., mat_filter].astype('float32')
    y_test = test_target
    dataset_test = Dataset(X_features_test, X_mat_test, y_test)

    dataloaders = {
        'test': DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
    }

    dl_iter = iter(dataloaders['test'])
    sample = next(dl_iter)
    n_features = sample[0].shape[-1]
    n_mat = sample[1].shape[1]
    n_class = sample[2].shape[-1]

    model = CNN_B(n_features=n_features, n_mat=n_mat, n_class=n_class)

    models_out_dir = os.path.join(args.model_dir, 'CNN')
    model_dict = torch.load(f'{models_out_dir}/{args.cnn_model_name}')
    model.load_state_dict(model_dict['model_state_dict'])
    model.to(device)
    model.eval()

    X_test, y_test = get_model_predictions(model, 'test', dataloaders=dataloaders, device=device)

    final_scaler = joblib.load(f'{args.scaler_dir}/final_scaler.pkl')
    X_test_scaled = final_scaler.transform(X_test)
    pf = joblib.load(f'{args.scaler_dir}/pf.pkl')
    # X_test_scaled_pf = pf.transform(X_test_scaled)
    X_test_scaled_pf = X_test

    lr_out_dir = os.path.join(args.model_dir, 'LR')
    lr = joblib.load(f'{lr_out_dir}/{args.lr_model_name}')
    preds_test = lr.predict_proba(X_test_scaled_pf)

    test_energy = test_features[:, 0]
    test_theta = test_features[:, 4]

    plt.figure(figsize=(8, 6))
    angle_min, angle_max = 0, 30
    energy_min, energy_max = 14, 15
    cond = np.where((test_theta >= angle_min) & (test_theta < angle_max) & \
                    (test_energy >= energy_min) & (test_energy < energy_max))
    preds_considered = preds_test[cond][:, 1]
    y_considered = y_test[cond]

    nbins = 10000
    bins_range = (0, 1)
    N, prob_bins = np.histogram(preds_considered, bins=nbins, range=bins_range)
    probs_bins_centers = prob_bins[1:] - 0.5*(prob_bins[1] - prob_bins[0])
    probs_bins_half_width = 0.5*(prob_bins[1] - prob_bins[0])

    cpg, ipp = [], []
    n_gamma = (y_considered == 0).sum()
    n_protons = (y_considered == 1).sum()
    for right_edge in prob_bins[1:]:
        cond = np.where((y_considered == 0) & (preds_considered < right_edge))
        correctly_predicted_gamma = y_considered[cond].shape[0]
        cond = np.where((y_considered == 1) & (preds_considered < right_edge))
        incorrectly_predicted_protons = y_considered[cond].shape[0]
        cpg.append(correctly_predicted_gamma)
        ipp.append(incorrectly_predicted_protons)
    plt.scatter(x=prob_bins[1:-1], y=cpg[:-1] / n_gamma, c='b', label='survived gamma')
    plt.scatter(x=prob_bins[1:-1], y=ipp[:-1] / n_protons, c='r', label='survived protons')
    # plt.semilogy()
    plt.loglog()
    plt.legend()
    plt.grid(which='both')
    # plt.xlim(0, 1)
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)
    plt.xlabel('probability of being proton threshold', fontsize=15)
    plt.ylabel('fraction', fontsize=15)
    plt.show()

if __name__ == '__main__':
    main(parse_arguments())
