import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot

import numpy as np
import os
import time
import copy
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt

from cnn_model import CNN_B
from dataset import Dataset
from tools import print_metrics, get_model_predictions


DEFAULT_DATA_DIR = 'datasets_LHC_gm_pr'
DEFAULT_RANDOM_STATE = 21
DEFAULT_SCALER_DIR = 'scalers'
DEFAULT_MODEL_DIR = 'models'
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 12
DEFAULT_WEIGHT = 80
DEFAULT_EPOCHS_NUM = 20
DEFAULT_LOAD_CNN_MODEL = None
DEFAULT_POLYFEATURES_DEGREE = 1


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", dest="data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help=f"Path to dir containing splitted datasets. Default: {DEFAULT_DATA_DIR}")
    parser.add_argument("--random_state", dest="random_state", type=int, default=DEFAULT_RANDOM_STATE,
                        help=f'Random state for all random generators used. Default: {DEFAULT_RANDOM_STATE}')
    parser.add_argument("--scaler_dir", dest="scaler_dir", type=str, default=DEFAULT_SCALER_DIR,
                        help=f"Path to dir containing fitted scalers. Default: {DEFAULT_SCALER_DIR}")
    parser.add_argument("--model_dir", dest="model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help=f"Path to dir containing trained models. Default: {DEFAULT_MODEL_DIR}")
    parser.add_argument("--batch_size", dest='batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", dest='num_workers', type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--weight", dest='weight', type=int, default=DEFAULT_WEIGHT)
    parser.add_argument("--epochs", dest='epochs', type=int, default=DEFAULT_EPOCHS_NUM)
    parser.add_argument("--load_cnn_model", dest='load_cnn_model', type=str, default=DEFAULT_LOAD_CNN_MODEL,
                        help=f'Path to load cnn model instead of training. Default: {DEFAULT_LOAD_CNN_MODEL}')
    parser.add_argument("-pf_degree", dest="pf_degree", type=int, default=DEFAULT_POLYFEATURES_DEGREE,
                        help=f"Degree for polyfeatures transform. Default: {DEFAULT_POLYFEATURES_DEGREE}")
    return parser.parse_args()


def train_model(model, optimizer, loss_fn, scheduler, num_epochs, n_class, dataloaders, device):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_train_loss = 1e10
    best_valid_loss = 1e10

    best_epoch = -1
    full_metrics = defaultdict(list)
    model_states = list()
    for epoch in range(num_epochs):
        print('-' * 10)
        print(f'Epoch {epoch+1}/{num_epochs}')

        since = time.time()
        
        # TRAIN
        model.train()  # Set model to training mode
        
        metrics_t = defaultdict(float)
        epoch_samples_t = 0
        
        LR = optimizer.param_groups[0]["lr"]
        full_metrics['LR'].append(LR)
        print(f'LR {LR:.2e}')

        metrics_t['loss'] = 0
        for inputs_1, inputs_2, labels in dataloaders['train']:
            inputs_1 = inputs_1.to(device)
            inputs_2 = inputs_2.to(device)
            labels = labels.to(device)

            outputs = model(inputs_1, inputs_2)
            loss = loss_fn(outputs,torch.argmax(labels, dim=1))
            metrics_t['loss'] += loss.detach().cpu().numpy()
            
            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # statistics
            epoch_samples_t += inputs_1.size(0)
        
        scheduler.step()
        time_elapsed_t = time.time() - since
        train_loss = metrics_t['loss'] / epoch_samples_t
        full_metrics['train_loss'].append(train_loss)
        full_metrics['train_time_elapsed'].append(time_elapsed_t)

        print_metrics(metrics_t, epoch_samples_t, 'train')  
            
        # VALIDATION
        since_v = time.time()
        model.eval()   # Set model to evaluate mode

        metrics_v = defaultdict(float)
        epoch_samples_v = 0
        metrics_v['loss'] = 0
        cm_valid = np.zeros((n_class, n_class), dtype=np.int)
        for inputs_1, inputs_2, labels in dataloaders['valid']:
            inputs_1 = inputs_1.to(device)
            inputs_2 = inputs_2.to(device)
            labels = labels.to(device)

            outputs = model(inputs_1, inputs_2)
            loss = loss_fn(outputs,torch.argmax(labels, dim=1))
            metrics_v['loss'] += loss.detach().cpu().numpy()

            # statistics
            epoch_samples_v += inputs_1.size(0)
        
        print(cm_valid)
        valid_loss = metrics_v['loss'] / epoch_samples_v
        full_metrics['valid_loss'].append(valid_loss)
        full_metrics['cm_valid'].append(cm_valid)

        print_metrics(metrics_v, epoch_samples_v, 'valid')               
        
        model_states.append(copy.deepcopy(model.state_dict()))
        if 0 < valid_loss < best_valid_loss:
            print("saving best model")
            best_train_loss = train_loss
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            
        # time calc
        time_elapsed_v = time.time() - since_v
        time_elapsed = time.time() - since
        full_metrics['time_elapsed'].append(time_elapsed)
        full_metrics['valid_time_elapsed'].append(time_elapsed_v)
        full_metrics['gm_accuracy'].append(cm_valid[0][0]/sum(cm_valid[0]))
        full_metrics['pr_accuracy'].append(cm_valid[1][1]/sum(cm_valid[1]))


        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
    best_loss_dict = dict(best_train_loss=best_train_loss, best_valid_loss=best_valid_loss)
    print(f'Best epoch: {best_epoch}, best_train_loss: {best_train_loss:.2e}, best_valid_loss: {best_valid_loss:.2e}')

    return model, best_loss_dict, best_epoch, model_states, full_metrics 


def main(args):
    np.random.seed(args.random_state)

    if not torch.cuda.is_available():
        raise Exception("GPU not availalbe. CPU training can be slow.")
    print("device name", torch.cuda.get_device_name(0))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_matrices = np.load(f'{args.data_dir}/train_matrices.npy')
    train_features = np.load(f'{args.data_dir}/train_features.npy')
    train_target = np.load(f'{args.data_dir}/train_target.npy')

    valid_matrices = np.load(f'{args.data_dir}/valid_matrices.npy')
    valid_features = np.load(f'{args.data_dir}/valid_features.npy')
    valid_target = np.load(f'{args.data_dir}/valid_target.npy')

    times_scaler = MinMaxScaler()
    edep_scaler = StandardScaler()
    muon_scaler = StandardScaler()
    features_scaler = StandardScaler()

    scaled_train_times = times_scaler.fit_transform(train_matrices[..., 0].flatten().reshape(-1, 1))
    scaled_train_edep = edep_scaler.fit_transform(np.log(train_matrices[..., 1].flatten().reshape(-1, 1) + 1))
    scaled_train_muons = muon_scaler.fit_transform(train_matrices[..., 2].flatten().reshape(-1, 1))
    scaled_train_matrices = train_matrices.copy()
    scaled_train_matrices[..., 0] = scaled_train_times.reshape(train_matrices.shape[0], 16, 16)
    scaled_train_matrices[..., 1] = scaled_train_edep.reshape(train_matrices.shape[0], 16, 16)
    scaled_train_matrices[..., 2] = scaled_train_muons.reshape(train_matrices.shape[0], 16, 16)
    scaled_train_features = features_scaler.fit_transform(train_features)

    scaled_valid_times = times_scaler.transform(valid_matrices[..., 0].flatten().reshape(-1, 1))
    scaled_valid_edep = edep_scaler.transform(np.log(valid_matrices[..., 1].flatten().reshape(-1, 1) + 1))
    scaled_valid_muons = muon_scaler.transform(valid_matrices[..., 2].flatten().reshape(-1, 1))
    scaled_valid_matrices = valid_matrices.copy()
    scaled_valid_matrices[..., 0] = scaled_valid_times.reshape(valid_matrices.shape[0], 16, 16)
    scaled_valid_matrices[..., 1] = scaled_valid_edep.reshape(valid_matrices.shape[0], 16, 16)
    scaled_valid_matrices[..., 2] = scaled_valid_muons.reshape(valid_matrices.shape[0], 16, 16)
    scaled_valid_features = features_scaler.transform(valid_features)

    os.makedirs(f'{args.scaler_dir}', exist_ok=True)

    joblib.dump(times_scaler, f'{args.scaler_dir}/times_scaler.pkl')
    joblib.dump(edep_scaler, f'{args.scaler_dir}/edep_scaler.pkl')
    joblib.dump(muon_scaler, f'{args.scaler_dir}/muon_scaler.pkl')
    joblib.dump(features_scaler, f'{args.scaler_dir}/features_scaler.pkl')

    features_filter = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    mat_filter = [1, 2]

    X_features_train = scaled_train_features[..., features_filter].astype('float32')
    X_mat_train = scaled_train_matrices[..., mat_filter].astype('float32')
    y_train = train_target
    X_features_valid = scaled_valid_features[..., features_filter].astype('float32')
    X_mat_valid = scaled_valid_matrices[..., mat_filter].astype('float32')
    y_valid = valid_target

    dataset_train = Dataset(X_features_train, X_mat_train, y_train)
    dataset_valid = Dataset(X_features_valid, X_mat_valid, y_valid)

    dataloaders = {
        'train': DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        'valid': DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
        }

    dl_iter = iter(dataloaders['train'])
    sample = next(dl_iter)
    n_features = sample[0].shape[-1]
    n_mat = sample[1].shape[1]
    n_class = sample[2].shape[-1]

    model = CNN_B(n_features=n_features, n_mat=n_mat, n_class=n_class)
    _, counts_class_train = np.unique(y_train, return_counts=True)
    weight = torch.tensor(1/counts_class_train)
    weight[1] *= args.weight

    optimizer_fn = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=weight.float().to(device))
    scheduler = lr_scheduler.StepLR(optimizer_fn, step_size=5, gamma=0.1)
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    if args.load_cnn_model is None:
        model, best_loss_dict, best_epoch, model_states, full_metrics = train_model(model.to(device),
                                                                            optimizer_fn, loss_fn, scheduler,
                                                                            num_epochs=args.epochs, n_class=n_class,
                                                                            dataloaders=dataloaders, device=device)

        model_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_fn_state_dict': optimizer_fn.state_dict(),
        }

        
        models_out_dir = os.path.join(args.model_dir, 'CNN')
        os.makedirs(models_out_dir, exist_ok=True)
        torch.save(model_dict, os.path.join(models_out_dir, f"CNN_dict_{timestamp}.pth"))
    else:
        models_out_dir = os.path.join(args.model_dir, 'CNN')
        model_dict = torch.load(f'{models_out_dir}/{args.load_cnn_model}')
        model.load_state_dict(model_dict['model_state_dict'])
        model.to(device)
        model.eval()

    X_train, y_train = get_model_predictions(model, 'train', dataloaders=dataloaders, device=device)
    X_valid, y_valid = get_model_predictions(model, 'valid', dataloaders=dataloaders, device=device)

    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train)
    X_valid_scaled = final_scaler.transform(X_valid)
    joblib.dump(final_scaler, f'{args.scaler_dir}/final_scaler.pkl')
    pf = PolynomialFeatures(degree=args.pf_degree)
    X_train_scaled_pf = X_train#pf.fit_transform(X_train_scaled)
    X_valid_scaled_pf = X_valid#pf.transform(X_valid_scaled)
    joblib.dump(pf, f'{args.scaler_dir}/pf.pkl')

    lr = LogisticRegression(C=0.1, random_state=args.random_state)
    # lr = SVC(kernel='linear', random_state=args.random_state, probability=True)
    lr.fit(X_train_scaled_pf, y_train)
    lr_out_dir = os.path.join(args.model_dir, 'LR')
    os.makedirs(lr_out_dir, exist_ok=True)
    joblib.dump(lr, f'{lr_out_dir}/LR_dict_{timestamp}.pkl')
    preds_valid = lr.predict_proba(X_valid_scaled_pf)

    valid_energy = valid_features[:, 0]
    valid_theta = valid_features[:, 4]

    plt.figure(figsize=(8, 6))
    angle_min, angle_max = 0, 30
    energy_min, energy_max = 15, 16
    cond = np.where((valid_theta >= angle_min) & (valid_theta < angle_max) & \
                    (valid_energy >= energy_min) & (valid_energy < energy_max))
    preds_considered = preds_valid[cond][:, 1]
    y_considered = y_valid[cond]

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
    # plt.xlim(0, 1)
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)
    plt.xlabel('probability of being proton threshold', fontsize=15)
    plt.ylabel('fraction', fontsize=15)
    plt.show()

if __name__ == '__main__':
    main(parse_arguments())
