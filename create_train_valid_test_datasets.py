import os
from argparse import ArgumentParser

import numpy as np
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian


DEFAULT_DATA_URL = 'https://kascade-sim-data.s3.eu-central-1.amazonaws.com'
DEFAULT_NPZ_DIR = 'npz'
DEFAULT_DATA_DIR = 'datasets'
DEFAULT_DATASET_NAME = 'LHC_gm_pr'
DEFAULT_RANDOM_STATE = 21
DEFAULT_AUGMENTATIONS_FRACTION = 0.3
DEFAULT_VALIDATION_SIZE = 0.2
DEFAULT_TEST_SIZE = 0.2


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--npz-dir", dest="npz_dir", type=str, default=DEFAULT_NPZ_DIR,
                        help=f"Path to dir containing '.npz' archives with data. Default: {DEFAULT_NPZ_DIR}")
    parser.add_argument("--data-dir", dest="data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help=f"Path to dir containing splitted datasets. Default: {DEFAULT_DATA_DIR}")
    parser.add_argument("--data_url", dest="data_url", type=str, default=DEFAULT_DATA_URL,
                        help=f"url for data downloading. Default: {DEFAULT_DATA_URL}")
    parser.add_argument("--dataset_name", dest="dataset_name", type=str, default=DEFAULT_DATASET_NAME,
                        help=f"Dataset name. Default: {DEFAULT_DATASET_NAME}")
    parser.add_argument("--random_state", dest="random_state", type=int, default=DEFAULT_RANDOM_STATE,
                        help=f'Random state for all random generators used. Default: {DEFAULT_RANDOM_STATE}')
    parser.add_argument("--augmentations", dest="augmentations_fraction", type=float, default=DEFAULT_AUGMENTATIONS_FRACTION,
                        help="fraction of data augmentations generated for each of the 4 possible rotations for 90*k degrees")
    parser.add_argument("--valid_size", dest="valid_size", type=float, default=DEFAULT_VALIDATION_SIZE,
                        help=f"fraction of data used for validation dataset creation. Default: {DEFAULT_VALIDATION_SIZE}")
    parser.add_argument("--test_size", dest="test_size", type=float, default=DEFAULT_TEST_SIZE,
                        help=f"fraction of data used for test dataset creation. Default: {DEFAULT_TEST_SIZE}")
    return parser.parse_args()


def download_dataset(dataset_name, data_url, npz_dir):
    if data_url.endswith('/'):
        data_url = data_url[:-1]
    os.makedirs(npz_dir, exist_ok=True)
    files_to_download = [f'{dataset_name}_matrices.npz', f'{dataset_name}_features.npz']
    for filename in files_to_download:
        dst_filepath = os.path.join(npz_dir, filename)
        if not os.path.exists(dst_filepath):
            print(f'Downloading {filename}... ', end='')
            urlretrieve(f'{data_url}/{filename}', dst_filepath)
            print('Done!')
        else:
            print(f'File {filename} already exists')
    print(f'Files in "{npz_dir}" directory: {os.listdir(npz_dir)}')


def rotate_x_y_Az(features_vector, n90):
    features_vector = features_vector.copy()
    if n90 == 1 or n90 == 2:
        features_vector[1] = -features_vector[1]
    if n90 == 2 or n90 == 3:
        features_vector[2] = -features_vector[2]
    features_vector[5] += n90 * 90
    if features_vector[5] < 0:
        features_vector[5] += 360
    if features_vector[5] > 360:
        features_vector[5] -= 360
    return features_vector


def generate_rotations(matrices, features, part_class, aug_size):
    matrices_train_rot_1 = np.rot90(matrices, 1, axes=(1, 2))
    matrices_train_rot_2 = np.rot90(matrices, 2, axes=(1, 2))
    matrices_train_rot_3 = np.rot90(matrices, 3, axes=(1, 2))
    features_train_rot_1 = np.apply_along_axis(rotate_x_y_Az, 1, features, n90=1)
    features_train_rot_2 = np.apply_along_axis(rotate_x_y_Az, 1, features, n90=2)
    features_train_rot_3 = np.apply_along_axis(rotate_x_y_Az, 1, features, n90=3)
    choice1 = np.random.randint(features.shape[0], size=int(features.shape[0] * aug_size))
    choice2 = np.random.randint(features.shape[0], size=int(features.shape[0] * aug_size))
    choice3 = np.random.randint(features.shape[0], size=int(features.shape[0] * aug_size))
    new_matrices = np.concatenate([matrices,
                                        matrices_train_rot_1[choice1, ...],
                                        matrices_train_rot_2[choice2, ...],
                                        matrices_train_rot_3[choice3, ...]], axis=0)
    new_features = np.concatenate([features,
                                        features_train_rot_1[choice1],
                                        features_train_rot_2[choice2],
                                        features_train_rot_3[choice3]], axis=0)
    new_part_class = np.concatenate([part_class,
                                        part_class[choice1],
                                        part_class[choice2],
                                        part_class[choice3]])
    return new_matrices, new_features, new_part_class


def to_XY_astropy(Ze, Az):
    r = 1
    ZeR = np.radians(90 - Ze)
    AzR = np.radians(Az)
    x, y, z = spherical_to_cartesian(r, ZeR, AzR)
    return x.value, y.value, z.value


def to_R_astropy(x, y, z=None):
    if z is None:
        z = np.sqrt(1 - x*x - y*y)
    _, ZeR, AzR = cartesian_to_spherical(x, y, z)
    Ze = 90 - np.degrees(ZeR.value)
    Az = np.degrees(AzR.value)
    return Ze, Az


def main(args):
    np.random.seed(args.random_state)
    download_dataset(args.dataset_name, args.data_url, args.npz_dir)
    matrices = np.load(os.path.join(args.npz_dir, f'{args.dataset_name}_matrices.npz'))['matrices']
    features = np.load(os.path.join(args.npz_dir, f'{args.dataset_name}_features.npz'))['features']
    part_class, used_features = features[:, 0], features[:, 1:]
    matrices_train, matrices_test, features_train, features_test, part_class_train, part_class_test = train_test_split(
            matrices, used_features, part_class, test_size=args.test_size, stratify=part_class,
            random_state=args.random_state
    )
    valid_size = 1. / (1 - args.test_size) * args.valid_size
    matrices_train, matrices_valid, features_train, features_valid, part_class_train, part_class_valid = train_test_split(
                matrices_train, features_train, part_class_train, test_size=valid_size, stratify=part_class_train,
                random_state=args.random_state
    )
    
    matrices_train, features_train, part_class_train = generate_rotations(matrices_train,
                                                                          features_train,
                                                                          part_class_train,
                                                                          args.augmentations_fraction)
    
    dir_train = to_XY_astropy(features_train[:, 4], features_train[:, 5])
    dir_valid = to_XY_astropy(features_valid[:, 4], features_valid[:, 5])
    dir_test = to_XY_astropy(features_test[:, 4], features_test[:, 5])
    features_train = np.concatenate([features_train, np.array(dir_train).T], axis=1)
    features_valid = np.concatenate([features_valid, np.array(dir_valid).T], axis=1)
    features_test = np.concatenate([features_test, np.array(dir_test).T], axis=1)

    print(f'train matrices: {matrices_train.shape}, ' 
           f'train features: {features_train.shape}, '
           f'train_target: {part_class_train.shape}')
    print(f'validation matrices: {matrices_valid.shape}, '
           f'validation features: {features_valid.shape}, '
           f'validation target: {part_class_valid.shape}')
    print(f'test matrices: {matrices_test.shape}, '
           f'test features: {features_test.shape}, '
           f'test target: {part_class_test.shape}')

    os.makedirs(f'{args.data_dir}_{args.dataset_name}', exist_ok=True)                                                                          
    np.save(f'{args.data_dir}_{args.dataset_name}/train_matrices', matrices_train)                                                                          
    np.save(f'{args.data_dir}_{args.dataset_name}/train_features', features_train)
    np.save(f'{args.data_dir}_{args.dataset_name}/train_target', part_class_train)
    np.save(f'{args.data_dir}_{args.dataset_name}/valid_matrices', matrices_valid)
    np.save(f'{args.data_dir}_{args.dataset_name}/valid_features', features_valid)
    np.save(f'{args.data_dir}_{args.dataset_name}/valid_target', part_class_valid)
    np.save(f'{args.data_dir}_{args.dataset_name}/test_matrices', matrices_test)
    np.save(f'{args.data_dir}_{args.dataset_name}/test_features', features_test)
    np.save(f'{args.data_dir}_{args.dataset_name}/test_target', part_class_test)


if __name__ == '__main__':
    main(parse_arguments())
