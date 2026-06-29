"""
This module defines the `Dataset` class and functions for generating datasets tailored to different model types.
A `Dataset` object in this module contains three different dataloaders, each providing a specific version of the data
required by different models:

- `raw_dataloaders`: Returns the raw time series data, suitable for recurrent neural networks (RNNs) and structured
  state space models (SSMs).
- `coeff_dataloaders`: Provides the coefficients of an interpolation of the data, used by Neural Controlled Differential
  Equations (NCDEs).
- `path_dataloaders`: Provides the log-signature of the data over intervals, used by Neural Rough Differential Equations
  (NRDEs) and Log-NCDEs.

The module also includes utility functions for processing and generating these datasets, ensuring compatibility with
different model requirements.
"""

import os
import pickle
from dataclasses import dataclass
from typing import Dict
import struct

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from data_dir.dataloaders import Dataloader
# from data_dir.generate_coeffs import calc_coeffs
# from data_dir.generate_paths import calc_paths


@dataclass
class Dataset:
    name: str
    raw_dataloaders: Dict[str, Dataloader]
    # coeff_dataloaders: Dict[str, Dataloader]
    # path_dataloaders: Dict[str, Dataloader]
    data_dim: int
    # logsig_dim: int
    # intervals: jnp.ndarray
    label_dim: int


def batch_calc_paths(data, stepsize, depth, inmemory=True):
    N = len(data)
    batchsize = 128
    num_batches = N // batchsize
    remainder = N % batchsize
    path_data = []
    if inmemory:
        out_func = lambda x: x
        in_func = lambda x: x
    else:
        out_func = lambda x: np.array(x)
        in_func = lambda x: jnp.array(x)
    for i in range(num_batches):
        path_data.append(
            out_func(
                calc_paths(
                    in_func(data[i * batchsize : (i + 1) * batchsize]), stepsize, depth
                )
            )
        )
    if remainder > 0:
        path_data.append(
            out_func(calc_paths(in_func(data[-remainder:]), stepsize, depth))
        )
    if inmemory:
        path_data = jnp.concatenate(path_data)
    else:
        path_data = np.concatenate(path_data)
    return path_data


def batch_calc_coeffs(data, include_time, T, inmemory=True):
    N = len(data)
    batchsize = 128
    num_batches = N // batchsize
    remainder = N % batchsize
    coeffs = []
    if inmemory:
        out_func = lambda x: x
        in_func = lambda x: x
    else:
        out_func = lambda x: np.array(x)
        in_func = lambda x: jnp.array(x)
    for i in range(num_batches):
        coeffs.append(
            out_func(
                calc_coeffs(
                    in_func(data[i * batchsize : (i + 1) * batchsize]), include_time, T
                )
            )
        )
    if remainder > 0:
        coeffs.append(
            out_func(calc_coeffs(in_func(data[-remainder:]), include_time, T))
        )
    if inmemory:
        coeffs = jnp.concatenate(coeffs)
    else:
        coeffs = np.concatenate(coeffs)
    return coeffs


def dataset_generator(
    name,
    data,
    labels,
    stepsize,
    depth,
    include_time,
    T,
    inmemory=True,
    idxs=None,
    use_presplit=False,
    *,
    key,
):
    N = len(data)
    if idxs is None:
        if use_presplit:
            train_data, val_data, test_data = data
            train_labels, val_labels, test_labels = labels
        else:
            permkey, key = jr.split(key)
            bound1 = int(N * 0.7)
            bound2 = int(N * 0.85)
            idxs_new = jr.permutation(permkey, N)
            train_data, train_labels = (
                data[idxs_new[:bound1]],
                labels[idxs_new[:bound1]],
            )
            val_data, val_labels = (
                data[idxs_new[bound1:bound2]],
                labels[idxs_new[bound1:bound2]],
            )
            test_data, test_labels = data[idxs_new[bound2:]], labels[idxs_new[bound2:]]
    else:
        train_data, train_labels = data[idxs[0]], labels[idxs[0]]
        val_data, val_labels = data[idxs[1]], labels[idxs[1]]
        test_data, test_labels = None, None

    # train_paths = batch_calc_paths(train_data, stepsize, depth)
    # val_paths = batch_calc_paths(val_data, stepsize, depth)
    # test_paths = batch_calc_paths(test_data, stepsize, depth)
    # intervals = jnp.arange(0, train_data.shape[1], stepsize)
    # intervals = jnp.concatenate((intervals, jnp.array([train_data.shape[1]])))
    # intervals = intervals * (T / train_data.shape[1])

    # train_coeffs = calc_coeffs(train_data, include_time, T)
    # val_coeffs = calc_coeffs(val_data, include_time, T)
    # test_coeffs = calc_coeffs(test_data, include_time, T)
    # train_coeff_data = (
    #     (T / train_data.shape[1])
    #     * jnp.repeat(
    #         jnp.arange(train_data.shape[1])[None, :], train_data.shape[0], axis=0
    #     ),
    #     train_coeffs,
    #     train_data[:, 0, :],
    # )
    # val_coeff_data = (
    #     (T / val_data.shape[1])
    #     * jnp.repeat(jnp.arange(val_data.shape[1])[None, :], val_data.shape[0], axis=0),
    #     val_coeffs,
    #     val_data[:, 0, :],
    # )
    # if idxs is None:
    #     test_coeff_data = (
    #         (T / test_data.shape[1])
    #         * jnp.repeat(
    #             jnp.arange(test_data.shape[1])[None, :], test_data.shape[0], axis=0
    #         ),
    #         test_coeffs,
    #         test_data[:, 0, :],
    #     )

    # train_path_data = (
    #     (T / train_data.shape[1])
    #     * jnp.repeat(
    #         jnp.arange(train_data.shape[1])[None, :], train_data.shape[0], axis=0
    #     ),
    #     train_paths,
    #     train_data[:, 0, :],
    # )
    # val_path_data = (
    #     (T / val_data.shape[1])
    #     * jnp.repeat(jnp.arange(val_data.shape[1])[None, :], val_data.shape[0], axis=0),
    #     val_paths,
    #     val_data[:, 0, :],
    # )
    # if idxs is None:
    #     test_path_data = (
    #         (T / test_data.shape[1])
    #         * jnp.repeat(
    #             jnp.arange(test_data.shape[1])[None, :], test_data.shape[0], axis=0
    #         ),
    #         test_paths,
    #         test_data[:, 0, :],
    #     )

    data_dim = train_data.shape[-1]
    if len(train_labels.shape) == 1 or name == "ppg":
        label_dim = 1
    else:
        label_dim = train_labels.shape[-1]
    # logsig_dim = train_paths.shape[-1]

    raw_dataloaders = {
        "train": Dataloader(train_data, train_labels, inmemory),
        "val": Dataloader(val_data, val_labels, inmemory),
        "test": Dataloader(test_data, test_labels, inmemory),
    }
    # coeff_dataloaders = {
    #     "train": Dataloader(train_coeff_data, train_labels, inmemory),
    #     "val": Dataloader(val_coeff_data, val_labels, inmemory),
    #     "test": Dataloader(test_coeff_data, test_labels, inmemory),
    # }

    # path_dataloaders = {
    #     "train": Dataloader(train_path_data, train_labels, inmemory),
    #     "val": Dataloader(val_path_data, val_labels, inmemory),
    #     "test": Dataloader(test_path_data, test_labels, inmemory),
    # }
    return Dataset(
        name,
        raw_dataloaders,
        # coeff_dataloaders,
        # path_dataloaders,
        data_dim,
        # logsig_dim,
        # intervals,
        label_dim,
    )


def create_uea_dataset(
    data_dir,
    name,
    use_idxs,
    use_presplit,
    stepsize,
    depth,
    include_time,
    T,
    *,
    key,
):

    if use_presplit:
        idxs = None
        with open(data_dir + f"/processed/UEA/{name}/X_train.pkl", "rb") as f:
            train_data = pickle.load(f)
        with open(data_dir + f"/processed/UEA/{name}/y_train.pkl", "rb") as f:
            train_labels = pickle.load(f)
        with open(data_dir + f"/processed/UEA/{name}/X_val.pkl", "rb") as f:
            val_data = pickle.load(f)
        with open(data_dir + f"/processed/UEA/{name}/y_val.pkl", "rb") as f:
            val_labels = pickle.load(f)
        with open(data_dir + f"/processed/UEA/{name}/X_test.pkl", "rb") as f:
            test_data = pickle.load(f)
        with open(data_dir + f"/processed/UEA/{name}/y_test.pkl", "rb") as f:
            test_labels = pickle.load(f)
        if include_time:
            ts = (T / train_data.shape[1]) * jnp.repeat(
                jnp.arange(train_data.shape[1])[None, :], train_data.shape[0], axis=0
            )
            train_data = jnp.concatenate([ts[:, :, None], train_data], axis=2)
            ts = (T / val_data.shape[1]) * jnp.repeat(
                jnp.arange(val_data.shape[1])[None, :], val_data.shape[0], axis=0
            )
            val_data = jnp.concatenate([ts[:, :, None], val_data], axis=2)
            ts = (T / test_data.shape[1]) * jnp.repeat(
                jnp.arange(test_data.shape[1])[None, :], test_data.shape[0], axis=0
            )
            test_data = jnp.concatenate([ts[:, :, None], test_data], axis=2)
        data = (train_data, val_data, test_data)
        onehot_labels = (train_labels, val_labels, test_labels)
    else:
        with open(data_dir + f"/processed/UEA/{name}/data.pkl", "rb") as f:
            data = pickle.load(f)
        with open(data_dir + f"/processed/UEA/{name}/labels.pkl", "rb") as f:
            labels = pickle.load(f)
        onehot_labels = jnp.zeros((len(labels), len(jnp.unique(labels))))
        onehot_labels = onehot_labels.at[jnp.arange(len(labels)), labels].set(1)
        if use_idxs:
            with open(data_dir + f"/processed/UEA/{name}/original_idxs.pkl", "rb") as f:
                idxs = pickle.load(f)
        else:
            idxs = None

        if include_time:
            ts = (T / data.shape[1]) * jnp.repeat(
                jnp.arange(data.shape[1])[None, :], data.shape[0], axis=0
            )
            data = jnp.concatenate([ts[:, :, None], data], axis=2)

    return dataset_generator(
        name,
        data,
        onehot_labels,
        stepsize,
        depth,
        include_time,
        T,
        idxs=idxs,
        use_presplit=use_presplit,
        key=key,
    )

def create_long_term_prediction_dataset(
    data_dir,
    name,
    use_idxs,
    use_presplit,
    stepsize,
    depth,
    include_time,
    T,
    *,
    key,
):
    """
    Create dataset loader for standard time series forecasting benchmarks.
    
    Args:
        data_dir (str): root directory with processed pickles
        name (str): dataset name (ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL, Traffic, Exchange)
        use_presplit (bool): must be True (since we always save train/val/test pickles)
        stepsize (int): stride for batching
        depth (int): hierarchical depth for dataset_generator
        include_time (bool): whether to add explicit time channel
        T (int): total time horizon for scaling time channel
        key: JAX PRNG key
    
    Returns:
        dataset_generator object
    """

    if not use_presplit:
        raise ValueError("For forecasting benchmarks, use_presplit=True is required.")

    # Load pre-split pickles
    base_path = os.path.join(data_dir, "processed", name)
    with open(os.path.join(base_path, "X_train.pkl"), "rb") as f:
        train_data = pickle.load(f)
    with open(os.path.join(base_path, "y_train.pkl"), "rb") as f:
        train_labels = pickle.load(f)
    with open(os.path.join(base_path, "X_val.pkl"), "rb") as f:
        val_data = pickle.load(f)
    with open(os.path.join(base_path, "y_val.pkl"), "rb") as f:
        val_labels = pickle.load(f)
    with open(os.path.join(base_path, "X_test.pkl"), "rb") as f:
        test_data = pickle.load(f)
    with open(os.path.join(base_path, "y_test.pkl"), "rb") as f:
        test_labels = pickle.load(f)

    if include_time:
        def add_time(x):
            ts = (T / x.shape[1]) * jnp.repeat(
                jnp.arange(x.shape[1])[None, :], x.shape[0], axis=0
            )
            return jnp.concatenate([ts[:, :, None], x], axis=2)

        train_data = add_time(train_data)
        val_data = add_time(val_data)
        test_data = add_time(test_data)

    data = (train_data, val_data, test_data)
    labels = (train_labels, val_labels, test_labels)

    return dataset_generator(
        name,
        data,
        labels,
        stepsize,
        depth,
        include_time,
        T,
        idxs=None,
        use_presplit=True,
        key=key,
    )

def load_mnist_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
        data = data.astype(np.float32) / 255.0
        # Reshape for time series: [num, timesteps, features]
        return data[:, :, None]

def load_mnist_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def create_smnist_dataset(
    data_dir,
    name,
    use_idxs,
    use_presplit,
    stepsize,
    depth,
    include_time,
    T,
    *,
    key,
):
    train_images = load_mnist_images(data_dir+"/processed/MNIST/raw/train-images-idx3-ubyte")
    train_labels = load_mnist_labels(data_dir+ "/processed/MNIST/raw/train-labels-idx1-ubyte")
    test_images = load_mnist_images(data_dir+ "/processed/MNIST/raw/t10k-images-idx3-ubyte")
    test_labels = load_mnist_labels(data_dir+ "/processed/MNIST/raw/t10k-labels-idx1-ubyte")

    # Optionally add time as a feature
    if include_time:
        def add_time(data, T):
            ts = (T / data.shape[1]) * jnp.repeat(
                jnp.arange(data.shape[1])[None, :], data.shape[0], axis=0
            )
            return jnp.concatenate([ts[:, :, None], data], axis=2)
        train_images = add_time(train_images, T)
        test_images = add_time(test_images, T)

    # One-hot encode labels
    train_labels_oh = jnp.zeros((len(train_labels), 10))
    train_labels_oh = train_labels_oh.at[jnp.arange(len(train_labels)), train_labels].set(1)
    test_labels_oh = jnp.zeros((len(test_labels), 10))
    test_labels_oh = test_labels_oh.at[jnp.arange(len(test_labels)), test_labels].set(1)

    # Train/val split
    N = train_images.shape[0]
    permkey, key = jr.split(key)
    idxs = jr.permutation(permkey, N)
    bound = int(N * 0.85)
    tr_idx, val_idx = idxs[:bound], idxs[bound:]
    train_data, val_data = train_images[tr_idx], train_images[val_idx]
    train_lbls, val_lbls = train_labels_oh[tr_idx], train_labels_oh[val_idx]

    data = (train_data, val_data, test_images)
    labels = (train_lbls, val_lbls, test_labels_oh)

    return dataset_generator(
        "smnist",
        data,
        labels,
        stepsize,
        depth,
        include_time,
        T,
        use_presplit=True,
        key=key,
    )


def load_shd_npy(data_dir, prefix='train'):
    X = np.load(data_dir + f"/processed/SHD/{prefix}X_4ms.npy")  # shape: [N, T, F]
    y = np.load(data_dir + f"/processed/SHD/{prefix}Y_4ms.npy")  # shape: [N] or [N, 1]
    y = y.squeeze()
    return X.astype(np.float32), y.astype(np.int32)

def create_shd_dataset(
    data_dir,
    name,
    use_idxs,
    use_presplit,
    stepsize,
    depth,
    include_time,
    T,
    *,
    key,
):
    train_X, train_y = load_shd_npy(data_dir, prefix='train')
    test_X, test_y = load_shd_npy(data_dir, prefix='test')

    # Optionally add time as a feature
    if include_time:
        def add_time(data, T):
            ts = (T / data.shape[1]) * jnp.repeat(
                jnp.arange(data.shape[1])[None, :, None], data.shape[0], axis=0
            )  # [N, T, 1]
            return jnp.concatenate([ts, data], axis=2)  # [N, T, F+1]
        train_X = add_time(train_X, T)
        test_X = add_time(test_X, T)

    # One-hot labels
    num_classes = int(np.max(np.concatenate([train_y, test_y]))) + 1
    train_y_oh = jnp.zeros((len(train_y), num_classes))
    train_y_oh = train_y_oh.at[jnp.arange(len(train_y)), train_y].set(1)
    test_y_oh = jnp.zeros((len(test_y), num_classes))
    test_y_oh = test_y_oh.at[jnp.arange(len(test_y)), test_y].set(1)

    # Train/val split (85%/15%)
    N = train_X.shape[0]
    permkey, key = jr.split(key)
    idxs = jr.permutation(permkey, N)
    bound = int(N * 0.85)
    tr_idx, val_idx = idxs[:bound], idxs[bound:]
    train_data, val_data = train_X[tr_idx], train_X[val_idx]
    train_lbls, val_lbls = train_y_oh[tr_idx], train_y_oh[val_idx]

    data = (train_data, val_data, test_X)
    labels = (train_lbls, val_lbls, test_y_oh)

    return dataset_generator(
        "shd",
        data,
        labels,
        stepsize,
        depth,
        include_time,
        T,
        use_presplit=True,
        key=key,
    )


def create_toy_dataset(data_dir, name, stepsize, depth, include_time, T, *, key):
    with open(data_dir + "/processed/toy/signature/data.pkl", "rb") as f:
        data = pickle.load(f)
    with open(data_dir + "/processed/toy/signature/labels.pkl", "rb") as f:
        labels = pickle.load(f)
    if name == "signature1":
        labels = ((jnp.sign(labels[0][:, 2]) + 1) / 2).astype(int)
    elif name == "signature2":
        labels = ((jnp.sign(labels[1][:, 2, 5]) + 1) / 2).astype(int)
    elif name == "signature3":
        labels = ((jnp.sign(labels[2][:, 2, 5, 0]) + 1) / 2).astype(int)
    elif name == "signature4":
        labels = ((jnp.sign(labels[3][:, 2, 5, 0, 3]) + 1) / 2).astype(int)
    onehot_labels = jnp.zeros((len(labels), len(jnp.unique(labels))))
    onehot_labels = onehot_labels.at[jnp.arange(len(labels)), labels].set(1)
    idxs = None

    if include_time:
        ts = (T / data.shape[1]) * jnp.repeat(
            jnp.arange(data.shape[1])[None, :], data.shape[0], axis=0
        )
        data = jnp.concatenate([ts[:, :, None], data], axis=2)

    return dataset_generator(
        "toy", data, onehot_labels, stepsize, depth, include_time, T, idxs, key=key
    )


def create_ppg_dataset(
    data_dir, use_presplit, stepsize, depth, include_time, T, *, key
):
    with open(data_dir + "/processed/PPG/ppg/X_train.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(data_dir + "/processed/PPG/ppg/y_train.pkl", "rb") as f:
        train_labels = pickle.load(f)
    with open(data_dir + "/processed/PPG/ppg/X_val.pkl", "rb") as f:
        val_data = pickle.load(f)
    with open(data_dir + "/processed/PPG/ppg/y_val.pkl", "rb") as f:
        val_labels = pickle.load(f)
    with open(data_dir + "/processed/PPG/ppg/X_test.pkl", "rb") as f:
        test_data = pickle.load(f)
    with open(data_dir + "/processed/PPG/ppg/y_test.pkl", "rb") as f:
        test_labels = pickle.load(f)

    if include_time:
        ts = (T / train_data.shape[1]) * jnp.repeat(
            jnp.arange(train_data.shape[1])[None, :], train_data.shape[0], axis=0
        )
        train_data = jnp.concatenate([ts[:, :, None], train_data], axis=2)
        ts = (T / val_data.shape[1]) * jnp.repeat(
            jnp.arange(val_data.shape[1])[None, :], val_data.shape[0], axis=0
        )
        val_data = jnp.concatenate([ts[:, :, None], val_data], axis=2)
        ts = (T / test_data.shape[1]) * jnp.repeat(
            jnp.arange(test_data.shape[1])[None, :], test_data.shape[0], axis=0
        )
        test_data = jnp.concatenate([ts[:, :, None], test_data], axis=2)

    if use_presplit:
        data = (train_data, val_data, test_data)
        labels = (train_labels, val_labels, test_labels)
    else:
        data = jnp.concatenate((train_data, val_data, test_data), axis=0)
        labels = jnp.concatenate((train_labels, val_labels, test_labels), axis=0)

    return dataset_generator(
        "ppg",
        data,
        labels,
        stepsize,
        depth,
        include_time,
        T,
        inmemory=False,
        use_presplit=use_presplit,
        key=key,
    )


def create_ucihar_dataset(
    data_dir,
    use_idxs,
    use_presplit,
    stepsize,
    depth,
    include_time,
    T,
    *,
    key,
):
    import pickle
    import jax.numpy as jnp
    # Load processed data
    with open(data_dir + "/data/ucihar/processed/UCIHAR/data.pkl", "rb") as f:
        data = pickle.load(f)
    with open(data_dir + "/data/ucihar/processed/UCIHAR/labels.pkl", "rb") as f:
        labels = pickle.load(f)
    # Optionally load original_idxs
    if use_idxs:
        with open(data_dir + "/data/ucihar/processed/UCIHAR/original_idxs.pkl", "rb") as f:
            idxs = pickle.load(f)
    else:
        idxs = None

    # Optionally add time as a feature
    if include_time:
        ts = (T / data.shape[1]) * jnp.repeat(
            jnp.arange(data.shape[1])[None, :], data.shape[0], axis=0
        )
        data = jnp.concatenate([ts[:, :, None], data], axis=2)

    # One-hot encode labels if classification
    num_classes = int(jnp.max(labels)) + 1
    onehot_labels = jnp.zeros((len(labels), num_classes))
    onehot_labels = onehot_labels.at[jnp.arange(len(labels)), labels].set(1)

    return dataset_generator(
        "ucihar",
        data,
        onehot_labels,
        stepsize,
        depth,
        include_time,
        T,
        idxs=idxs,
        use_presplit=use_presplit,
        key=key,
    )

def create_hhar_dataset(
    data_dir,
    use_idxs,
    use_presplit,
    stepsize,
    depth,
    include_time,
    T,
    *,
    key,
):
    import pickle
    import jax.numpy as jnp
    # Load processed data
    with open(data_dir + "/data/ucihar/processed/HHAR/data.pkl", "rb") as f:
        data = pickle.load(f)
    with open(data_dir + "/data/ucihar/processed/HHAR/labels.pkl", "rb") as f:
        labels = pickle.load(f)
    # Optionally load original_idxs
    if use_idxs:
        with open(data_dir + "/data/ucihar/processed/HHAR/original_idxs.pkl", "rb") as f:
            idxs = pickle.load(f)
    else:
        idxs = None

    # Optionally add time as a feature
    if include_time:
        ts = (T / data.shape[1]) * jnp.repeat(
            jnp.arange(data.shape[1])[None, :], data.shape[0], axis=0
        )
        data = jnp.concatenate([ts[:, :, None], data], axis=2)

    # One-hot encode labels if classification
    num_classes = int(jnp.max(labels)) + 1
    onehot_labels = jnp.zeros((len(labels), num_classes))
    onehot_labels = onehot_labels.at[jnp.arange(len(labels)), labels].set(1)

    return dataset_generator(
        "hhar",
        data,
        onehot_labels,
        stepsize,
        depth,
        include_time,
        T,
        idxs=idxs,
        use_presplit=use_presplit,
        key=key,
    )

def create_shar_dataset(
    data_dir,
    use_idxs,
    use_presplit,
    stepsize,
    depth,
    include_time,
    T,
    *,
    key,
):
    import pickle
    import jax.numpy as jnp
    # Load processed data
    with open(data_dir + "/data/UniMiB-SHAR/processed/SHAR/data.pkl", "rb") as f:
        data = pickle.load(f)
    with open(data_dir + "/data/UniMiB-SHAR/processed/SHAR/labels.pkl", "rb") as f:
        labels = pickle.load(f)
    # Optionally load original_idxs
    if use_idxs:
        with open(data_dir + "/data/ucihar/processed/SHAR/original_idxs.pkl", "rb") as f:
            idxs = pickle.load(f)
    else:
        idxs = None

    # Optionally add time as a feature
    if include_time:
        ts = (T / data.shape[1]) * jnp.repeat(
            jnp.arange(data.shape[1])[None, :], data.shape[0], axis=0
        )
        data = jnp.concatenate([ts[:, :, None], data], axis=2)

    # One-hot encode labels if classification
    num_classes = int(jnp.max(labels)) + 1
    onehot_labels = jnp.zeros((len(labels), num_classes))
    onehot_labels = onehot_labels.at[jnp.arange(len(labels)), labels].set(1)

    return dataset_generator(
        "shar",
        data,
        onehot_labels,
        stepsize,
        depth,
        include_time,
        T,
        idxs=idxs,
        use_presplit=use_presplit,
        key=key,
    )

def create_dataset(
    data_dir,
    name,
    use_idxs,
    use_presplit,
    stepsize,
    depth,
    include_time,
    T,
    *,
    key,
):
    uea_subfolders = [
        f.name for f in os.scandir(data_dir + "/processed/UEA") if f.is_dir()
    ]
    # toy_subfolders = [
    #     f.name for f in os.scandir(data_dir + "/processed/toy") if f.is_dir()
    # ]

    if name in uea_subfolders:
        return create_uea_dataset(
            data_dir,
            name,
            use_idxs,
            use_presplit,
            stepsize,
            depth,
            include_time,
            T,
            key=key,
        )

    reg_data = [
        "ETTh1",
        "ETTh2",
        "ETTm1",
        "ETTm2",
        "ECL",
        "Weather",
        "Traffic",
        "Exchange",
    ]
    reg_subfolders = []
    for dataset_name in reg_data:
        folder_path = os.path.join(data_dir, "processed", dataset_name)
        if os.path.isdir(folder_path):
            reg_subfolders.append(dataset_name)
    if name in reg_subfolders:
        return create_long_term_prediction_dataset(
            data_dir,
            name,
            use_idxs,
            use_presplit,
            stepsize,
            depth,
            include_time,
            T,
            key=key,
        )

    # elif name[:-1] in toy_subfolders:
    #     return create_toy_dataset(
    #         data_dir, name, stepsize, depth, include_time, T, key=key
    #     )
    if name == "smnist":
        return create_smnist_dataset(
            data_dir,
            name,
            use_idxs,
            use_presplit,
            stepsize,
            depth,
            include_time,
            T,
            key=key,
        )
    
    elif name == "shd":
        return create_shd_dataset(
            data_dir,
            name,
            use_idxs,
            use_presplit,
            stepsize,
            depth,
            include_time,
            T,
            key=key,
        )
    
    elif name == "ppg":
        return create_ppg_dataset(
            data_dir, use_presplit, stepsize, depth, include_time, T, key=key
        )
    
    elif name.startswith("ppg_") and name[4:].isdigit():
        return create_ppg_dataset(
            data_dir, use_presplit, stepsize, depth, include_time, T, key=key
        )
    elif name == "ucihar":
        return create_ucihar_dataset(
            data_dir,
            use_idxs,
            use_presplit,
            stepsize,
            depth,
            include_time,
            T,
            key=key,
        )
    elif name == "shar":
        return create_shar_dataset(
            data_dir,
            use_idxs,
            use_presplit,
            stepsize,
            depth,
            include_time,
            T,
            key=key,
        )
    elif name == "hhar":
        return create_hhar_dataset(
            data_dir,
            use_idxs,
            use_presplit,
            stepsize,
            depth,
            include_time,
            T,
            key=key,
        )
    else:
        raise ValueError(f"Dataset {name} not found in UEA folder and not toy dataset")
