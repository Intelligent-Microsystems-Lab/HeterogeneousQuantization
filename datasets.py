import argparse, time, pickle, uuid, os, datetime

from functools import partial
import jax.numpy as jnp
from jax.experimental import optimizers
from jax import grad, jit, lax, vmap, value_and_grad, custom_vjp, random, device_put
import jax

from snn_util import vr_loss, nll_loss

def dl_create(data_set, batch_size):
    if data_set == 'Smile':
        # 700 input neurons, 250 time steps, 250 output neurons
        if os.path.exists("data/smile_data_set/input_700_250_25.pkl") and os.path.exists("data/smile_data_set/smile95.pkl"):
            with open("data/smile_data_set/input_700_250_25.pkl", 'rb') as f:
                x_train = jnp.array(pickle.load(f)).transpose()
                x_train = x_train.reshape((1, x_train.shape[0], x_train.shape[1]))
            with open("data/smile_data_set/smile95.pkl", 'rb') as f:
                y_train = jnp.array(pickle.load(f))
                y_train = y_train.reshape((1, y_train.shape[0], y_train.shape[1]))
        else:
            raise Exception("Smile data set files do not exist please place them into data/smile_data_set")
        train_dl = [(x_train, y_train)]
        test_dl = [(x_train, y_train)]
        loss_fn = vr_loss
    elif data_set == 'Yin_Yang':
        # 4 input channels and 3 classes
        from data.yin_yang_data_set.dataset import YinYangDataset, to_spike_train
        from torch.utils.data import DataLoader

        dataset_train = YinYangDataset(size=5000, seed=42, transform=to_spike_train(100))
        dataset_validation = YinYangDataset(size=1000, seed=41, transform=to_spike_train(100))
        dataset_test = YinYangDataset(size=1000, seed=40, transform=to_spike_train(100))

        train_dl = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(dataset_validation, batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(dataset_test, batch_size=1000, shuffle=False)

        loss_fn = nll_loss
    elif data_set == 'NMNIST':
        # 2* 32* 32 = 2048 and 10 classes
        from torchneuromorphic.nmnist.nmnist_dataloaders import create_events_hdf5, create_dataloader
        import torchneuromorphic.transforms as transforms

        if os.path.exists('data/nmnist/n_mnist.hdf5'):
            pass
        elif os.path.exists('data/nmnist/'):
            out = create_events_hdf5('data/nmnist', 'data/nmnist/n_mnist.hdf5')
        else:
            raise Exception("NMNIST data set does not exist, download and place raw data into data/nmnist")
        
        train_dl, test_dl = create_dataloader(
                root='data/nmnist/n_mnist.hdf5',
                batch_size=batch_size,
                ds=1,
                num_workers=4)
        loss_fn = nll_loss
    elif data_set == 'DVS_Gestures':
        # 2* 32* 32 = 2048 and 11 classes
        from torchneuromorphic.dvs_gestures.dvsgestures_dataloaders import create_events_hdf5, create_dataloader
        import torchneuromorphic.transforms as transforms

        if os.path.exists('data/dvsgesture/dvs_gestures_build19.hdf5'):
            pass
        elif os.path.exists('data/dvsgesture/raw/'):
            out = create_events_hdf5('data/dvsgesture/raw/', 'data/dvsgesture/dvs_gestures_build19.hdf5')
        else:
            raise Exception("DVS Gestures data set does not exist, download and place raw data into data/dvsgesture/raw/")
        
        train_dl, test_dl = create_dataloader(
                root='data/dvsgesture/dvs_gestures_build19.hdf5',
                batch_size=batch_size,
                ds=4,
                num_workers=4)
        loss_fn = nll_loss
    else:
        raise Exception("Unknown data set")

    return train_dl, test_dl, loss_fn