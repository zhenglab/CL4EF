import os
import numpy as np
import pickle
import torch
import random
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def data_dropout(arr, p):
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B*T, False, dtype=np.bool)
    ele_sel = np.random.choice(
        B*T,
        size=int(B*T*p),
        replace=False
    )
    mask[ele_sel] = True
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res

def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def plot_loss(data, path, name):
    plt.figure(figsize=(15, 7))
    plot_x = np.linspace(1, len(data), len(data))
    plt.plot(plot_x, data, marker='.')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    mkdir(path)
    plt.savefig(os.path.join(path, name))

def plot_metrics_seq2seq(data, path, name, metric):
    plt.figure(figsize=(15, 7))
    plot_x = np.linspace(1, len(data), len(data))
    plt.plot(plot_x, data, marker='.')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    mkdir(path)
    plt.savefig(os.path.join(path, name))

def save_data(data, path, name):
    df = pd.DataFrame(data)
    mkdir(path)
    df.to_csv(os.path.join(path, name))

def save_seq2seq_gpu(path, name, args, train_num, test_num, params, flops):
    with open(os.path.join(path, name), 'a') as file:
        for arg, value in sorted(vars(args).items()):
            file.writelines('Argument %s: %r \n' % (arg, value))
        file.writelines('Train num: %d \n' % train_num)
        file.writelines('Test num: %d \n' % test_num)
        file.writelines('FLOPs: %.5fM \n' % (flops / 1e6))
        file.writelines('Params: %.5fM \n' % (params / 1e6))

        file.writelines('----------End--------------- \n')

def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)
    return fpr

def false_negative_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fnr = fn / (fn + tp)
    return fnr

def save_txt_gpu_test(path, name, args, params, flops, train_label, test_label):
    with open(os.path.join(path, name), 'a') as file:
        for arg, value in sorted(vars(args).items()):
            file.writelines('Argument %s: %r \n' % (arg, value))
        file.writelines('Train_Num_eq: %d \n' % int(train_label[1]))
        file.writelines('Train_Num_noEq: %d \n' % int(train_label[0]))
        file.writelines('Test_Num_eq: %d \n' % int(test_label[1]))
        file.writelines('Test_Num_noEq: %d \n' % int(test_label[0]))
        file.writelines('FLOPs: %.5fM \n' % (flops / 1e6))
        file.writelines('Params: %.5fM \n' % (params / 1e6))
        file.writelines('----------End--------------- \n')

def plot_metrics_one(data, path, name, metric):
    plt.figure(figsize=(15, 7))
    plot_x = np.linspace(1, len(data), len(data))
    plot_y = []
    for i in range(len(data)):
        plot_y.append(data[i][1])
    plt.plot(plot_x, plot_y, marker='.')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    mkdir(path)
    plt.savefig(os.path.join(path, name))

def plot_metrics_two(data, path, name, metric):
    noEq = []
    Eq = []
    for i in range(len(data)):
        noEq.append(data[i][1])
        Eq.append(data[i][2])

    plt.figure(figsize=(30, 7))

    plt.subplot(121)
    plot_noEq = np.linspace(1, len(noEq), len(noEq))
    plt.plot(plot_noEq, noEq, marker='.')
    plt.xlabel('Epoch')
    plt.ylabel(metric + '-' + 'Aseismic')

    plt.subplot(122)
    plot_Eq = np.linspace(1, len(Eq), len(Eq))
    plt.plot(plot_Eq, Eq, marker='.')
    plt.xlabel('Epoch')
    plt.ylabel(metric + '-' + 'Earthquake')

    mkdir(path)
    plt.savefig(os.path.join(path, name))