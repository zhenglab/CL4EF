import os
import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


from utils import *
import AETA_data
from torch.utils.data import DataLoader
from cl4ef import series_decomp, BiLSTM, Info_Cls
from models import TSEncoder
from focal_loss import FocalLoss

from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='AETA', help='The dataset name')
parser.add_argument('--run_name', type=str, default='checkpoints', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
parser.add_argument('--loader', type=str, default='AETA', help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
parser.add_argument('--gpu', type=int, default=2, help='The gpu no. used for training and inference (defaults to 0)')
parser.add_argument('--batch_size', type=int, default=16, help='The batch size (defaults to 8)')
parser.add_argument('--pre_batch_size', type=int, default=16, help='The batch size (defaults to 8)')
parser.add_argument('--pre_lr', type=float, default=0.00001, help='The learning rate (defaults to 0.001)')
parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate (defaults to 0.001)')
parser.add_argument('--repr_dims', type=int, default=320, help='The representation dimension (defaults to 320)')
parser.add_argument('--max_train_length', type=int, default=1008, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
parser.add_argument('--epochs', type=int, default=100, help='The number of epochs')
parser.add_argument('--save_every', type=int, default=True, help='Save the checkpoint every <save_every> iterations/epochs')
parser.add_argument('--seed', type=int, default=77, help='The random seed')
parser.add_argument('--max_threads', type=int, default=8, help='The maximum allowed number of threads used by this process')
parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    

# AETA
parser.add_argument('--dataroot', type=str, default='./datasets', help='path of data')
parser.add_argument('--data_type', type=str, default='merge') # 'magn' or 'sound' or 'merge'
parser.add_argument('--fea_use', type=str, default='abs_mean')
parser.add_argument('--num_classes', type=int, default=2) # 类别数量
parser.add_argument('--sample', type=str, default='undersampling') # 'undersampling' or 'totalOversampling' or 'underOver' or 'partOversampling'
parser.add_argument('--train_phase', type=str, default='train') # 'train' or 'test'
parser.add_argument('--results', type=str, default='./results', help='location of model checkpoints')
parser.add_argument('--pred_len', type=int, default=1008, help='prediction sequence length')
parser.add_argument('--seq_len', type=int, default=1008, help='input sequence length of Informer encoder')
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--num_workers', default=4, type=int, help='number of data loading workers')

parser.add_argument('--model_name', type=str, default='results.pth')
parser.add_argument('--model_cls', type=str, default='BiLSTM', help='network: BiLSTM | FC')
parser.add_argument('--model_pred_state', type=str, default='frozen')

parser.add_argument('--hidden_nc', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)

# Informer
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--freq', type=str, default='t',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')


def main():
    args = parser.parse_args()
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    SEED = args.seed
    torch.manual_seed(SEED)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))


    test_data, test_loader = get_data(args, 'test')
    print(len(test_loader))
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    
    model_pre = TSEncoder(args).to(device)
    model_pre = torch.optim.swa_utils.AveragedModel(model_pre)

    if args.model_cls == 'BiLSTM':
        model_cls = BiLSTM(args.d_model*2, args.hidden_nc, args.num_layers, args.num_classes, args)    
    
    model = Info_Cls(model_pre, model_cls, 'concat')

    checkpoint_magn = torch.load(os.path.join(args.results, args.model_name))
    model.load_state_dict(checkpoint_magn)
    for name, params in model_pre.named_parameters():
        params.requires_grad = False


    # set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        memory_origin = torch.cuda.memory_allocated(args.gpu)
        print('Memory_origin:', torch.cuda.memory_allocated(args.gpu))
    print(model)

    true_label, pred_magn, outputs_magn, pred_sound, outputs_sound  = test(test_loader, model, args)
    acc_magn, fp_magn, fn_magn, auc_magn, f1_avg_magn = evaluate(true_label, pred_magn, outputs_magn)
    acc_sound, fp_sound, fn_sound, auc_sound, f1_avg_sound = evaluate(true_label, pred_sound, outputs_sound)
    print('AUC: %.3f' % ((auc_magn + auc_sound)/2))
    print('Acc: %.3f' % ((acc_magn + acc_sound)/2))
    print('F1: %.3f' % ((f1_avg_magn + f1_avg_sound)/2))
    print('FPR: %.3f' % ((fp_magn + fp_sound)/2))
    print('FNR: %.3f' % ((fn_magn + fn_sound)/2))

def get_data(args, flag):

        data_set = AETA_data.load_AETA_downstream(args, flag)

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size


        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        
        return data_set, data_loader


def test(test_loader, model, args):
    model.eval()

    true_label = []
    pred_magn_list = []
    outputs_magn_list = []
    pred_sound_list = []
    outputs_sound_list = []

    with torch.no_grad():
        for i, (batch_x, labels) in enumerate(test_loader, 0):
            batch_x = batch_x.to(torch.float32)
            labels = labels.to(torch.long)

            channels = batch_x.size(-1) - 5
            x_magn = batch_x[:, :, :int(channels / 2)]
            x_sound = batch_x[:, :, int(channels / 2):-5]
            x_mark = batch_x[:, :, -5:]
            
            if args.gpu is not None:
                x_magn = x_magn.cuda(args.gpu)
                x_sound = x_sound.cuda(args.gpu)
                x_mark = x_mark.cuda(args.gpu)
                labels = labels.cuda(args.gpu)

            ser_decom = series_decomp(kernel_size = 25)
            magn_season, magn_trend = ser_decom(x_magn)
            sound_season, sound_trend = ser_decom(x_sound)

            outputs_magn = model(magn_season, magn_trend, x_mark)
            outputs_sound = model(sound_season, sound_trend, x_mark)

            pred_magn = torch.argmax(outputs_magn, dim=1)
            pred_sound = torch.argmax(outputs_sound, dim=1)

            true_label.append(labels.cpu().numpy())
            pred_magn_list.append(pred_magn.cpu().numpy())
            outputs_magn_list.append(outputs_magn.cpu().numpy())
            pred_sound_list.append(pred_sound.cpu().numpy())
            outputs_sound_list.append(outputs_sound.cpu().numpy())

    true_label = np.concatenate(true_label, axis=0)
    pred_magn_list = np.concatenate(pred_magn_list, axis=0)
    outputs_magn_list = np.concatenate(outputs_magn_list, axis=0)
    pred_sound_list = np.concatenate(pred_sound_list, axis=0)
    outputs_sound_list = np.concatenate(outputs_sound_list, axis=0)

    return true_label, pred_magn_list, outputs_magn_list, pred_sound_list, outputs_sound_list


def evaluate(label, pred, output):
    accuracy = accuracy_score(label, pred)
    f1 = f1_score(label, pred, average=None) 
    f1_avg = f1_score(label, pred, average='weighted')
    fp = false_positive_rate(label, pred) 
    fn = false_negative_rate(label, pred) 
    fpr, tpr, thresholds = roc_curve(label, output[:, 1], pos_label=1) 
    auc_score = auc(fpr, tpr)
    return accuracy, fp, fn, auc_score, f1_avg


if __name__ == '__main__':
    main()


