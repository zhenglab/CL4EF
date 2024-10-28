import os
import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile
from collections import Counter
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

# TS2Vec
parser.add_argument('--dataset', type=str, default='AETA', help='The dataset name')
parser.add_argument('--run_name', type=str, default='checkpoints', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
parser.add_argument('--loader', type=str, default='AETA', help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
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
parser.add_argument('--data_type', type=str, default='merge') 
parser.add_argument('--fea_use', type=str, default='abs_mean')
parser.add_argument('--sample', type=str, default='undersampling') 
parser.add_argument('--train_phase', type=str, default='train') 
parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='location of model checkpoints')
parser.add_argument('--results', type=str, default='./results', help='location of model checkpoints')
parser.add_argument('--pred_len', type=int, default=1008, help='prediction sequence length')
parser.add_argument('--seq_len', type=int, default=1008, help='input sequence length of Informer encoder')
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--num_workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--model_name', type=str, default='model.pkl')
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


    checkpoint_path = os.path.join(args.checkpoints, args.fea_use)
    results_path = os.path.join(args.results, args.fea_use, args.model_cls, args.model_pred_state)
    mkdir(results_path)

    train_data, train_loader = get_data(args, 'train')
    print(len(train_loader))
    test_data, test_loader = get_data(args, 'test')
    print(len(test_loader))
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    
    model_pre = TSEncoder(args).to(device)
    model_pre = torch.optim.swa_utils.AveragedModel(model_pre)

    if args.model_pred_state == 'frozen':
        checkpoint = torch.load(os.path.join(checkpoint_path, args.model_name))
        model_pre.load_state_dict(checkpoint)
        for name, params in model_pre.named_parameters():
            params.requires_grad = False
    elif args.model_pred_state == 'scratch':
        model_pre = model_pre

    if args.model_cls == 'BiLSTM':
        model_cls = BiLSTM(args.d_model*2, args.hidden_nc, args.num_layers, args.num_classes, args)    
    
    model = Info_Cls(model_pre, model_cls, 'concat')

    # set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        memory_origin = torch.cuda.memory_allocated(args.gpu)
        print('Memory_origin:', torch.cuda.memory_allocated(args.gpu))
    print(model)


    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(parameters, lr=args.lr)

    best_magn_auc = 0.
    best_sound_auc = 0.
    loss_list = []
    loss_magn_list = []
    loss_sound_list = []
    test_loss_list = []

    acc_magn_list = []
    f1_magn_list = []
    fp_magn_list = []
    fn_magn_list = []
    auc_magn_list = []

    acc_sound_list = []
    f1_sound_list = []
    fp_sound_list = []
    fn_sound_list = []
    auc_sound_list = []

    for epoch in range(args.epochs):
        loss_magn_list, loss_sound_list, loss_list, features, fea_mark = train(train_loader, model, criterion, optimizer, epoch, loss_magn_list, loss_sound_list, loss_list, args)
        true_label, pred_magn, outputs_magn, pred_sound, outputs_sound, test_loss_list  = test(test_loader, model, criterion, test_loss_list, args)
        acc_magn, f1_magn, fp_magn, fn_magn, auc_magn, f1_avg_magn = evaluate(true_label, pred_magn, outputs_magn)
        acc_sound, f1_sound, fp_sound, fn_sound, auc_sound, f1_avg_sound = evaluate(true_label, pred_sound, outputs_sound)

        acc_magn_list.append([epoch, acc_magn])
        f1_magn_list.append([epoch, f1_magn[0], f1_magn[1], f1_avg_magn])
        fp_magn_list.append([epoch, fp_magn])
        fn_magn_list.append([epoch, fn_magn])
        auc_magn_list.append([epoch, auc_magn])

        acc_sound_list.append([epoch, acc_sound])
        f1_sound_list.append([epoch, f1_sound[0], f1_sound[1], f1_avg_sound])
        fp_sound_list.append([epoch, fp_sound])
        fn_sound_list.append([epoch, fn_sound])
        auc_sound_list.append([epoch, auc_sound])

        # remember best auc and save checkpoint
        is_best_magn = auc_magn > best_magn_auc
        best_magn_auc = max(auc_magn, best_magn_auc)
        is_best_sound = auc_sound > best_sound_auc
        best_sound_auc = max(auc_sound, best_sound_auc)

        torch.save(model.state_dict(), os.path.join(results_path, 'results_{}.pth'.format(epoch)))

    save_data(loss_list, os.path.join(results_path, 'files'), 'loss.csv') 
    save_data(loss_magn_list, os.path.join(results_path, 'files'), 'loss_magn.csv')
    save_data(loss_sound_list, os.path.join(results_path, 'files'), 'loss_sound.csv')
    save_data(test_loss_list, os.path.join(results_path, 'files'), 'test_loss.csv')
    
    save_data(acc_magn_list, os.path.join(results_path, 'files'), 'accuracy_magn.csv')
    save_data(f1_magn_list, os.path.join(results_path, 'files'), 'f1_magn.csv')
    save_data(fp_magn_list, os.path.join(results_path, 'files'), 'fp_magn.csv')
    save_data(fn_magn_list, os.path.join(results_path, 'files'), 'fn_magn.csv')
    save_data(auc_magn_list, os.path.join(results_path, 'files'), 'auc_magn.csv')
    
    save_data(acc_sound_list, os.path.join(results_path, 'files'), 'accuracy_sound.csv')
    save_data(f1_sound_list, os.path.join(results_path, 'files'), 'f1_sound.csv')
    save_data(fp_sound_list, os.path.join(results_path, 'files'), 'fp_sound.csv')
    save_data(fn_sound_list, os.path.join(results_path, 'files'), 'fn_sound.csv')
    save_data(auc_sound_list, os.path.join(results_path, 'files'), 'auc_sound.csv')

    plot_loss(loss_list, os.path.join(results_path, 'figs'), 'loss.png')
    plot_loss(loss_magn_list, os.path.join(results_path, 'figs'), 'loss_magn.png')
    plot_loss(loss_sound_list, os.path.join(results_path, 'figs'), 'loss_sound.png')
    plot_loss(test_loss_list, os.path.join(results_path, 'figs'), 'test_loss.png')
    
    plot_metrics_one(acc_magn_list, os.path.join(results_path, 'figs'), 'accuracy_magn.png', 'Accuracy')
    plot_metrics_two(f1_magn_list, os.path.join(results_path, 'figs'), 'f1_magn.png', 'F1 Score')
    plot_metrics_one(fp_magn_list, os.path.join(results_path, 'figs'), 'fp_magn.png', 'False Positive Rate')
    plot_metrics_one(fn_magn_list, os.path.join(results_path, 'figs'), 'fn_magn.png', 'False Negative Rate')
    plot_metrics_one(auc_magn_list, os.path.join(results_path, 'figs'), 'auc_magn.png', 'AUC')

    plot_metrics_one(acc_sound_list, os.path.join(results_path, 'figs'), 'accuracy_sound.png', 'Accuracy')
    plot_metrics_two(f1_sound_list, os.path.join(results_path, 'figs'), 'f1_sound.png', 'F1 Score')
    plot_metrics_one(fp_sound_list, os.path.join(results_path, 'figs'), 'fp_sound.png', 'False Positive Rate')
    plot_metrics_one(fn_sound_list, os.path.join(results_path, 'figs'), 'fn_sound.png', 'False Negative Rate')
    plot_metrics_one(auc_sound_list, os.path.join(results_path, 'figs'), 'auc_sound.png', 'AUC')

    input_f = torch.randn_like(features[0:1].detach().cpu()).cuda(args.gpu)
    input_m = torch.randn_like(fea_mark[0:1].detach().cpu()).cuda(args.gpu)

    flops, params = profile(model, inputs=(input_f, input_f, input_m))
    print("FLOPs: %.2fM" % (flops / 1e6), "Params: %.5fM" % (params / 1e6))

    if args.gpu is not None:
        save_txt_gpu_test(os.path.join(results_path, 'files'), 'parameters.txt', 
                          args, params, flops, dict(Counter(train_data.label)), 
                          dict(Counter(test_data.label)))
    
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

def train(train_loader, model, criterion, optimizer, epoch, loss_magn_list, loss_sound_list, loss_list, args):
    model.train()
    running_loss = 0.0
    loss_magn_all = 0.0
    loss_sound_all = 0.0

    for i, (batch_x, labels) in enumerate(train_loader, 0):
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

        loss_magn = criterion(outputs_magn, labels)
        loss_sound = criterion(outputs_sound, labels)
        loss = loss_magn + loss_sound

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loss_magn_all += loss_magn.item()
        loss_sound_all += loss_sound.item()

        if i % 10 == 9:
            print('[Epoch:%d Iteration:%5d] loss: %.3f' % (epoch+1, i+1, running_loss / 100))
            loss_list.append(running_loss / 100)
            loss_magn_list.append(loss_magn_all / 100)
            loss_sound_list.append(loss_sound_all / 100)
            running_loss = 0.0
            loss_magn_all = 0.0
            loss_sound_all = 0.0
    return loss_magn_list, loss_sound_list, loss_list, x_magn, x_mark

def test(test_loader, model, criterion, test_loss_list, args):
    model.eval()
    running_loss = 0.0

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

            loss_magn = criterion(outputs_magn, labels)
            loss_sound = criterion(outputs_sound, labels)
            loss = loss_magn + loss_sound

            running_loss += loss.item()

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

    correct_magn = torch.sum(torch.tensor(pred_magn_list) == torch.tensor(true_label)).item()  
    correct_sound = torch.sum(torch.tensor(pred_sound_list) == torch.tensor(true_label)).item()  

    total = len(true_label)
    print('Accuracy Magn: %.2f %%' % (100 * correct_magn / total))
    print('Accuracy Sound: %.2f %%' % (100 * correct_sound / total))

    test_loss_list.append(running_loss / (i+1))


    return true_label, pred_magn_list, outputs_magn_list, pred_sound_list, outputs_sound_list, test_loss_list


def evaluate(label, pred, output):
    accuracy = accuracy_score(label, pred)
    f1 = f1_score(label, pred, average=None) 
    f1_avg = f1_score(label, pred, average='weighted')
    fp = false_positive_rate(label, pred) 
    fn = false_negative_rate(label, pred) 
    fpr, tpr, thresholds = roc_curve(label, output[:, 1], pos_label=1)
    auc_score = auc(fpr, tpr)
    return accuracy, f1, fp, fn, auc_score, f1_avg


if __name__ == '__main__':
    main()


