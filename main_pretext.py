import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from cl4ef import CL4EF
import AETA_data
from utils import init_dl_program, data_dropout

def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AETA', help='The dataset name')
    parser.add_argument('--dataroot', type=str, default='./datasets', help='path of data')
    parser.add_argument('--data_type', type=str, default='merge') 
    parser.add_argument('--fea_use', type=str, default='abs_mean')
    parser.add_argument('--sample', type=str, default='undersampling') 
    parser.add_argument('--run_name', type=str, default='checkpoints', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, default='AETA', help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=3, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch_size', type=int, default=16, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.00001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr_dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max_train_length', type=int, default=1008, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=600, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=50, help='The number of epochs')
    parser.add_argument('--save_every', type=int, default=1, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=77, help='The random seed')
    parser.add_argument('--max_threads', type=int, default=8, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    
    parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='location of model checkpoints')
    parser.add_argument('--pred_len', type=int, default=1008, help='prediction sequence length')
    parser.add_argument('--seq_len', type=int, default=1008, help='input sequence length of Informer encoder')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    
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
    

    args = parser.parse_args()
    
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    
    print('Loading data... ', end='')
    if args.loader == 'AETA':
        task_type = 'forecasting'
        train_data, train_labels = AETA_data.load_AETA_pretext(args, 'train')
        test_data, test_labels = AETA_data.load_AETA_pretext(args, 'test')
    else:
        raise ValueError(f"Unknown loader {args.loader}.")
        
        
    if args.irregular > 0:
        if task_type == 'classification':
            train_data = data_dropout(train_data, args.irregular)
            test_data = data_dropout(test_data, args.irregular)
        else:
            raise ValueError(f"Task type {task_type} is not supported when irregular>0.")
    print('done')
    
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    
    run_dir = os.path.join(args.checkpoints, args.fea_use)
    
    os.makedirs(run_dir, exist_ok=True)
    
    t = time.time()
    
    model = CL4EF(
        args,
        input_dims=args.enc_in,
        device=device,
        **config
    )
    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )
    model.save(f'{run_dir}/model.pkl')

    model.plot_loss(loss_log, run_dir, 'train_loss.png')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    # if args.eval:
    #     if task_type == 'classification':
    #         out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')
    #     elif task_type == 'forecasting':
    #         out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)
    #     elif task_type == 'anomaly_detection':
    #         out, eval_res = tasks.eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
    #     elif task_type == 'anomaly_detection_coldstart':
    #         out, eval_res = tasks.eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
    #     else:
    #         assert False
    #     pkl_save(f'{run_dir}/out.pkl', out)
    #     pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
    #     print('Evaluation result:', eval_res)

    print("Finished.")
