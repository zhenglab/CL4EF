import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
from models import TSEncoder
from models.losses import hierarchical_loss_fluctuation, hierarchical_loss_trend
import matplotlib.pyplot as plt

class CL4EF:
    '''The CL4EF model'''
    
    def __init__(
        self,
        args,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=16,
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None
    ):
        ''' Initialize a CL4EF model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''
        
        super().__init__()
        # self.device = torch.device("cuda:%d" % gpu if torch.cuda.is_available() else "cpu")
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        
        self._net = TSEncoder(args).to(self.device)
        self.series_decomp = series_decomp(kernel_size=25)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0
    
    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):

        print('Start Training ..........................')
        assert train_data.ndim == 3
        
        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600  

        
        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
        
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        
        loss_log = []
        
        for epoch in range(n_epochs):

            
            cum_loss = 0
            n_epoch_iters = 0
            
            for iters, batch in enumerate(train_loader):

                x = batch[0]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)
                channels = x.size(-1) - 5
                x_magn = x[:, :, :int(channels / 2)]
                x_sound = x[:, :, int(channels / 2):-5]
                x_mark = x[:, :, -5:]
                
                optimizer.zero_grad()

                magn_fluc, magn_trend = self.series_decomp(x_magn)
                sound_fluc, sound_trend = self.series_decomp(x_sound)

                out1_fluc = self._net(magn_fluc, x_mark, 'fluctuation')
                out2_fluc = self._net(sound_fluc, x_mark, 'fluctuation')

                out1_trend = self._net(magn_trend, x_mark, 'trend')
                out2_trend = self._net(sound_trend, x_mark, 'trend')

                loss_fluc = hierarchical_loss_fluctuation(
                    out1_fluc,
                    out2_fluc,
                    temporal_unit=self.temporal_unit
                )
                
                z1_type = torch.zeros(out1_trend.size(0), dtype=torch.long)  
                z2_type = torch.ones(out2_trend.size(0), dtype=torch.long)   
                loss_trend = hierarchical_loss_trend(
                    out1_trend,
                    out2_trend,
                    z1_type,
                    z2_type,
                    temporal_unit=self.temporal_unit
                )

                loss = loss_fluc + loss_trend 
                
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)
                    
                cum_loss += loss.item()
                n_epoch_iters += 1
                
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())

            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{epoch}: loss={cum_loss}")
            self.n_epochs += 1
            
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)
            
        return loss_log

    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
    
    def mkdir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    def plot_loss(self, data, path, name):
        plt.figure(figsize=(15, 7))
        plot_x = np.linspace(1, len(data), len(data))
        plt.plot(plot_x, data, marker='.')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        self.mkdir(path)
        plt.savefig(os.path.join(path, name))
    
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1) 
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

    
class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes, args):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.gpu = args.gpu

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim)
        if self.gpu is not None:
            torch.cuda.set_device(self.gpu)
            h0 = h0.cuda(self.gpu)
            c0 = c0.cuda(self.gpu)

        out, _ = self.lstm(x, (h0, c0)) 
        out = out[:, -1, :] 
        out = self.fc1(out)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        return out
    
class Info_Cls(nn.Module):
    def __init__(self, model_pre, model_cls, model_cls_name):
        super(Info_Cls, self).__init__()

        self.model_pre = model_pre.module
        self.model_cls = model_cls
        self.model_cls_name = model_cls_name
        self.map = nn.Linear(505, 1008)
    
    def forward(self, x, y, x_mark, mask=None):
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        enc_out_x = self.model_pre.enc_embedding(x, x_mark)
        # generate & apply mask
        mask = enc_out_x.new_full((enc_out_x.size(0), enc_out_x.size(1)), True, dtype=torch.bool)
        mask &= nan_mask
        enc_out_x[~mask] = 0
        enc_out_x = self.model_pre.informer(enc_out_x)
        enc_out_x = enc_out_x.permute(0, 2, 1)
        enc_out_x = enc_out_x.reshape(-1, 505)
        out_fluc = self.map(enc_out_x)
        out_fluc = out_fluc.reshape(x.size(0), 512, 1008)
        out_fluc = out_fluc.permute(0, 2, 1)
    

        nan_mask = ~y.isnan().any(axis=-1)
        y[~nan_mask] = 0
        enc_out_y = self.model_pre.enc_embedding(y, x_mark)
        # generate & apply mask
        mask = enc_out_y.new_full((enc_out_y.size(0), enc_out_y.size(1)), True, dtype=torch.bool)
        mask &= nan_mask
        enc_out_y[~mask] = 0
        out_trend = self.model_pre.autoformer(enc_out_y)


        if self.model_cls_name == 'concat':
            output = torch.concat((out_fluc, out_trend), dim=-1)

        cls_out = self.model_cls(output)

        return cls_out