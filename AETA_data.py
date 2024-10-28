import os
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from typing import List
import time
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

    
def load_AETA_pretext(args, flag):
    root_path = os.path.join(args.dataroot, args.data_type)
    data_path = os.path.join(root_path, flag, 'data')
    label_path = os.path.join(root_path, flag, 'label')
    label_files = sorted(os.listdir(label_path))

    data_list = []
    label_list = []
    for file in label_files:
        label_file = pd.read_csv(os.path.join(label_path, file))
        label_file.drop('Unnamed: 0', axis=1, inplace=True)
        for i in range(len(label_file.values)): 
            src_all = pd.read_csv(os.path.join(data_path, label_file.iloc[i,0]))
            src_all.drop('Unnamed: 0', axis=1, inplace=True)
            src = choose_input_fea(args.data_type, args.fea_use, src_all)

            
            columns_to_exclude = ['StationID', 'TimeStamp']
            columns_to_normalize = [col for col in src.columns if col not in columns_to_exclude]
            
            data_excluded = src[columns_to_exclude]
            data_to_normalize = src[columns_to_normalize]
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data_to_normalize)
            normalized_df = pd.DataFrame(normalized_data, columns=columns_to_normalize)
            final_df = pd.concat([data_excluded, normalized_df], axis=1)
            sta_data_sel = final_df[src.columns]

            src_data = sta_data_sel.iloc[:, 2:].to_numpy()
            src_stamp = src['TimeStamp']
            src_stamp = src_stamp.map(stamp2date)
            src_stamp2date = pd.to_datetime(src_stamp)
            src_data_stamp = time_features(src_stamp2date)

            data = np.concatenate((src_data, src_data_stamp), axis=-1)

            data_list.append(data)
            label_list.append(label_file.iloc[i,1])

    data = np.array(data_list)
    label = np.array(label_list)
    return data, label



class load_AETA_downstream(Dataset):
    def __init__(self, args, flag):
        # info
        self.root_path = os.path.join(args.dataroot, args.data_type)
    
        self.args = args
        # init
        assert flag in ['train', 'test']     
        self.flag = flag

        self.__read_data__()
    
    def __read_data__(self):
    
        data_path = os.path.join(self.root_path, self.flag, 'data')
        label_path = os.path.join(self.root_path, self.flag, 'label')
        label_files = sorted(os.listdir(label_path))

        data_list = []
        label_list = []

        if self.flag == 'train':
            label_all = pd.DataFrame()
            for file in label_files:
                label_file = pd.read_csv(os.path.join(label_path, file))
                label_file.drop('Unnamed: 0', axis=1, inplace=True)
                label_all = pd.concat([label_all, label_file], axis=0)
            
            label_eq = label_all[label_all['label'] == 1]
            label_noEq = label_all[label_all['label'] == 0].sample(n=len(label_eq.index.values), replace=False)
            label_train_all = pd.concat([label_eq, label_noEq])

            
            for i in range(len(label_train_all)):
                label = label_train_all.iloc[i, 1]
                src_name = label_train_all.iloc[i, 0]
                src_all = pd.read_csv(os.path.join(data_path, src_name))
                src_all.drop('Unnamed: 0', axis=1, inplace=True)
                src = choose_input_fea(self.args.data_type, self.args.fea_use, src_all)

                columns_to_exclude = ['StationID', 'TimeStamp']
                columns_to_normalize = [col for col in src.columns if col not in columns_to_exclude]
                data_excluded = src[columns_to_exclude]
                data_to_normalize = src[columns_to_normalize]
                scaler = MinMaxScaler()
                normalized_data = scaler.fit_transform(data_to_normalize)
                normalized_df = pd.DataFrame(normalized_data, columns=columns_to_normalize)
                final_df = pd.concat([data_excluded, normalized_df], axis=1)
                sta_data_sel = final_df[src.columns]

                src_data = sta_data_sel.iloc[:, 2:].to_numpy()
                src_stamp = src['TimeStamp']
                src_stamp = src_stamp.map(stamp2date)
                src_stamp2date = pd.to_datetime(src_stamp)
                src_data_stamp = time_features(src_stamp2date)

                data = np.concatenate((src_data, src_data_stamp), axis=-1)
                    
                data_list.append(data)
                label_list.append(label)

        elif self.flag == 'test':
            for file in label_files:
                label_file = pd.read_csv(os.path.join(label_path, file))
                label_file.drop('Unnamed: 0', axis=1, inplace=True)
                for i in range(len(label_file.values)): 
                    label = label_file.iloc[i, 1]
                    src_all = pd.read_csv(os.path.join(data_path, label_file.iloc[i,0]))
                    src_all.drop('Unnamed: 0', axis=1, inplace=True)
                    src = choose_input_fea(self.args.data_type, self.args.fea_use, src_all)

                    columns_to_exclude = ['StationID', 'TimeStamp']
                    columns_to_normalize = [col for col in src.columns if col not in columns_to_exclude]
                    data_excluded = src[columns_to_exclude]
                    data_to_normalize = src[columns_to_normalize]
                    scaler = MinMaxScaler()
                    normalized_data = scaler.fit_transform(data_to_normalize)
                    normalized_df = pd.DataFrame(normalized_data, columns=columns_to_normalize)
                    final_df = pd.concat([data_excluded, normalized_df], axis=1)
                    sta_data_sel = final_df[src.columns]

                    src_data = sta_data_sel.iloc[:, 2:].to_numpy()
                    src_stamp = src['TimeStamp']
                    src_stamp = src_stamp.map(stamp2date)
                    src_stamp2date = pd.to_datetime(src_stamp)
                    src_data_stamp = time_features(src_stamp2date)

                    data = np.concatenate((src_data, src_data_stamp), axis=-1)
                    
                    data_list.append(data)
                    label_list.append(label)

        self.data = np.array(data_list)
        self.label = np.array(label_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        return data, label
    
    

def choose_input_fea(data_type, fea_use, data):
    if data_type == 'merge':
        if fea_use == 'all':
            fea_data = data
        elif fea_use == 'abs_mean':
            fea_data = pd.concat([data.iloc[:, :2], data['magn@ulf_abs_mean'], data['sound@abs_mean']], axis=1)
            return fea_data

def stamp2date(stamp):
    timeArray = time.localtime(stamp)
    date = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return date

def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        # dt.weekofyear.to_numpy(),
        dt.isocalendar().week.to_numpy(),
    ], axis=1).astype(np.float64)


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5

def time_features(dates, timeenc=1, freq='min'):
    if timeenc==0:
        dates['month'] = dates.date.apply(lambda row:row.month,1)
        dates['day'] = dates.date.apply(lambda row:row.day,1)
        dates['weekday'] = dates.date.apply(lambda row:row.weekday(),1)
        dates['hour'] = dates.date.apply(lambda row:row.hour,1)
        dates['minute'] = dates.date.apply(lambda row:row.minute,1)
        dates['minute'] = dates.minute.map(lambda x:x//15)
        freq_map = {
            'y':[],'m':['month'],'w':['month'],'d':['month','day','weekday'],
            'b':['month','day','weekday'],'h':['month','day','weekday','hour'],
            't':['month','day','weekday','hour','minute'],
        }
        return dates[freq_map[freq.lower()]].values
    if timeenc==1:
        dates = pd.to_datetime(dates.values)
        out_list = []
        for feat in time_features_from_frequency_str(freq):
            out_list.append(feat(dates)) 
        out = np.vstack(out_list).transpose(1, 0)
        out2 = np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)]).transpose(1,0)
        return out

def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)
