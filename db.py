import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.model_selection
from torch.utils.data import Dataset
import torch


class GamerPPG(Dataset):
    def __init__(self, record_list, db_dir, pre_processing):
        self.sampling_rate = 100
        self.record_list = record_list
        self.db_dir = db_dir
        self.pre_processing = pre_processing

    def __len__(self):
        return len(self.record_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        record = np.load(f'{self.db_dir}/{self.record_list[idx]}.npy', allow_pickle=True).item()

        record['label'] = int(record['label']) - 1

        if self.pre_processing:
            record = self.pre_processing(record)

        return record['ppg'], record['label']


class ToTensor(object):
    def __call__(self, sample):
        for key, value in sample.items():
            if key != 'ppg':
                sample[key] = torch.tensor(value, dtype=torch.long)
            else:
                sample[key] = torch.tensor(value, dtype=torch.float)

        return sample


class Normalization(object):
    def __call__(self, sample):
        sample['ppg'] = (sample['ppg'] - np.min(sample['ppg'])) / np.max(sample['ppg'])

        return sample


def gen_samples(root_dir, sub_list, save_dir, window):
    sampling_rate = 100
    os.makedirs(save_dir, exist_ok=True)
    label_col = 'Stanford Sleepiness Self-Assessment (1-7)'

    record_list = []
    for sub in sub_list:
        print(sub)
        ann = pd.read_csv(f'{root_dir}/{sub}-annotations.csv')
        ann = ann[ann['Event'] == label_col]

        for i in range(len(ann)):       # By hour (24+)
            ann_ = ann.iloc[i]
            d, t = ann_['Datetime'].split('T')      # d=yyyy-mm-dd, t=hh:mm:ss
            label = ann_['Value']

            ppg = pd.read_csv(f'{root_dir}/{sub}-ppg-{d}.csv')
            ppg = ppg[ppg['Time'].str[:2] == t[:2]]
            if len(ppg) > sampling_rate * window:
                print(f'From {d} {pd.to_datetime(t).hour} to {pd.to_datetime(t).hour + 1}')
                ppg = ppg[:int(len(ppg) / (sampling_rate * window)) * (sampling_rate * window)]
                ppg = ppg['Red_Signal'].to_numpy()
                ppg = ppg.reshape([-1, sampling_rate * window])

                for num, ppg_ in enumerate(ppg):
                    file_name = f'{sub}-{d}-{pd.to_datetime(t).hour}-{num}'
                    if np.isnan(ppg_).any():
                        print(f'{file_name} has NaN value.')
                    else:
                        np.save(f'{save_dir}/{file_name}', {'ppg': ppg_, 'label': label})
                        record_list.append(file_name)
            else:
                print(f'Sample {sub}-ppg-{d} from {pd.to_datetime(t).hour} to {pd.to_datetime(t).hour + 1}'
                      f' do not have enough data [{len(ppg)}].')

    np.savetxt('record_list', record_list, fmt='%s', delimiter=',')


def split(record_list):
    tr, val = sklearn.model_selection.train_test_split(record_list, test_size=.2, random_state=4)
    np.savetxt('./train', tr, fmt='%s', delimiter=',')
    np.savetxt('./val', val, fmt='%s', delimiter=',')


if __name__ == '__main__':
    gen_samples(root_dir='./archive/',
                sub_list=['gamer1', 'gamer2', 'gamer3', 'gamer4', 'gamer5'],
                save_dir='./db/',
                window=300)
    record_list = np.loadtxt('./record_list', delimiter=',', dtype='str')
    split(record_list)
    pass
