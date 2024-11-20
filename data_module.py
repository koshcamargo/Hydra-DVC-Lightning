import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import rowmean  # Ваш модуль на C++

class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.loadtxt(data_path, delimiter=',')  # Пример: загрузка данных из CSV

    def __getitem__(self, index):
        row = self.data[index]
        mean_value = rowmean.row_mean(row.tolist())  # Используем ваш C++ биндинг
        return torch.tensor(row, dtype=torch.float32), torch.tensor(mean_value, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

class CustomDataModule(LightningDataModule):
    def __init__(self, data_path, batch_size):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = CustomDataset(self.data_path)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

