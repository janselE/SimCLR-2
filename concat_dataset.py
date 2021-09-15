import torch


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        print(type(datasets))

    def __getitem__(self, i):
        tup = tuple(d[i] for d in self.datasets)
        return tup

    def __len__(self):
        return min(len(d) for d in self.datasets)
