import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import ipdb


class SeqDataset(Dataset):
    def __init__(self, file_root, max_length) -> None:
        super(SeqDataset).__init__()

        self.sentences = pd.read_csv(file_root)
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        # 字符串处理
        sentence_a = self.sentences.sentence_a[index][1:-1].split(",")
        sentence_b = self.sentences.sentence_b[index][1:-1].split(",")
        # ['3', '4', '5']
        # ['6', '7', '8']

        # listz转array
        sentence_a = np.array([int(x) for x in sentence_a])
        sentence_b = np.array([int(x) for x in sentence_b])
        # array([3, 4, 5])
        # array([6, 7, 8])

        # 补齐
        sentence_a = np.pad(sentence_a, (0, self.max_length-sentence_a.shape[0]), 'constant', constant_values=(0,0))
        sentence_b = np.pad(sentence_b, (0, self.max_length-sentence_b.shape[0]), 'constant', constant_values=(0,0))
        # array([3, 4, 5, 0, 0, 0, 0, 0, 0, 0])
        # array([6, 7, 8, 0, 0, 0, 0, 0, 0, 0])

        return sentence_a, sentence_b



if __name__ == "__main__":
    dataset = SeqDataset("./numbers.csv", 10)
    # print(dataset.__len__())
    # print(dataset.__getitem__(0))
    # print(dataset.__getitem__(6))


    dataloader = DataLoader(dataset, batch_size=4,
                        shuffle=False, num_workers=0,  collate_fn=None)

    for batch_idx, batch in enumerate(dataloader):
        src, trg = batch
        print(src.shape, trg.shape)
        print(src)
        print(trg)
