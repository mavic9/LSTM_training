import torch
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import numpy as np
from einops import rearrange
import random



def getTable(table):
    df = pd.read_csv(table, sep="\t")
    df = df.drop(["Unnamed: 0"], axis=1)
    df = df.sort_values(["tube", "day"])
    return df

class CustomDataset(IterableDataset):
    def __init__(self, df, start_day, look_back, days, features, val=False, ahead=1, step=1, shuffle=False):
        self.df = df
        self.start_day = start_day
        self.look_back = look_back
        self.days = days
        self.features = features
        self.ahead = ahead
        self.val = val
        # Internal state to keep track of order across epochs:
        self.sequences = None
        self.current_index = 0
        self.shuffle = shuffle
        self.step = step

    def _generate_sequences(self):
        sequences = []
        # Create a list of back values and shuffle it only once per epoch.
        back_list = list(range(self.start_day, self.look_back + 1, self.step))
        random.shuffle(back_list)
        for back in back_list:
            # Get a copy of the unique tube IDs and shuffle them.
            tube_list = self.df["tube"].unique().copy()
            np.random.shuffle(tube_list)
            for tube in tube_list:
                tube_df = self.df[self.df['tube'] == tube]
                if self.val:
                    if tube_df.shape[0] <= 50:
                        continue
                else:
                    if tube_df.shape[0] < back:
                        continue
                data = tube_df.sort_values(['day']).copy()
                data = data.drop(columns=['day', 'tube'])
                data = data.reset_index(drop=True)
                scaled_data = data.to_numpy()
                # Create sequences based on the current "back" value and ahead.
                for i in range(0, len(scaled_data) - back - self.ahead):
                    # Check if the last feature (assumed to mark tube id or transfer indicator)
                    if scaled_data[i + back - 1, -1] == scaled_data[i + back + self.ahead - 1 , -1]:
                        a = scaled_data[i:(i+back), :-2]
                        b = scaled_data[i:(i+back+self.ahead), :-2]
                        # Skip if the conditions aren’t met.
                        if len(b) < (self.ahead+1) or np.any(a[-1] != b[-(self.ahead+1)]) or b[-self.ahead][-1] != b[-1][-1]:
                            continue
                        X_tensor = torch.tensor(a, dtype=torch.float32)
                        y_tensor = torch.tensor(b, dtype=torch.float32)
                        sequences.append((X_tensor, y_tensor))
                        if len(sequences) % 100_000 == 0:
                            print(f"Collected {len(sequences)} sequences")
        print(f"In total {len(sequences)} sequences are collected")
        return sequences

    def __iter__(self):
        # Build the sequence list once per epoch.
        if self.sequences is None:
            self.sequences = self._generate_sequences()

        # Yield sequences starting from the current index.
        for idx in range(self.current_index, len(self.sequences)):
            yield self.sequences[idx]
            self.current_index = idx + 1

        # Once we've yielded all sequences, reset for the next epoch.
        if self.current_index >= len(self.sequences):
            self.current_index = 0
            if self.shuffle:
                np.random.shuffle(self.sequences)
                print(f"Shuffled {len(self.sequences)} sequences")

def collate_fn_factory(look_back, features, initial_values=0.0, ahead=10):
    def collate_fn(batch):
        X_list, y_list = zip(*batch)
        batch_size = len(X_list)
        X_padded = torch.full((batch_size, look_back, features), initial_values, dtype=torch.float32)
        y_padded = torch.full((batch_size, look_back, features), initial_values, dtype=torch.float32)
        for idx, seq in enumerate(X_list):
            seq_len = seq.shape[0]
            if seq_len <= look_back:
                X_padded[idx, -seq_len:, :] = seq
                # does not affect the loss error
                # X_padded[idx, :-seq_len, -3] = seq[0, -3]
                # X_padded[idx, :-seq_len, -2] = seq[0, -2]
                # X_padded[idx, :-seq_len, -1] = seq[0, -1]
            else:
                X_padded[idx, :, :] = seq[-look_back:, :]
        for idx, seq in enumerate(y_list):
            seq_len = seq.shape[0]
            if seq_len <= look_back:
                y_padded[idx, -seq_len:, :] = seq
                # y_padded[idx, :-seq_len, -3] = seq[0, -3]
                # y_padded[idx, :-seq_len, -2] = seq[0, -2]
                # y_padded[idx, :-seq_len, -1] = seq[0, -1]
            else:
                y_padded[idx, :, :] = seq[-look_back:, :]
        return X_padded, y_padded
    return collate_fn
