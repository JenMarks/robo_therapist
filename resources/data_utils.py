import os

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=512):
        assert os.path.isfile(file_path)
        print(f"Creating features from dataset file at {file_path}")

        with open(file_path, encoding="utf-8") as f:
            lines = [line.strip() for line in f.read().splitlines() if len(line.strip()) > 0]

        input_ids = []
        attention_mask = []
        for line in lines:
            input_tokenized = tokenizer.tokenize(" " + line+tokenizer.eos_token)
            extra_tokens = (block_size - 2) - len(input_tokenized)
            if extra_tokens < 0:
                input_tokenized = input_tokenized[:extra_tokens]
            in_ids = tokenizer.convert_tokens_to_ids(input_tokenized)
            at_mask = [1] * len(in_ids)

            input_ids.append(in_ids)
            attention_mask.append(at_mask)

        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return torch.tensor(self.input_ids[i], dtype=torch.long), torch.tensor(self.attention_mask[i], dtype=torch.long)


def pad_tensors(sample, pad_to, pad_on_dimension, pad_value):
    pad_size = list(sample.shape)
    pad_size[pad_on_dimension] = pad_to - sample.size(pad_on_dimension)
    return torch.cat([sample, torch.zeros(*pad_size, dtype=sample.dtype) + pad_value], dim=pad_on_dimension)


class PadCollate:
    def __init__(self, pad_values, dim=0):
        self.dim = dim
        self.pad_values = pad_values

    def pad_collate(self, batch):
        max_len = max(map(lambda tuple_xs: tuple_xs[0].shape[self.dim], batch))
        # stack all
        input_ids = torch.stack(
            list(
                map(
                    lambda tuple_xs: pad_tensors(tuple_xs[0], max_len, self.dim, self.pad_values["input_ids"]),
                    batch
                )
            ),
            dim=0
        )
        attention_mask = torch.stack(
            list(
                map(
                    lambda tuple_xs: pad_tensors(tuple_xs[1], max_len, self.dim, self.pad_values["attention_mask"]),
                    batch
                )
            ),
            dim=0
        )

        return input_ids, attention_mask

    def __call__(self, batch):
        return self.pad_collate(batch)
