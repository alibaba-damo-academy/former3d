import numpy as np
import torch


def sparse_collate_fn(batch):
    if isinstance(batch[0], dict):
        batch_size = batch.__len__()
        ans_dict = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], np.ndarray):
                ans_dict[key] = torch.stack(
                    [torch.from_numpy(sample[key]) for sample in batch], axis=0
                )
            elif isinstance(batch[0][key], torch.Tensor):
                ans_dict[key] = torch.stack([sample[key] for sample in batch], axis=0)
            elif isinstance(batch[0][key], dict):
                ans_dict[key] = sparse_collate_fn([sample[key] for sample in batch])
            else:
                ans_dict[key] = [sample[key] for sample in batch]
        return ans_dict
    else:
        batch_size = batch.__len__()
        ans_dict = tuple()
        for i in range(len(batch[0])):
            key = batch[0][i]
            if isinstance(key, np.ndarray):
                ans_dict += (
                    torch.stack(
                        [torch.from_numpy(sample[i]) for sample in batch], axis=0
                    ),
                )
            elif isinstance(key, torch.Tensor):
                ans_dict += (torch.stack([sample[i] for sample in batch], axis=0),)
            elif isinstance(key, dict):
                ans_dict += (sparse_collate_fn([sample[i] for sample in batch]),)
            else:
                ans_dict += ([sample[i] for sample in batch],)
        return ans_dict
