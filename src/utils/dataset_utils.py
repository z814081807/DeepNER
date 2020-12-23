import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, task_type, features, mode, **kwargs):

        self.nums = len(features)

        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks).float() for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]

        self.labels = None
        self.start_ids, self.end_ids = None, None
        self.ent_type = None
        self.pseudo = None
        if mode == 'train':
            self.pseudo = [torch.tensor(example.pseudo).long() for example in features]
            if task_type == 'crf':
                self.labels = [torch.tensor(example.labels) for example in features]
            else:
                self.start_ids = [torch.tensor(example.start_ids).long() for example in features]
                self.end_ids = [torch.tensor(example.end_ids).long() for example in features]

        if kwargs.pop('use_type_embed', False):
            self.ent_type = [torch.tensor(example.ent_type) for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index],
                'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index]}

        if self.ent_type is not None:
            data['ent_type'] = self.ent_type[index]

        if self.labels is not None:
            data['labels'] = self.labels[index]

        if self.pseudo is not None:
            data['pseudo'] = self.pseudo[index]

        if self.start_ids is not None:
            data['start_ids'] = self.start_ids[index]
            data['end_ids'] = self.end_ids[index]

        return data



