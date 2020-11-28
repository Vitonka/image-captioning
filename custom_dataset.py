import os
import json
from collections import namedtuple
from torch.utils.data import Dataset


Caption = namedtuple('Caption', ['text', 'image_file_name'])


class SimpleCaptionsDataset(Dataset):
    def __init__(self, images_path, annotations_path, w2i_path, i2w_path, *args, **kwargs):
        super(SimpleCaptionsDataset, self).__init__(*args, **kwargs)
        self._images_path = images_path

        with open(annotations_path) as f:
            annotations = json.load(f)

        image_id_to_file_name = {}
        for image in annotations['images']:
            image_id_to_file_name[image['id']] = image['file_name']

        self.captions = []
        for annotation in annotations['annotations']:
            self.captions.append(Caption(text=annotation['caption'], image_file_name=image_id_to_file_name[annotation['image_id']]))

        with open(w2i_path) as f:
            self.w2i = json.load(f)

        with open(i2w_path) as f:
            self.i2w = json.load(f)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.captions[idx]
