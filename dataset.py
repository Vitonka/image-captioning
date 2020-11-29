import os
import json
import h5py
from collections import namedtuple, defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image

Caption = namedtuple('Caption', ['text', 'image_id'])

class SimpleCaptionsDatasetBase():
    def __init__(self, annotations_path, hdf5_path, *args, **kwargs):
        self._h5file = h5py.File(hdf5_path, 'r')
        self._images_data = self._h5file['images']

        with open(annotations_path) as f:
            annotations = json.load(f)

        self._image_ids = []
        for image in annotations['images']:
            self._image_ids.append(image['id'])

        self._image_ids.sort()
        self._image_id_to_idx = {}
        for i, image_id in enumerate(self._image_ids):
            self._image_id_to_idx[image_id] = i

        self._captions = []
        self._image_id_to_captions = defaultdict(list)
        for annotation in annotations['annotations']:
            self._image_id_to_captions[annotation['image_id']].append(len(self._captions))
            self._captions.append(Caption(text=annotation['caption'], image_id=annotation['image_id']))

class SimpleCaptionsDatasetByCaption(SimpleCaptionsDatasetBase, Dataset):
    def __init__(self, *args, **kwargs):
        super(SimpleCaptionsDatasetByCaption, self).__init__(*args,**kwargs)

    def __len__(self):
        return len(self._captions)

    def __getitem__(self, idx):
        image_idx = self._image_id_to_idx[self._captions[idx].image_id]
        return (torch.from_numpy(self._images_data[image_idx]), self._captions[idx].text)

class SimpleCaptionsDatasetByImage(SimpleCaptionsDatasetBase, Dataset):
    def __init__(self, *args, **kwargs):
        super(SimpleCaptionsDatasetByImage, self).__init__(*args,**kwargs)

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, idx):
        image_id = self._image_ids[idx]
        return (torch.from_numpy(self._images_data[idx]), [self._captions[i].text for i in self._image_id_to_captions[image_id]])

def get_coco_datasets(dataset_path):
    images_path = os.path.join(dataset_path, 'images')
    annotations_path = os.path.join(dataset_path, 'annotations')

    return (
        SimpleCaptionsDatasetByImage(os.path.join(annotations_path, 'captions_train2014.json'), os.path.join(dataset_path, 'train.h5')),
        SimpleCaptionsDatasetByImage(os.path.join(annotations_path, 'captions_val2014.json'), os.path.join(dataset_path, 'val.h5')),
        SimpleCaptionsDatasetByImage(os.path.join(annotations_path, 'captions_test2014.json'), os.path.join(dataset_path, 'test.h5')))


def collate_fn_train(batch):
    images_list = []
    texts_list = []
    for image, texts in batch:
        if not isinstance(texts, list):
            texts = [texts]
        for text in texts:
            images_list.append(image)
            texts_list.append(torch.tensor(text))

    images_list, texts_list = \
        list(zip(*sorted(zip(images_list, texts_list), key=lambda x: x[1].shape[0], reverse=True)))

    inputs = [text[:-1] for text in texts_list]
    outputs = [text[1:] for text in texts_list]

    packed_inputs = torch.nn.utils.rnn.pack_sequence(inputs, enforce_sorted=True)
    packed_outputs = torch.nn.utils.rnn.pack_sequence(outputs, enforce_sorted=True)
    return torch.stack(images_list), packed_inputs, packed_outputs

def collate_fn_test(batch):
    images_list = []
    texts_list = []
    for image, texts in batch:
        images_list.append(image)
        texts_list.append(texts)

    return torch.stack(images_list), texts_list

def get_coco_dataloaders(dataset_path, train_bs, val_bs, test_bs):
    train_dataset, val_dataset, test_dataset = get_coco_datasets(dataset_path)
    return (
        DataLoader(train_dataset, batch_size=train_bs, shuffle=True, collate_fn=collate_fn_train),
        DataLoader(val_dataset, batch_size=val_bs, shuffle=False, collate_fn=collate_fn_test),
        DataLoader(test_dataset, batch_size=test_bs, shuffle=False, collate_fn=collate_fn_test))
