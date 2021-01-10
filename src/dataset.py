import os
import json
import h5py
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random
from collections import defaultdict


ROOT = '../datasets'


def load_npy_images(path):
    return np.load(path)


def load_h5py_images(path):
    h5file = h5py.File(path, 'r')
    return h5file['data']


IMAGES_LOADERS = {
    'npy': load_npy_images,
    'h5py': load_h5py_images
}


class RandomCaptionByImageDataset(Dataset):
    def __init__(self, annotations_path, images_path, images_loader):

        with open(annotations_path) as f:
            annotations = json.load(f)
        self._idx_to_captions = defaultdict(list)
        for annotation in annotations['annotations']:
            idx = annotation['idx']
            self._idx_to_captions[idx].append(annotation['caption'])

        self._images = images_loader(images_path)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = self._images[idx]

        captions_for_image = self._idx_to_captions[idx]
        caption = random.choice(captions_for_image)

        return torch.tensor(image, dtype=torch.float32), \
            torch.tensor(caption, dtype=torch.int64)


class CaptionsByImageDataset(Dataset):
    def __init__(self, annotations_path, images_path, images_loader):

        with open(annotations_path) as f:
            annotations = json.load(f)
        self._idx_to_captions = defaultdict(list)
        for annotation in annotations['annotations']:
            idx = annotation['idx']
            self._idx_to_captions[idx].append(annotation['caption'])

        self._images = images_loader(images_path)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = self._images[idx]
        captions_for_image = self._idx_to_captions[idx]

        return torch.tensor(image, dtype=torch.float32), \
            captions_for_image


def get_coco_datasets(config):
    with open(os.path.join(ROOT, config['images_config_path'])) as f:
        images_config = json.load(f)
    with open(os.path.join(ROOT, config['annotations_config_path'])) as f:
        annotations_config = json.load(f)

    annotations_root_path = \
        os.path.join(
            ROOT, 'coco', 'annotations',
            annotations_config['out_data_folder'])
    annotations_path = \
        os.path.join(
            annotations_root_path,
            'captions_{0}2014.json')
    images_root_path = \
        os.path.join(
            ROOT, 'coco', 'images',
            images_config['out_data_folder'])
    images_path = \
        os.path.join(
            images_root_path,
            '{0}.' + images_config['data_type'])
    images_loader = IMAGES_LOADERS[images_config['data_type']]

    return (
        RandomCaptionByImageDataset(
            annotations_path.format('train'),
            images_path.format('train'),
            images_loader),
        CaptionsByImageDataset(
            annotations_path.format('val'),
            images_path.format('val'),
            images_loader),
        CaptionsByImageDataset(
            annotations_path.format('test'),
            images_path.format('test'),
            images_loader))


def collate_fn_train_packed(batch):
    images_list = []
    texts_list = []
    for image, texts in batch:
        if not isinstance(texts[0], list):
            texts = [texts]
        for text in texts:
            images_list.append(image)
            texts_list.append(text)

    images_list, texts_list = \
        list(zip(*sorted(
            zip(images_list, texts_list),
            key=lambda x: x[1].shape[0], reverse=True)))

    inputs = [text[:-1] for text in texts_list]
    outputs = [text[1:] for text in texts_list]

    packed_inputs = \
        torch.nn.utils.rnn.pack_sequence(inputs, enforce_sorted=True)
    packed_outputs = \
        torch.nn.utils.rnn.pack_sequence(outputs, enforce_sorted=True)
    return torch.stack(images_list), packed_inputs, packed_outputs


def collate_fn_train_padded(batch):
    images_list = []
    texts_list = []
    max_len = 0
    for image, text in batch:
        images_list.append(image)
        texts_list.append(text)
        max_len = max(max_len, len(text))
    texts_tensors = []
    for text in texts_list:
        matrix = np.zeros(max_len, dtype='int64')
        matrix[0:len(text)] = text
        texts_tensors.append(torch.tensor(matrix))

    inputs = [text[:-1] for text in texts_tensors]
    outputs = [text[1:] for text in texts_tensors]

    return torch.stack(images_list), torch.stack(inputs), torch.stack(outputs)


def collate_fn_test(batch):
    images_list = []
    texts_list = []
    for image, texts in batch:
        images_list.append(image)
        texts_list.append(texts)

    return torch.stack(images_list), texts_list


def get_coco_dataloaders(config):
    train_dataset, val_dataset, test_dataset = get_coco_datasets(config)

    assert config['data_mode'] == 'packed' or config['data_mode'] == 'padded'
    if config['data_mode'] == 'packed':
        collate_fn_train = collate_fn_train_packed
    elif config['data_mode'] == 'padded':
        collate_fn_train = collate_fn_train_padded

    return (
        DataLoader(
            train_dataset, batch_size=config['train_batch_size'],
            shuffle=True, collate_fn=collate_fn_train),
        DataLoader(
            val_dataset, batch_size=config['val_batch_size'],
            shuffle=False, collate_fn=collate_fn_test),
        DataLoader(
            test_dataset, batch_size=config['test_batch_size'],
            shuffle=False, collate_fn=collate_fn_test))
