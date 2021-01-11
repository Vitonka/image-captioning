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


class CaptionsByImageDataset(Dataset):
    def __init__(
            self, annotations_path, images_path,
            images_loader, captions_choice_mode):

        with open(annotations_path) as f:
            annotations = json.load(f)
        self._idx_to_captions = defaultdict(list)
        for annotation in annotations['annotations']:
            idx = annotation['idx']
            self._idx_to_captions[idx].append(annotation['caption'])

        self._images = images_loader(images_path)

        assert captions_choice_mode == 'all' or \
            captions_choice_mode == 'random'
        self._captions_choice_mode = captions_choice_mode

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = self._images[idx]
        captions_for_image = self._idx_to_captions[idx]
        if self._captions_choice_mode == 'all':
            pass
        elif self._captions_choice_mode == 'random':
            captions_for_image = [random.choice(captions_for_image)]
        else:
            assert False, 'Unknown captions choice mode'

        return image, captions_for_image


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
        CaptionsByImageDataset(
            annotations_path.format('train'),
            images_path.format('train'),
            images_loader,
            captions_choice_mode=config['train_captions_choice_mode']),
        CaptionsByImageDataset(
            annotations_path.format('val'),
            images_path.format('val'),
            images_loader,
            captions_choice_mode='all'),
        CaptionsByImageDataset(
            annotations_path.format('test'),
            images_path.format('test'),
            images_loader,
            captions_choice_mode='all'))


def collate_fn_train(batch, data_mode):
    assert data_mode == 'packed' or data_mode == 'padded'

    images_list = []
    texts_list = []
    for image, texts in batch:
        if not isinstance(texts[0], list):
            texts = [texts]
        for text in texts:
            images_list.append(torch.tensor(image, dtype=torch.float32))
            texts_list.append(text)

    images_list, texts_list = \
        list(zip(*sorted(
            zip(images_list, texts_list),
            key=lambda x: len(x[1]), reverse=True)))

    if data_mode == 'packed':
        texts_list = [torch.tensor(text) for text in texts_list]
    elif data_mode == 'padded':
        max_len = len(texts_list[0])
        padded_texts = []
        for text in texts_list:
            padded_text = np.zeros(max_len, dtype='int64')
            padded_text[0:len(text)] = text
            padded_texts.append(torch.tensor(padded_text))
        texts_list = padded_texts

    inputs = [text[:-1] for text in texts_list]
    outputs = [text[1:] for text in texts_list]

    if data_mode == 'packed':
        inputs = \
            torch.nn.utils.rnn.pack_sequence(inputs, enforce_sorted=True)
        outputs = \
            torch.nn.utils.rnn.pack_sequence(outputs, enforce_sorted=True)
    elif data_mode == 'padded':
        inputs, outputs = torch.stack(inputs), torch.stack(outputs)

    return torch.stack(images_list), inputs, outputs


def collate_fn_test(batch):
    images_list = []
    texts_list = []
    for image, texts in batch:
        images_list.append(torch.tensor(image, dtype=torch.float32))
        texts_list.append(texts)

    return torch.stack(images_list), texts_list


def get_coco_dataloaders(config):
    train_dataset, val_dataset, test_dataset = get_coco_datasets(config)

    assert config['data_mode'] == 'packed' or config['data_mode'] == 'padded'

    def collate_fn(batch):
        return collate_fn_train(batch, config['data_mode'])

    return (
        DataLoader(
            train_dataset, batch_size=config['train_batch_size'],
            shuffle=True, collate_fn=collate_fn),
        DataLoader(
            val_dataset, batch_size=1,
            shuffle=False, collate_fn=collate_fn_test),
        DataLoader(
            test_dataset, batch_size=1,
            shuffle=False, collate_fn=collate_fn_test))
