import os
import json
from collections import namedtuple, defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image

Caption = namedtuple('Caption', ['text', 'image_id'])

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class SimpleCaptionsDatasetBase():
    def __init__(self, images_path, annotations_path, transform=None, *args, **kwargs):
        self._images_path = images_path
        self._transform = transform

        with open(annotations_path) as f:
            annotations = json.load(f)

        self._image_id_to_file_name = {}
        for image in annotations['images']:
            self._image_id_to_file_name[image['id']] = image['file_name']

        self._captions = []
        self._image_id_to_captions = defaultdict(list)
        for annotation in annotations['annotations']:
            self._image_id_to_captions[annotation['image_id']].append(len(self._captions))
            self._captions.append(Caption(text=annotation['caption'], image_id=annotation['image_id']))

class SimpleCaptionsTrainDataset(SimpleCaptionsDatasetBase, Dataset):
    def __init__(self, *args, **kwargs):
        super(SimpleCaptionsTrainDataset, self).__init__(*args,**kwargs)

    def __len__(self):
        return len(self._captions)

    def __getitem__(self, idx):
        image_filename = self._image_id_to_file_name[self._captions[idx].image_id]
        image = Image.open(os.path.join(self._images_path, image_filename))
        if self._transform is not None:
            image = self._transform(image)
        return (image, self._captions[idx].text)

class SimpleCaptionsTestDataset(SimpleCaptionsDatasetBase, Dataset):
    def __init__(self, *args, **kwargs):
        super(SimpleCaptionsTestDataset, self).__init__(*args,**kwargs)

    def __len__(self):
        return len(self._image_id_to_file_name)

    def __getitem__(self, idx):
        image_id = list(self._image_id_to_file_name.keys())[idx]
        image_filename = self._image_id_to_file_name[image_id]
        image = Image.open(os.path.join(self._images_path, image_filename))
        if self._transform is not None:
            image = self._transform(image)
        return (image, [self._captions[i].text for i in self._image_id_to_captions[image_id]])

def get_coco_datasets(dataset_path):
    images_path = os.path.join(dataset_path, 'images')
    annotations_path = os.path.join(dataset_path, 'annotations')

    return (
        SimpleCaptionsTrainDataset(os.path.join(images_path, 'train2014'), os.path.join(annotations_path, 'captions_train2014.json'), IMAGE_TRANSFORM),
        SimpleCaptionsTestDataset(os.path.join(images_path, 'val2014'), os.path.join(annotations_path, 'captions_val2014.json'), IMAGE_TRANSFORM),
        SimpleCaptionsTestDataset(os.path.join(images_path, 'test2014'), os.path.join(annotations_path, 'captions_test2014.json'), IMAGE_TRANSFORM))


def collate_fn_train(batch):
    images_list = []
    texts_list = []
    for image, text in batch:
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
