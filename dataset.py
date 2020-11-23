import torch
import torchvision
import os
from torchvision import transforms
import contextlib

from utils.text_utils import clean_text, transform_text


IMAGES_PATH = 'images/{0}2014'
ANNOTATIONS_PATH = 'annotations/captions_{0}2014.json'
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
DATASET_ROOT = 'data/datasets'

# TEMPORARY UGLY COSTYL
w2i = {}

def get_train_dataset(dataset_name):
    with contextlib.redirect_stdout(None):
        train_dataset = torchvision.datasets.CocoCaptions(
            root = os.path.join(DATASET_ROOT, dataset_name, IMAGES_PATH.format('train')),
            annFile = os.path.join(DATASET_ROOT, dataset_name, ANNOTATIONS_PATH.format('train')))
    return train_dataset

def collate_fn_train(batch):
    images_list = []
    texts_list = []
    for image, texts in batch:
        image = IMAGE_TRANSFORM(image)
        images_list += [image] * len(texts)

        for text in texts:
            texts_list.append(torch.tensor(transform_text(text, w2i)))

    images_list, texts_list = \
        list(zip(*sorted(zip(images_list, texts_list), key=lambda x: x[1].shape[0], reverse=True)))

    inputs = [text[:-1] for text in texts_list]
    outputs = [text[1:] for text in texts_list]

    packed_inputs = torch.nn.utils.rnn.pack_sequence(inputs, enforce_sorted=True)
    packed_outputs = torch.nn.utils.rnn.pack_sequence(outputs, enforce_sorted=True)
    return torch.stack(images_list), packed_inputs, packed_outputs

def get_train_dataloader(dataset_name, batch_size, w2i_in):
    global w2i
    w2i = w2i_in
    train_dataset = get_train_dataset(dataset_name)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_train)
    return trainloader

def collate_fn_validate(batch):
    images_list = []
    texts_list = []
    for image, texts in batch:
        images_list.append(IMAGE_TRANSFORM(image))
        texts = list(map(lambda text: ' '.join(clean_text(text)), texts))
        texts_list.append(texts)
    return torch.stack(images_list), texts_list

def get_val_dataloader(dataset_name, batch_size):
    val_dataset = get_train_dataset(dataset_name)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_validate)
    return valloader
