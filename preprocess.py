#!/opt/conda/bin/python3
import h5py
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import torchvision
from torch import nn
import json
from shutil import copyfile
import argparse
import os
from utils.text_utils import create_dictionary_from_annotations, transform_text, clean_text

ROOT = 'data/datasets'
ANNOTATIONS_PATH = 'annotations/captions_{0}2014.json'
IMAGES_PATH = 'images/{0}2014'
KARPATHY_PATH = 'dataset_coco.json'
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def read_karpaty_splits(karpathy_split_path):
    with open(karpathy_split_path) as f:
        splits = json.load(f)

    train = set()
    val = set()
    test = set()
    for image in splits['images']:
        if image['split'] == 'val':
            val.add(image['cocoid'])
        elif image['split'] == 'test':
            test.add(image['cocoid'])
        else:
            train.add(image['cocoid'])

    return train, val, test


def extract_split_annotations(split, train_raw, val_raw):
    split_annotations = train_raw.copy()

    annotations = []
    for raw in [train_raw, val_raw]:
        for annotation in raw['annotations']:
            if annotation['image_id'] in split:
                annotations.append(annotation)

    images = []
    for raw in [train_raw, val_raw]:
        for image in raw['images']:
            if image['id'] in split:
                images.append(image)

    split_annotations['annotations'] = annotations
    split_annotations['images'] = images
    return split_annotations


def copy_image_files(dataset, out_dataset, split_name, split, train_raw, val_raw):
    os.mkdir(os.path.join(ROOT, out_dataset, IMAGES_PATH.format(split_name)))

    for raw_split, raw in zip(['train', 'val'], [train_raw, val_raw]):
        for image in raw['images']:
            if image['id'] in split:
                old_image_path = os.path.join(ROOT, dataset, IMAGES_PATH.format(raw_split), image['file_name'])
                new_image_path = os.path.join(ROOT, out_dataset, IMAGES_PATH.format(split_name), image['file_name'])
                copyfile(old_image_path, new_image_path)

def images_to_hdf5(dataset_folder, images_folder, images, out_filename):
    h5_file = h5py.File(os.path.join(dataset_folder, out_filename), 'w')
    data = h5_file.create_dataset(
        'images', shape=(len(images), 3, 224, 224), dtype=np.float32, fillvalue=0)
    images.sort(key=lambda x: x['id'])
    for i, image in tqdm(list(enumerate(images))):
        img = Image.open(os.path.join(images_folder, image['file_name'])).convert('RGB')
        img = IMAGE_TRANSFORM(img)
        data[i] = img.numpy()

    h5_file.close()

def images_to_embeddings(dataset_folder, h5_images, out_filename):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with h5py.File(os.path.join(dataset_folder, h5_images), 'r') as f:
        out_file = h5py.File(os.path.join(dataset_folder, out_filename), 'w')
        data = out_file.create_dataset('features', shape=(len(f['images']), 2048, 7, 7), dtype=np.float32, fillvalue=0)
        resnet = torchvision.models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        resnet = nn.Sequential(*modules)
        resnet.to(device)
        for i, image in tqdm(enumerate(f['images'])):
            image = torch.from_numpy(image)
            image = image.unsqueeze(0)
            image = image.to(device)
            temp = resnet(image)
            data[i] = temp[0].cpu().detach().numpy()

        out_file.close()

def transform_annotations(annotations, w2i):
    transformed_annotations = []
    for annotation in annotations:
        annotation['caption'] = transform_text(annotation['caption'], w2i)
        transformed_annotations.append(annotation)
    return transformed_annotations


def clean_annotations(annotations):
    cleaned_annotations = []
    for annotation in annotations:
        annotation['caption'] = ' '.join(clean_text(annotation['caption']))
        cleaned_annotations.append(annotation)
    return cleaned_annotations


def preprocess_coco(dataset, out_dataset):
    # Get Karpathy splits
    train, val, test = read_karpaty_splits(os.path.join(ROOT, dataset, KARPATHY_PATH))

    # Open raw train and validation annotations
    with open(os.path.join(ROOT, dataset, ANNOTATIONS_PATH.format('train'))) as f:
        train_raw = json.load(f)

    with open(os.path.join(ROOT, dataset, ANNOTATIONS_PATH.format('val'))) as f:
        val_raw = json.load(f)

    # Split annotations according to Karpathy splits
    train_annotations, val_annotations, test_annotations = \
        extract_split_annotations(train, train_raw, val_raw), \
        extract_split_annotations(val, train_raw, val_raw), \
        extract_split_annotations(test, train_raw, val_raw)

    # Create dictionary and preprocess train annotations
    w2i, i2w = create_dictionary_from_annotations(train_annotations)
    train_annotations['annotations'] = transform_annotations(train_annotations['annotations'], w2i)

    # Clean val and test annotations
    val_annotations['annotations'] = clean_annotations(val_annotations['annotations'])
    test_annotations['annotations'] = clean_annotations(test_annotations['annotations'])

    # Create dataset directories
    os.mkdir(os.path.join(ROOT, out_dataset))
    os.mkdir(os.path.join(ROOT, out_dataset, 'images'))
    os.mkdir(os.path.join(ROOT, out_dataset, 'annotations'))

    # Copy dataset data into dataset directories
    for split_name, split_annotations, split_ids in zip(['train', 'val', 'test'], [train_annotations, val_annotations, test_annotations], [train, val, test]):
        # Save annotations
        with open(os.path.join(ROOT, out_dataset, ANNOTATIONS_PATH.format(split_name)), 'w') as f:
            json.dump(split_annotations, f)

        # Copy images
        copy_image_files(dataset, out_dataset, split_name, split_ids, train_raw, val_raw)

        # Preprocess images and save to hdf5 file
        images_to_hdf5(os.path.join(ROOT, out_dataset), os.path.join(ROOT, out_dataset, IMAGES_PATH.format(split_name)), split_annotations, split_name + '.h5')

        # Preprocess image features with resnet and save to hdf5 file
        images_to_embeddings(os.path.join(ROOT, out_dataset), split_name + '.h5', split_name + '_features.h5')

    # Save dictionary
    with open(os.path.join(ROOT, out_dataset, 'w2i.json'), 'w') as f:
        json.dump(w2i, f)

    with open(os.path.join(ROOT, out_dataset, 'i2w.json'), 'w') as f:
        json.dump(i2w, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='input dataset name')
    parser.add_argument('--out_dataset', help='output dataset name')
    args = parser.parse_args()

    preprocess_coco(args.dataset, args.out_dataset)
