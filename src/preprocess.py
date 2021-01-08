#!/opt/conda/bin/python3
import h5py
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import torchvision
from torch import nn
import json
import argparse
import os
from utils.text_utils import (
    create_dictionary_from_annotations, transform_text, clean_text)
from tqdm import tqdm

# Default data paths
ROOT = '../datasets'
COCO_RAW = 'coco/raw'
ANNOTATIONS_RAW_PATH = 'annotations/captions_{0}2014_raw.json'
ANNOTATIONS_PATH = 'annotations/captions_{0}2014.json'
CAPTIONS_PATH = 'captions_{0}2014.json'
IMAGES_RAW_PATH = 'images/{0}2014'
KARPATHY_PATH = 'dataset_coco.json'


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
                annotations.append(annotation.copy())

    images = []
    for raw in [train_raw, val_raw]:
        for image in raw['images']:
            if image['id'] in split:
                images.append(image)

    images.sort(key=lambda x: x['id'])
    id_to_idx = {}
    for i, image in enumerate(images):
        id_to_idx[image['id']] = i
    for annotation in annotations:
        annotation['idx'] = id_to_idx[annotation['image_id']]
    split_annotations['annotations'] = annotations
    split_annotations['images'] = images
    return split_annotations


def split_annotations():
    # Get Karpathy splits
    train, val, test = \
        read_karpaty_splits(os.path.join(ROOT, COCO_RAW, KARPATHY_PATH))

    # Open raw train and validation annotations
    with open(os.path.join(
            ROOT, COCO_RAW, ANNOTATIONS_RAW_PATH.format('train'))) as f:
        train_raw = json.load(f)

    with open(os.path.join(
            ROOT, COCO_RAW, ANNOTATIONS_RAW_PATH.format('val'))) as f:
        val_raw = json.load(f)

    # Split annotations according to Karpathy splits
    train_annotations, val_annotations, test_annotations = \
        extract_split_annotations(train, train_raw, val_raw), \
        extract_split_annotations(val, train_raw, val_raw), \
        extract_split_annotations(test, train_raw, val_raw)

    for split, annotations in zip(
            ['train', 'val', 'test'],
            [train_annotations, val_annotations, test_annotations]):
        with open(os.path.join(
                ROOT, COCO_RAW, ANNOTATIONS_PATH.format(split)), 'w') as f:
            json.dump(annotations, f)


def preprocess_annotations(config):
    with open(os.path.join(
            ROOT, COCO_RAW, ANNOTATIONS_PATH.format('train'))) as f:
        train_annotations = json.load(f)
    with open(os.path.join(
            ROOT, COCO_RAW, ANNOTATIONS_PATH.format('val'))) as f:
        val_annotations = json.load(f)
    with open(os.path.join(
            ROOT, COCO_RAW, ANNOTATIONS_PATH.format('test'))) as f:
        test_annotations = json.load(f)

    # Create dictionary and preprocess train annotations
    w2i, i2w = create_dictionary_from_annotations(
        train_annotations, min_word_freq=config['min_word_freq'])
    train_annotations['annotations'] = \
        transform_annotations(train_annotations['annotations'], w2i)

    # Clean val and test annotations
    val_annotations['annotations'] = \
        clean_annotations(val_annotations['annotations'])
    test_annotations['annotations'] = \
        clean_annotations(test_annotations['annotations'])

    os.makedirs(os.path.join(ROOT, config['out_data_folder']), exist_ok=True)

    for split, annotations in zip(
            ['train', 'val', 'test'],
            [train_annotations, val_annotations, test_annotations]):
        with open(os.path.join(
                config['out_data_folder'],
                CAPTIONS_PATH.format(split)), 'w') as f:
            json.dump(annotations, f)

    with open(os.path.join(config['out_data_folder'], 'w2i.json'), 'w') as f:
        json.dump(w2i, f)

    with open(os.path.join(config['out_data_folder'], 'i2w.json'), 'w') as f:
        json.dump(i2w, f)


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


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_image_path(image):
    if 'train' in image['file_name']:
        split = 'train'
    elif 'val' in image['file_name']:
        split = 'val'
    elif 'test' in image['file_name']:
        split = 'test'
    else:
        assert False, 'Unknown split for image: ' + str(image)
    return os.path.join(
        ROOT, COCO_RAW, IMAGES_RAW_PATH.format(split), image['file_name'])


def read_and_convert_image(image):
    image = Image.open(get_image_path(image)).convert('RGB')
    image = IMAGE_TRANSFORM(image)
    return image.numpy()


def apply_model_to_image(model, image, device):
    image = torch.tensor(read_and_convert_image(image))
    image = image.unsqueeze(0)
    image = image.to(device)
    image = model(image)
    return image[0].squeeze(-1).squeeze(-1).cpu().detach().numpy()


def process_images_to_npy(images, data_getter, out_path):
    all_data = []
    for image in tqdm(images):
        all_data.append(data_getter(image))
    np.save(out_path, np.array(all_data))


def process_images_to_h5py(images, data_getter, out_path):
    h5_file = h5py.File(out_path + '.h5py', 'w')
    if config['data'] == 'images':
        shape = (len(images), 3, 224, 224)
    elif config['data'] == 'features':
        shape = (len(images), 2048)
    data = h5_file.create_dataset(
        'data', shape=shape, dtype=np.float32, fillvalue=0)
    for i, image in tqdm(list(enumerate(images))):
        data[i] = data_getter(image)
    h5_file.close()


def preprocess_images(config):
    if config['data'] == 'images':
        data_getter = read_and_convert_image
    elif config['data'] == 'features':
        if config['model'] == 'resnet':
            resnet = torchvision.models.resnet101(pretrained=True)
            resnet.eval()
            modules = list(resnet.children())[:-1]
            model = nn.Sequential(*modules)
        else:
            assert False, 'Unknown model'

        for param in model.parameters():
            param.requires_grad = False

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        def data_getter(image):
            return apply_model_to_image(model, image, device)
    else:
        assert False, 'Unknown data'

    with open(os.path.join(
            ROOT, COCO_RAW, ANNOTATIONS_PATH.format('train'))) as f:
        train_annotations = json.load(f)
    with open(os.path.join(
            ROOT, COCO_RAW, ANNOTATIONS_PATH.format('val'))) as f:
        val_annotations = json.load(f)
    with open(os.path.join(
            ROOT, COCO_RAW, ANNOTATIONS_PATH.format('test'))) as f:
        test_annotations = json.load(f)

    for split, annotations in zip(
            ['train', 'val', 'test'],
            [train_annotations, val_annotations, test_annotations]):
        images = annotations['images']
        images.sort(key=lambda x: x['id'])

        id_to_idx, idx_to_id = {}, {}
        for i, image in zip(range(len(images)), images):
            id_to_idx[image['id']] = i
            idx_to_id[i] = image['id']
        with open(os.path.join(
                config['out_data_folder'], 'id_to_idx.json'), 'w') as f:
            json.dump(id_to_idx, f)
        with open(os.path.join(
                config['out_data_folder'], 'idx_to_id.json'), 'w') as f:
            json.dump(idx_to_id, f)

        out_path = os.path.join(config['out_data_folder'], split)
        if config['data_type'] == 'npy':
            process_images_to_npy(images, data_getter, out_path)
        elif config['data_type'] == 'h5py':
            process_images_to_h5py(images, data_getter, out_path)
        else:
            assert False, 'Unknown data type'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--split_annotations',
        action='store_true',
        help='split annotations according to a karpathy split')
    parser.add_argument(
        '--annotations_processing_config',
        help='config to preprocess annotations for training')
    parser.add_argument(
        '--images_processing_config',
        help='config to preprocess images for training')
    args = parser.parse_args()

    if args.split_annotations:
        split_annotations()

    if args.annotations_processing_config:
        with open(args.annotations_processing_config) as f:
            config = json.load(f)
            config['out_data_folder'] = os.path.join(
                ROOT, 'coco', 'annotations', config['out_data_folder'])
            preprocess_annotations(config)

    if args.images_processing_config:
        with open(args.images_processing_config) as f:
            config = json.load(f)
            config['out_data_folder'] = os.path.join(
                ROOT, 'coco', 'images', config['out_data_folder'])
            preprocess_images(config)
