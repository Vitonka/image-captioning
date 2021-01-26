#!/opt/conda/bin/python3
import argparse
import json
import os
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import time

from dataset import get_coco_dataloaders
from train import train, validate
from models import (
    ShowAndTellWithPretrainedImageEmbeddings,
    ShowAndTellLSTM,
    ShowAttendTell
)

DATASETS_ROOT = '../datasets'
SUMMARY_WRITER_ROOT = '../training_logs'
MODEL_ROOT = '../models'
MODEL_FILE = 'model.pth'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config file')
    args = parser.parse_args()

    # Load config file
    with open(args.config) as f:
        config = json.load(f)

    data_config = config['data_config']
    assert data_config['data_mode'] == 'packed' or \
        data_config['data_mode'] == 'padded'

    # Set up device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    annotations_config_path = \
        os.path.join(DATASETS_ROOT, data_config['annotations_config_path'])
    with open(annotations_config_path) as f:
        annotations_config = json.load(f)
    annotations_path = \
        os.path.join(
            DATASETS_ROOT, 'coco', 'annotations',
            annotations_config['out_data_folder'])

    with open(os.path.join(annotations_path, 'w2i.json')) as f:
        w2i = json.load(f)

    with open(os.path.join(annotations_path, 'i2w.json')) as f:
        i2w = json.load(f)
        i2w = {int(i): i2w[i] for i in i2w}

    trainloader, valloader, _ = get_coco_dataloaders(data_config)

    model_config = config['model_config']
    if (model_config['model_name'] ==
            'ShowAndTellWithPretrainedImageEmbeddings'):
        model_type = 'rnn'
        model = ShowAndTellWithPretrainedImageEmbeddings(
            dict_size=len(w2i),
            embedding_dim=model_config['embedding_dim'],
            hidden_size=model_config['hidden_size'],
            data_mode=data_config['data_mode'],
            pad_idx=w2i['<PAD>'])
    elif model_config['model_name'] == 'ShowAndTellLSTM':
        model_type = 'lstm'
        model = ShowAndTellLSTM(
            dict_size=len(w2i),
            embedding_dim=model_config['embedding_dim'],
            hidden_size=model_config['hidden_size'],
            data_mode=data_config['data_mode'],
            pad_idx=w2i['<PAD>'])
    elif model_config['model_name'] == 'ShowAttendTell':
        model_type = 'lstm'
        model = ShowAttendTell(
            dict_size=len(w2i),
            embedding_dim=model_config['embedding_dim'],
            hidden_size=model_config['hidden_size'],
            attention_size=model_config['attention_size'],
            data_mode=data_config['data_mode'],
            pad_idx=w2i['<PAD>'])
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=w2i['<PAD>'])
    training_config = config['training_config']
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=training_config['optimizer_learning_rate'])

    MODEL_DIR = os.path.join(MODEL_ROOT, config['experiment_name'])
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

    writer = SummaryWriter(
        os.path.join(SUMMARY_WRITER_ROOT, config['experiment_name']))

    for epoch in range(training_config['epochs']):
        print('-----')
        print('Epoch: ', epoch)
        epoch_start_time = time.time()

        print('Train')
        loss = train(
            model, trainloader,
            criterion, optimizer,
            device, data_config['data_mode'])

        torch.save(model.state_dict(), MODEL_PATH)

        with torch.no_grad():
            print('Validate')
            scores = validate(
                model, model_type, valloader, device,
                w2i, i2w, data_config['data_mode'])

        epoch_duration = time.time() - epoch_start_time
        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('duration', epoch_duration, epoch)
        for score_name in scores:
            writer.add_scalar(score_name, scores[score_name], epoch)

        print('Loss: ', loss)
        print('Duration: ', epoch_duration)
        print('Scores: ', scores)
