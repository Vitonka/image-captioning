#!/opt/conda/bin/python3
import argparse
import json
import os
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn

from utils.text_utils import create_dictionary
from dataset import get_train_dataset, get_train_dataloader, get_val_dataloader
from beam_search import simple_beam_search
from train import train, validate
from models import SimpleModel, SimpleModelWithEncoder

SUMMARY_WRITER_ROOT='data/runs'
MODEL_ROOT = 'data/models'
MODEL_FILE = 'model.pth'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config file')
    args = parser.parse_args()

    # Load config file
    with open(args.config) as f:
        config = json.load(f)

    # Set up device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = get_train_dataset(config['dataset_name'])

    w2i, i2w = create_dictionary(train_dataset)

    trainloader = get_train_dataloader(config['dataset_name'], config['batch_size'], w2i)
    valloader = get_val_dataloader(config['dataset_name'], batch_size=1)

    model = SimpleModelWithEncoder(dict_size=len(w2i), embedding_dim=config['embedding_dim'], hidden_size=config['hidden_size'])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad ], lr=config['optimizer_learning_rate'])

    MODEL_DIR = os.path.join(MODEL_ROOT, config['model_name'])
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

    writer = SummaryWriter(os.path.join(SUMMARY_WRITER_ROOT, config['model_name']))

    for epoch in range(config['epochs']):
        print('-----')
        print('Epoch: ', epoch)

        print('Train')
        loss = train(model, trainloader, criterion, optimizer, device)

        torch.save(model.state_dict(), MODEL_PATH)

        with torch.no_grad():
            print('Validate')
            scores = validate(model, valloader, device, w2i, i2w)

        writer.add_scalar('loss', loss, epoch)
        for score_name in scores:
            writer.add_scalar(score_name, scores[score_name], epoch)

        print('Loss: ', loss)
        print('Scores: ', scores)
