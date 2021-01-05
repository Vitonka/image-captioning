#!/opt/conda/bin/python3
import argparse
import json
import os
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn

from utils.text_utils import create_dictionary
from dataset import get_coco_dataloaders
from beam_search import beam_search
from train import train, validate
from models import SimpleModel, SimpleModelWithEncoder, SimpleModelWithPreptrainedImageEmbeddings

DATASETS_ROOT = 'data/datasets'
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

    assert config['data_mode'] == 'packed' or config['data_mode'] == 'padded'

    # Set up device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = os.path.join(DATASETS_ROOT, config['dataset_name'])

    with open(os.path.join(dataset_path, 'w2i.json')) as f:
        w2i = json.load(f)

    with open(os.path.join(dataset_path, 'i2w.json')) as f:
        i2w = json.load(f)
        i2w = {int(i): i2w[i] for i in i2w}
    # TODO(vitonka): move to a dict creation
    w2i['<PAD>'] = len(w2i)
    i2w[len(w2i) - 1] = '<PAD>'

    trainloader, valloader, _ = get_coco_dataloaders(dataset_path, config['batch_size'], 1, 1, config['data_mode'])

#    model = SimpleModelWithEncoder(dict_size=len(w2i), embedding_dim=config['embedding_dim'], hidden_size=config['hidden_size'])
    model = SimpleModelWithPreptrainedImageEmbeddings(
        dict_size=len(w2i),
        embedding_dim=config['embedding_dim'],
        hidden_size=config['hidden_size'],
        data_mode=config['data_mode'],
        pad_idx=w2i['<PAD>'])
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=w2i['<PAD>'])
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad ], lr=config['optimizer_learning_rate'])

    MODEL_DIR = os.path.join(MODEL_ROOT, config['model_name'])
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

    writer = SummaryWriter(os.path.join(SUMMARY_WRITER_ROOT, config['model_name']))

    for epoch in range(config['epochs']):
        print('-----')
        print('Epoch: ', epoch)

        print('Train')
        loss = train(model, trainloader, criterion, optimizer, device, config['data_mode'])

        torch.save(model.state_dict(), MODEL_PATH)

        with torch.no_grad():
            print('Validate')
            scores = validate(model, valloader, device, w2i, i2w)

        writer.add_scalar('loss', loss, epoch)
        for score_name in scores:
            writer.add_scalar(score_name, scores[score_name], epoch)

        print('Loss: ', loss)
        print('Scores: ', scores)
