from torch import nn
from torchvision import models
import torchvision
import torch

class SimpleModel(nn.Module):
    def __init__(self, dict_size, embedding_dim, hidden_size, *args, **kwargs):
        super(SimpleModel, self).__init__(*args, **kwargs)
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(in_features=13520, out_features=hidden_size)
        self.encoder_layers = [
            self.conv1, self.pooling, self.relu,
            self.conv2, self.pooling, self.relu,
            self.conv3, self.pooling, self.relu]

        self.embedding = nn.Embedding(num_embeddings=dict_size, embedding_dim=embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=dict_size)

    def encoder(self, image):
        for layer in self.encoder_layers:
            image = layer(image)
        return self.linear1(image.view(-1, 13520)).view(-1, self.hidden_size)

    def decoder(self, image_vector, input_captions):
        embeddings = nn.utils.rnn.PackedSequence(
            self.embedding(input_captions.data),
            input_captions.batch_sizes)
        decoded, hiddens = self.rnn(embeddings, image_vector)
        probs = self.linear2(decoded.data)
        return nn.utils.rnn.PackedSequence(probs, decoded.batch_sizes), hiddens

    def forward(self, image, input_captions):
        image_vector = self.encoder(image)
        image_vector = image_vector.unsqueeze(0)
        return self.decoder(image_vector, input_captions)


class SimpleModelWithEncoder(nn.Module):
    def __init__(self, dict_size, embedding_dim, hidden_size, *args, **kwargs):
        super(SimpleModelWithEncoder, self).__init__(*args, **kwargs)

        resnet = torchvision.models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False
        # TODO: try to use mean instead of flatten all the features
        self.linear1 = nn.Linear(in_features=100352, out_features=hidden_size)

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=dict_size, embedding_dim=embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=dict_size)

    def encoder(self, image):
        image = self.resnet(image)
        return self.linear1(image.view(image.shape[0], -1)).view(-1, self.hidden_size)

    def decoder(self, image_vector, input_captions):
        embeddings = nn.utils.rnn.PackedSequence(
            self.embedding(input_captions.data),
            input_captions.batch_sizes)
        decoded, hiddens = self.rnn(embeddings, image_vector)
        probs = self.linear2(decoded.data)
        return nn.utils.rnn.PackedSequence(probs, decoded.batch_sizes), hiddens

    def forward(self, image, input_captions):
        image_vector = self.encoder(image)
        image_vector = image_vector.unsqueeze(0)
        return self.decoder(image_vector, input_captions)

class SimpleModelWithPreptrainedImageEmbeddings(nn.Module):
    def __init__(self, dict_size, embedding_dim, hidden_size, *args, **kwargs):
        super(SimpleModelWithPreptrainedImageEmbeddings, self).__init__(*args, **kwargs)
        self.linear1 = nn.Linear(in_features=2048, out_features=embedding_dim)

        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=dict_size, embedding_dim=embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, nonlinearity='relu')
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=dict_size)

    def encoder(self, image):
        image = image.mean(dim=(2, 3))
        return self.linear1(image).view(-1, self.embedding_dim)

    def decoder(self, hiddens, input_captions):
        embeddings = nn.utils.rnn.PackedSequence(
            self.embedding(input_captions.data),
            input_captions.batch_sizes)
        decoded, hiddens = self.rnn(embeddings, hiddens)
        probs = self.linear2(decoded.data)
        return nn.utils.rnn.PackedSequence(probs, decoded.batch_sizes), hiddens

    def forward(self, image, input_captions):
        image_vector = self.encoder(image)

        image_vectors = [v.unsqueeze(0) for v in image_vector]
        image_embeddings = torch.nn.utils.rnn.pack_sequence(image_vectors)
        _, hiddens = self.rnn(image_embeddings)

        return self.decoder(hiddens, input_captions)


class ResNetLSTM(nn.Module):
    def __init__(self, dict_size, embedding_dim, hidden_size, *args, **kwargs):
        super(ResNetLSTM, self).__init__(*args, **kwargs)
        # TODO: try to use mean instead of flatten all the features
        self.linear_h = nn.Linear(in_features=100352, out_features=hidden_size)
        self.linear_c = nn.Linear(in_features=100352, out_features=hidden_size)

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=dict_size, embedding_dim=embedding_dim)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=dict_size)

    def encoder(self, image):
        return self.linear1(image.view(image.shape[0], -1)).view(-1, self.hidden_size)

    def decoder(self, h0, c0, input_captions):
        embeddings = nn.utils.rnn.PackedSequence(
            self.embedding(input_captions.data),
            input_captions.batch_sizes)
        decoded, (hiddens, cells) = self.rnn(embeddings, (h0, c0))
        probs = self.linear2(decoded.data)
        return nn.utils.rnn.PackedSequence(probs, decoded.batch_sizes), hiddens, cells

    def get_h(self, image):
        h = self.linear_h(image.view(image.shape[0], -1)).view(-1, self.hidden_size)
        return h

    def get_c(self, image):
        c = self.linear_c(image.view(image.shape[0], -1)).view(-1, self.hidden_size)
        return c

    def forward(self, image, input_captions):
        h0 = self.get_h(image).unsqueeze(0)
        c0 = self.get_c(image).unsqueeze(0)
        return self.decoder(h0, c0, input_captions)
