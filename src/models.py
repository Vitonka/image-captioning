from torch import nn
import torch


class ShowAndTellWithPretrainedImageEmbeddings(nn.Module):
    def __init__(
            self, dict_size,
            embedding_dim, hidden_size,
            data_mode, pad_idx=None):
        super(ShowAndTellWithPretrainedImageEmbeddings, self).__init__()

        assert data_mode == 'packed' or data_mode == 'padded'
        self._data_mode = data_mode
        if self._data_mode == 'padded':
            assert pad_idx is not None

        self.linear1 = nn.Linear(in_features=2048, out_features=embedding_dim)

        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.embedding = \
            nn.Embedding(
                num_embeddings=dict_size,
                embedding_dim=embedding_dim,
                padding_idx=pad_idx)
        self.rnn = \
            nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                batch_first=True)
        self.linear2 = \
            nn.Linear(in_features=hidden_size, out_features=dict_size)

    def encoder(self, image):
        return self.linear1(image)

    def decoder(self, hiddens, input_captions):
        if self._data_mode == 'packed':
            embeddings = nn.utils.rnn.PackedSequence(
                self.embedding(input_captions.data),
                input_captions.batch_sizes)
        elif self._data_mode == 'padded':
            embeddings = self.embedding(input_captions)
        decoded, hiddens = self.rnn(embeddings, hiddens)
        if self._data_mode == 'packed':
            probs = self.linear2(decoded.data)
            return nn.utils.rnn.PackedSequence(probs, decoded.batch_sizes), \
                hiddens
        elif self._data_mode == 'padded':
            probs = self.linear2(decoded)
            return probs, hiddens

    def forward(self, image, input_captions):
        image_vector = self.encoder(image)

        if self._data_mode == 'packed':
            image_vectors = [v.unsqueeze(0) for v in image_vector]
            image_embeddings = torch.nn.utils.rnn.pack_sequence(image_vectors)
            _, hiddens = self.rnn(image_embeddings)
        elif self._data_mode == 'padded':
            _, hiddens = self.rnn(image_vector.unsqueeze(1))

        return self.decoder(hiddens, input_captions)


class ShowAndTellLSTM(nn.Module):
    def __init__(
            self, dict_size,
            embedding_dim, hidden_size,
            data_mode, pad_idx=None):
        super(ShowAndTellLSTM, self).__init__()

        assert data_mode == 'packed' or data_mode == 'padded'
        self._data_mode = data_mode
        if self._data_mode == 'padded':
            assert pad_idx is not None

        self.linear1 = nn.Linear(in_features=2048, out_features=embedding_dim)

        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.embedding = \
            nn.Embedding(
                num_embeddings=dict_size,
                embedding_dim=embedding_dim,
                padding_idx=pad_idx)
        self.lstm = \
            nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                batch_first=True)
        self.linear2 = \
            nn.Linear(in_features=hidden_size, out_features=dict_size)

    def encoder(self, image):
        return self.linear1(image)

    def decoder(self, history, input_captions):
        if self._data_mode == 'packed':
            embeddings = nn.utils.rnn.PackedSequence(
                self.embedding(input_captions.data),
                input_captions.batch_sizes)
        elif self._data_mode == 'padded':
            embeddings = self.embedding(input_captions)
        decoded, history = self.lstm(embeddings, history)
        if self._data_mode == 'packed':
            probs = self.linear2(decoded.data)
            return nn.utils.rnn.PackedSequence(probs, decoded.batch_sizes), \
                history
        elif self._data_mode == 'padded':
            probs = self.linear2(decoded)
            return probs, history

    def forward(self, image, input_captions):
        image_vector = self.encoder(image)

        if self._data_mode == 'packed':
            image_vectors = [v.unsqueeze(0) for v in image_vector]
            image_embeddings = torch.nn.utils.rnn.pack_sequence(image_vectors)
            _, history = self.lstm(image_embeddings)
        elif self._data_mode == 'padded':
            _, history = self.lstm(image_vector.unsqueeze(1))

        return self.decoder(history, input_captions)
