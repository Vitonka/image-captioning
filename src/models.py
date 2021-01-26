from torch import nn
import torch
import torchvision


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


class ShowAttendTell(nn.Module):
    def __init__(
            self, dict_size,
            embedding_dim, hidden_size, attention_size,
            data_mode, pad_idx=None):
        super(ShowAttendTell, self).__init__()
        self._data_mode = data_mode
        self.dict_size = dict_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.linear_h = nn.Linear(in_features=2048, out_features=hidden_size)
        self.linear_c = nn.Linear(in_features=2048, out_features=hidden_size)
        self.linear_out = \
            nn.Linear(in_features=hidden_size, out_features=dict_size)

        self.encoded_att = \
            nn.Linear(in_features=2048, out_features=attention_size)
        self.hidden_att = \
            nn.Linear(in_features=hidden_size, out_features=attention_size)
        self.full_att = nn.Linear(in_features=attention_size, out_features=1)

        self.embedding = \
            nn.Embedding(
                num_embeddings=dict_size,
                embedding_dim=embedding_dim,
                padding_idx=pad_idx)

        self.lstm_cell = \
            nn.LSTMCell(
                input_size=embedding_dim + 2048,
                hidden_size=hidden_size)

    def encoder(self, image):
        encoded = self.resnet(image)
        assert encoded.shape[1:] == (2048, 7, 7), \
            'Shapes mismatch, actual shape is ' + str(encoded.shape[1:])

        encoded = encoded.reshape(encoded.shape[0], encoded.shape[1], -1)
        assert encoded.shape[1:] == (2048, 49)

        return encoded

    def attention(self, encoded, hidden):
        assert encoded.shape[1:] == (2048, 49)
        assert hidden.shape[1:] == (self.hidden_size,)

        encoded_att = self.encoded_att(torch.transpose(encoded, 1, 2))
        hidden_att = self.hidden_att(hidden)
        combined_att = self.relu(encoded_att + hidden_att.unsqueeze(1))
        att = self.full_att(combined_att).squeeze(-1)
        att = self.softmax(att)
        assert att.shape[1:] == (49,)
        att = att.unsqueeze(1)

        attended = torch.sum(att * encoded, dim=2)
        assert attended.shape[1:] == (2048,), \
            'Shapes mismatch, actual shape is ' + str(attended.shape[1:])

        return attended

    def forward(self, image, input_captions):
        image_vector = self.encoder(image)
        image_mean = torch.mean(image_vector, dim=2)
        assert image_mean.shape[1:] == (2048,)

        h = self.linear_h(image_mean)
        c = self.linear_c(image_mean)

        attended = self.attention(image_vector, h)
        embeddings = self.embedding(input_captions)
        assert embeddings.shape[2] == self.embedding_dim
        seq_len = embeddings.shape[1]
        ans = []

        for i in range(seq_len):
            cur_embeddings = embeddings[:, i, :]
            assert cur_embeddings.shape[1] == self.embedding_dim
            assert attended.shape[1] == 2048
            assert cur_embeddings.shape[0] == attended.shape[0]

            x = torch.cat((cur_embeddings, attended), dim=1)
            h, c = self.lstm_cell(x, (h, c))
            assert h.shape[1] == self.hidden_size
            assert c.shape[1] == self.hidden_size

            attended = self.attention(image_vector, h)
            assert attended.shape[1:] == (2048,)
            ans.append(self.linear_out(h))

        ans = torch.stack(ans, dim=1)
        assert ans.shape[1:] == (seq_len, self.dict_size), \
            'Shape mismatch, actual shape is ' + str(ans.shape[1:])
        return ans, (h, c)
