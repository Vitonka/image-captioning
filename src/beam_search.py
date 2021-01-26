import torch
from utils.text_utils import START, END


def beam_search(
        model, model_type, image,
        w2i, i2w,
        device, max_length=15, beam_size=1, data_mode='packed'):
    if model_type == 'rnn':
        beam_func = beam_search_rnn
    elif model_type == 'lstm':
        beam_func = beam_search_lstm
    elif model_type == 'attention':
        beam_func = beam_search_attention
    return beam_func(
        model, image,
        w2i, i2w,
        device, max_length, beam_size, data_mode)


def beam_search_rnn(
        model, image,
        w2i, i2w,
        device, max_length=15, beam_size=1, data_mode='packed'):
    # Run only in known mode
    assert data_mode == 'packed' or data_mode == 'padded'

    # Move model and image to the same device
    model = model.to(device)
    image = image.to(device)

    # Initialize list with final generated captions
    final_hyps = []
    final_probs = []

    # Calculate initial hidden state
    image_embed = model.encoder(image).unsqueeze(0)
    if data_mode == 'packed':
        image_embed = torch.nn.utils.rnn.pack_sequence(image_embed)
    elif data_mode == 'padded':
        pass
    _, h0 = model.rnn(image_embed)
    h0 = h0.squeeze(0)

    # Initialize start hypothesis, their probabilities and hidden states
    cur_hyps = torch.tensor([[w2i[START]]]).to(device)
    cur_probs = torch.tensor([1.]).to(device)
    cur_hiddens = h0

    for i in range(max_length):
        if len(cur_hyps) == 0:
            break

        # Get last words and prepare them as an input
        last_words = [hyp[-1].unsqueeze(0) for hyp in cur_hyps]
        if data_mode == 'packed':
            cur_inputs = \
                torch.nn.utils.rnn.pack_sequence(
                    last_words, enforce_sorted=True)
        elif data_mode == 'padded':
            cur_inputs = torch.stack(last_words)
        cur_inputs = cur_inputs.to(device)

        # Use model decoder to get words probabilities and hidden states
        cur_hiddens = cur_hiddens.unsqueeze(0)
        words_probs, hiddens = model.decoder(cur_hiddens, cur_inputs)

        if data_mode == 'packed':
            words_probs, _ = \
                torch.nn.utils.rnn.pad_packed_sequence(
                    words_probs, batch_first=True)
        elif data_mode == 'padded':
            pass
        words_probs, hiddens = words_probs.squeeze(1), hiddens.squeeze(0)
        words_probs = torch.nn.Softmax(dim=1)(words_probs)

        # Determine the most probable words for each hypothesis
        max_words = words_probs.argsort(dim=1)[:, -beam_size:]

        # Generate new hypothesis and count their probabilities
        new_hyps, new_probs, new_hiddens = [], [], []
        for cur_prob, cur_hyp, words_prob, hidden, words in \
                zip(cur_probs, cur_hyps, words_probs, hiddens, max_words):
            # Calculate new hypothesis probs
            words_prob = words_prob[words] * cur_prob
            new_probs.append(words_prob)

            # Generate hypothesis with a new word
            for word in words:
                new_hyp = torch.cat((cur_hyp, word.unsqueeze(0)), dim=0)
                new_hyps.append(new_hyp)
                new_hiddens.append(hidden)

        # Get the best of the best
        new_hyps, new_hiddens, new_probs = \
            torch.stack(new_hyps), \
            torch.stack(new_hiddens), \
            torch.cat(new_probs)
        best_hyps = new_probs.argsort()[-beam_size:]
        cur_hyps, cur_probs, cur_hiddens = \
            new_hyps[best_hyps], new_probs[best_hyps], new_hiddens[best_hyps]

        # Working with END
        end_indices = torch.flatten((cur_hyps[:, -1] == w2i[END]).nonzero())
        end_indices = end_indices.flip(dims=(0, ))
        for index in end_indices:
            final_hyps.append(cur_hyps[index])
            final_probs.append(cur_probs[index])
        filter_index = cur_hyps[:, -1] != w2i[END]
        cur_hyps, cur_probs, cur_hiddens = \
            cur_hyps[filter_index], \
            cur_probs[filter_index], \
            cur_hiddens[filter_index]

    final_probs = torch.tensor(final_probs)
    final_sort = final_probs.argsort(descending=True)
    final_hyps = [final_hyps[i] for i in final_sort]
    final_probs = final_probs[final_sort]

    return final_hyps[:beam_size]


def beam_search_lstm(
        model, image,
        w2i, i2w,
        device, max_length=15, beam_size=1, data_mode='packed'):
    # Run only in known mode
    assert data_mode == 'packed' or data_mode == 'padded'

    # Move model and image to the same device
    model = model.to(device)
    image = image.to(device)

    # Initialize list with final generated captions
    final_hyps = []
    final_probs = []

    # Calculate initial hidden state
    image_embed = model.encoder(image).unsqueeze(0)
    if data_mode == 'packed':
        image_embed = torch.nn.utils.rnn.pack_sequence(image_embed)
    elif data_mode == 'padded':
        pass
    _, (h0, c0) = model.lstm(image_embed)
    h0 = h0.squeeze(0)
    c0 = c0.squeeze(0)

    # Initialize start hypothesis, their probabilities and hidden states
    cur_hyps = torch.tensor([[w2i[START]]]).to(device)
    cur_probs = torch.tensor([1.]).to(device)
    cur_hiddens = h0
    cur_cells = c0

    for i in range(max_length):
        if len(cur_hyps) == 0:
            break

        # Get last words and prepare them as an input
        last_words = [hyp[-1].unsqueeze(0) for hyp in cur_hyps]
        if data_mode == 'packed':
            cur_inputs = \
                torch.nn.utils.rnn.pack_sequence(
                    last_words, enforce_sorted=True)
        elif data_mode == 'padded':
            cur_inputs = torch.stack(last_words)
        cur_inputs = cur_inputs.to(device)

        # Use model decoder to get words probabilities and hidden states
        cur_hiddens = cur_hiddens.unsqueeze(0)
        cur_cells = cur_cells.unsqueeze(0)
        words_probs, (hiddens, cells) = \
            model.decoder((cur_hiddens, cur_cells), cur_inputs)

        if data_mode == 'packed':
            words_probs, _ = \
                torch.nn.utils.rnn.pad_packed_sequence(
                    words_probs, batch_first=True)
        elif data_mode == 'padded':
            pass
        words_probs, hiddens, cells = \
            words_probs.squeeze(1), hiddens.squeeze(0), cells.squeeze(0)
        words_probs = torch.nn.Softmax(dim=1)(words_probs)

        # Determine the most probable words for each hypothesis
        max_words = words_probs.argsort(dim=1)[:, -beam_size:]

        # Generate new hypothesis and count their probabilities
        new_hyps, new_probs, new_hiddens, new_cells = [], [], [], []
        for cur_prob, cur_hyp, words_prob, hidden, cell, words in \
                zip(cur_probs, cur_hyps, words_probs,
                    hiddens, cells, max_words):
            # Calculate new hypothesis probs
            words_prob = words_prob[words] * cur_prob
            new_probs.append(words_prob)

            # Generate hypothesis with a new word
            for word in words:
                new_hyp = torch.cat((cur_hyp, word.unsqueeze(0)), dim=0)
                new_hyps.append(new_hyp)
                new_hiddens.append(hidden)
                new_cells.append(cell)

        # Get the best of the best
        new_hyps, new_hiddens, new_cells, new_probs = \
            torch.stack(new_hyps), \
            torch.stack(new_hiddens), \
            torch.stack(new_cells), \
            torch.cat(new_probs)
        best_hyps = new_probs.argsort()[-beam_size:]
        cur_hyps, cur_probs, cur_hiddens, cur_cells = \
            new_hyps[best_hyps], new_probs[best_hyps],\
            new_hiddens[best_hyps], new_cells[best_hyps]

        # Working with END
        end_indices = torch.flatten((cur_hyps[:, -1] == w2i[END]).nonzero())
        end_indices = end_indices.flip(dims=(0, ))
        for index in end_indices:
            final_hyps.append(cur_hyps[index])
            final_probs.append(cur_probs[index])
        filter_index = cur_hyps[:, -1] != w2i[END]
        cur_hyps, cur_probs, cur_hiddens, cur_cells = \
            cur_hyps[filter_index], \
            cur_probs[filter_index], \
            cur_hiddens[filter_index], \
            cur_cells[filter_index]

    final_probs = torch.tensor(final_probs)
    final_sort = final_probs.argsort(descending=True)
    final_hyps = [final_hyps[i] for i in final_sort]
    final_probs = final_probs[final_sort]

    return final_hyps[:beam_size]


def beam_search_attention(
        model, image,
        w2i, i2w,
        device, max_length=15, beam_size=1, data_mode='packed'):
    # Run only in known mode
    assert data_mode == 'padded'

    # Move model and image to the same device
    model = model.to(device)
    image = image.to(device)

    # Initialize list with final generated captions
    final_hyps = []
    final_probs = []

    # Calculate initial hidden state
    image_vector = model.encoder(image)
    image_mean = torch.mean(image_vector, dim=2)
    assert image_mean.shape[1:] == (2048,)

    h0 = model.linear_h(image_mean)
    c0 = model.linear_c(image_mean)
    attended = model.attention(image_vector, h0)

    # Initialize start hypothesis, their probabilities and hidden states
    cur_hyps = torch.tensor([[w2i[START]]]).to(device)
    cur_probs = torch.tensor([1.]).to(device)
    cur_hiddens = h0
    cur_cells = c0
    cur_attendeds = attended

    for i in range(max_length):
        if len(cur_hyps) == 0:
            break

        # Get last words and prepare them as an input
        last_words = [hyp[-1].unsqueeze(0) for hyp in cur_hyps]
        cur_inputs = torch.stack(last_words)
        cur_inputs = cur_inputs.to(device)

        # Use model decoder to get words probabilities and hidden states
        words_probs, (hiddens, cells, attendeds) = \
            model.decoder(
                (cur_hiddens, cur_cells, cur_attendeds),
                cur_inputs, image_vector)

        words_probs = words_probs.squeeze(1)
        words_probs = torch.nn.Softmax(dim=1)(words_probs)

        # Determine the most probable words for each hypothesis
        max_words = words_probs.argsort(dim=1)[:, -beam_size:]

        # Generate new hypothesis and count their probabilities
        new_hyps, new_probs, new_hiddens, new_cells, new_attendeds = \
            [], [], [], [], []
        for cur_prob, cur_hyp, words_prob, hidden, cell, att, words in \
                zip(cur_probs, cur_hyps, words_probs,
                    hiddens, cells, attendeds, max_words):
            # Calculate new hypothesis probs
            words_prob = words_prob[words] * cur_prob
            new_probs.append(words_prob)

            # Generate hypothesis with a new word
            for word in words:
                new_hyp = torch.cat((cur_hyp, word.unsqueeze(0)), dim=0)
                new_hyps.append(new_hyp)
                new_hiddens.append(hidden)
                new_cells.append(cell)
                new_attendeds.append(att)

        # Get the best of the best
        new_hyps, new_hiddens, new_cells, new_attendeds, new_probs = \
            torch.stack(new_hyps), \
            torch.stack(new_hiddens), \
            torch.stack(new_cells), \
            torch.stack(new_attendeds), \
            torch.cat(new_probs)
        best_hyps = new_probs.argsort()[-beam_size:]
        cur_hyps, cur_probs, cur_hiddens, cur_cells, cur_attendeds = \
            new_hyps[best_hyps], new_probs[best_hyps], \
            new_hiddens[best_hyps], new_cells[best_hyps], \
            new_attendeds[best_hyps]

        # Working with END
        end_indices = torch.flatten((cur_hyps[:, -1] == w2i[END]).nonzero())
        end_indices = end_indices.flip(dims=(0, ))
        for index in end_indices:
            final_hyps.append(cur_hyps[index])
            final_probs.append(cur_probs[index])
        filter_index = cur_hyps[:, -1] != w2i[END]
        cur_hyps, cur_probs, cur_hiddens, cur_cells, cur_attendeds = \
            cur_hyps[filter_index], \
            cur_probs[filter_index], \
            cur_hiddens[filter_index], \
            cur_cells[filter_index], \
            cur_attendeds[filter_index]

    final_probs = torch.tensor(final_probs)
    final_sort = final_probs.argsort(descending=True)
    final_hyps = [final_hyps[i] for i in final_sort]
    final_probs = final_probs[final_sort]

    return final_hyps[:beam_size]
