import torch
import numpy as np

from utils.text_utils import START, END


def beam_search(model, image, w2i, i2w, device, max_length=15, beam_size=1):
    # Move model and image to the same device
    model = model.to(device)
    image = image.to(device)

    # Initialize list with final generated captions
    final_hyps = []
    final_probs = []

    # Calculate initial hidden state
    h0 = model.encoder(image)

    # Initialize start hypothesis, their probabilities and hidden states
    cur_hyps = torch.tensor([[w2i[START]]])
    cur_probs = torch.tensor([1.])
    cur_hiddens = h0

    for i in range(max_length):
        # Get last words and prepare them as an input
        last_words = [hyp[-1].unsqueeze(0) for hyp in cur_hyps]
        packed_inputs = torch.nn.utils.rnn.pack_sequence(last_words, enforce_sorted=True)

        # Use model decoder to get words probabilities and hidden states
        cur_hiddens = cur_hiddens.unsqueeze(0)
        words_probs, hiddens = model.decoder(cur_hiddens, packed_inputs)

        words_probs, _ = torch.nn.utils.rnn.pad_packed_sequence(words_probs)
        words_probs, hiddens = words_probs.squeeze(0), hiddens.squeeze(0)
        # TEMP WHILE MODEL DON'T HAVE SOFTMAX
        words_probs = torch.nn.Softmax(dim=1)(words_probs)

        # Determine the most probable words for each hypothesis
        max_words = words_probs.argsort(dim=1)[:, -beam_size:]

        # Generate new hypothesis and count their probabilities
        new_hyps, new_probs, new_hiddens = [], [], []
        for cur_prob, cur_hyp, words_prob, hidden, words in zip(cur_probs, cur_hyps, words_probs, hiddens, max_words):
            # Calculate new hypothesis probs
            words_prob = words_prob[words] * cur_prob
            new_probs.append(words_prob)

            # Generate hypothesis with a new word
            for word in words:
                new_hyp = torch.cat((cur_hyp, word.unsqueeze(0)), dim=0)
                new_hyps.append(new_hyp)
                new_hiddens.append(hidden)

        # Get the best of the best
        new_hyps, new_hiddens, new_probs = torch.stack(new_hyps), torch.stack(new_hiddens), torch.cat(new_probs)
        best_hyps = new_probs.argsort()[-beam_size:]
        cur_hyps, cur_probs, cur_hiddens = new_hyps[best_hyps], new_probs[best_hyps], new_hiddens[best_hyps]

        # Working with END
        end_indices = torch.flatten((cur_hyps[: ,-1] == w2i[END]).nonzero())
        end_indices = end_indices.flip(dims=(0,))
        for index in end_indices:
            final_hyps.append(cur_hyps[index])
            final_probs.append(cur_probs[index])
        filter_index = cur_hyps[:, -1] != w2i[END]
        cur_hyps, cur_probs, cur_hiddens = cur_hyps[filter_index], cur_probs[filter_index], cur_hiddens[filter_index]

    final_probs = torch.tensor(final_probs)
    final_sort = final_probs.argsort(descending=True)
    final_hyps =  [final_hyps[i] for i in final_sort]
    final_probs = final_probs[final_sort]

    return final_hyps[:beam_size]
