import torch
import numpy as np

from utils.text_utils import START, END

def simple_beam_search(model, image, w2i, i2w, device, max_length=15, beam_size=1):
    # Here are two problems
    # 1. Size of new_hyps on every iteration is beam_size ** 2,
    # while we can use only 2 * beam_size memory
    # 2. Here are some cycles that can be replaced with a numpy vectorized operations
    #image = transform(image)
    #image = torch.unsqueeze(image, 0)

    model = model.to(device)
    image = image.to(device)

    cur_hyps = [[w2i[START]]]
    cur_probs = [1.]
    cur_hiddens = model.encoder(image)
    cur_hiddens = torch.unsqueeze(cur_hiddens, 0)
    for i in range(max_length):
        packed_inputs = torch.nn.utils.rnn.pack_sequence(
            [torch.tensor([hyp[-1]]) for hyp in cur_hyps], enforce_sorted=True)
        cur_hiddens = cur_hiddens.to(device)
        packed_inputs = packed_inputs.to(device)
        probs, hiddens = model.decoder(cur_hiddens, packed_inputs)
        new_hyps = []
        new_probs = []
        new_hiddens = []
        for hyp, cur_prob, prob, hidden in zip(cur_hyps, cur_probs, probs.data, hiddens.data.tolist()[0]):
            if hyp[-1] == w2i[END]:
                new_hyps.append(hyp)
                new_probs.append(cur_prob)
                new_hiddens.append(hidden)
                continue
            max_words = torch.argsort(prob)[-beam_size:]
            for word in max_words:
                new_hyp = hyp.copy()
                new_hyp.append(word.item())
                new_hyps.append(new_hyp)
                new_probs.append(cur_prob * prob[word].item())
                new_hiddens.append(hidden)
        new_probs = np.array(new_probs)
        new_hiddens = torch.tensor(new_hiddens)
        best_hyps = np.argsort(new_probs)[-beam_size:]
        cur_probs = new_probs[best_hyps]
        cur_hiddens = new_hiddens[best_hyps]
        cur_hiddens = torch.unsqueeze(cur_hiddens, 0)
        cur_hyps = []
        for hyp_num in best_hyps:
            cur_hyps.append(new_hyps[hyp_num])

    assert(np.argmax(cur_probs) == len(cur_probs) - 1)

    return cur_hyps, cur_probs


def beam_search(model, image, w2i, i2w, device, max_length=15, beam_size=1):
    # Move model and image to the same device
    model = model.to(device)
    image = image.to(device)

    # Initialize list with final generated captions
    final_captions = []

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
        end_indices = (cur_hyps[: ,-1] == w2i[END])
        print(end_indices.nonzero())
        print(cur_hyps[:, -1])

    return final_captions
