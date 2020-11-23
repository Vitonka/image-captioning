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

    return cur_hyps[-1]
