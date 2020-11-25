import evaluation
from beam_search import simple_beam_search, beam_search
from tqdm import tqdm


def train(model, dataloader, criterion, optimizer, device):
    total_loss = 0.0
    total_samples = 0
    for image, inputs, outputs in tqdm(dataloader, total=len(dataloader)):
        image = image.to(device)
        inputs = inputs.to(device)
        outputs = outputs.to(device)

        total_samples += image.shape[0]
        optimizer.zero_grad()

        ans, _ = model(image, inputs)
        loss = criterion(ans.data, outputs.data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * image.shape[0]
    total_loss /= total_samples

    return total_loss

def validate(model, dataloader, device, w2i, i2w, max_length=15, beam_size=3):
    gts_dict = {}
    hyps_dict = {}
    for i, (image, texts) in tqdm(enumerate(dataloader), total=len(dataloader)):
        hyp, probs = simple_beam_search(model, image, w2i, i2w, device, max_length, beam_size)
        print(hyp)
        print(probs)
        hyp = hyp[-1][1:]
        hyp2, probs2 = beam_search(model, image, w2i, i2w, device, max_length, beam_size)
        print(hyp2)
        print(probs2)
        if hyp[-1] == w2i['<END>']:
            hyp = hyp[:-1]
        hyp = ' '.join([i2w[word] for word in hyp])
        gts_dict[i] = texts[0]
        hyps_dict[i] = [hyp]
    return evaluation.compute_scores(gts_dict, hyps_dict)
