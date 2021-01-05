import evaluation
from beam_search import beam_search
from tqdm import tqdm


def train(model, dataloader, criterion, optimizer, device, data_mode):
    assert data_mode == 'packed' or data_mode == 'padded'

    total_loss = 0.0
    total_samples = 0
    for image, inputs, outputs in tqdm(dataloader, total=len(dataloader)):
        image = image.to(device)
        inputs = inputs.to(device)
        outputs = outputs.to(device)

        total_samples += image.shape[0]
        optimizer.zero_grad()

        ans, _ = model(image, inputs)

        if data_mode == 'packed':
            ans = ans.data
            outputs = outputs.data
        elif data_mode == 'padded':
            ans = ans.view(-1, ans.shape[-1])
            outputs = outputs.view(-1)

        loss = criterion(ans, outputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * image.shape[0]
    total_loss /= total_samples

    return total_loss

def validate(model, dataloader, device, w2i, i2w, max_length=15, beam_size=3):
    gts_dict = {}
    hyps_dict = {}
    bad_count = 0
    for i, (image, texts) in tqdm(enumerate(dataloader), total=len(dataloader)):
        hyps = beam_search(model, image, w2i, i2w, device, max_length, beam_size)
        if len(hyps) == 0:
            bad_count += 1
            continue
        hyp = hyps[0][1:]
        if hyp[-1] == w2i['<END>']:
            hyp = hyp[:-1]
        hyp = ' '.join([i2w[word] for word in hyp.tolist()])
        gts_dict[i] = texts[0]
        # Temporary
        gts_dict[i] = ' '.join([i2w[idx.item()] for idx in gts_dict[i]])
        hyps_dict[i] = [hyp]
    if len(list(gts_dict.keys())) == 0:
        print('Bad validation')
        return {}
    print('Bad hypothesis count: ', bad_count)
    return evaluation.compute_scores(gts_dict, hyps_dict)
