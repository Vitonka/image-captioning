from .bleu import Bleu
from .meteor import Meteor
from .rouge import Rouge
from .cider import Cider
from .spice import Spice
from .tokenizer import PTBTokenizer

def compute_scores(gts, gen):
    metrics = (Bleu(), Meteor(), Rouge(), Cider(), Spice())
    all_score = {}
    for metric in metrics:
        score, _ = metric.compute_score(gts, gen)
        if str(metric) == 'BLEU':
            for i, score_item in enumerate(score):
                all_score['BLEU' + str(i + 1)] = score_item
        else:
            all_score[str(metric)] = score

    return all_score
