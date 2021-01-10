from nltk.tokenize import word_tokenize
import string
from collections import defaultdict

START = '<START>'
UNK = '<UNK>'
END = '<END>'
PAD = '<PAD>'


def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return word_tokenize(text)


def transform_text(text, w2i):
    text = clean_text(text)
    sequence = [w2i[START]]
    for word in text:
        if word in w2i:
            sequence.append(w2i[word])
        else:
            sequence.append(w2i[UNK])
    sequence.append(w2i[END])
    return sequence


def create_dictionary(dataset, min_word_freq=1):
    c = defaultdict(int)

    for image, texts in dataset:
        for text in texts:
            text = clean_text(text)
            for word in text:
                c[word] += 1

    c_filtered = [word for word in c if c[word] > min_word_freq]
    c_filtered.append(START)
    c_filtered.append(UNK)
    c_filtered.append(END)
    c_filtered.append(PAD)

    i2w = {}
    w2i = {}

    for index, word in enumerate(c_filtered):
        i2w[index] = word
        w2i[word] = index

    return w2i, i2w


def create_dictionary_from_annotations(annotations, min_word_freq=1):
    c = defaultdict(int)

    for annotation in annotations['annotations']:
        text = clean_text(annotation['caption'])
        for word in text:
            c[word] += 1

    c_filtered = []
    c_filtered.append(PAD)
    c_filtered.append(START)
    c_filtered.append(UNK)
    c_filtered.append(END)
    c_filtered += [word for word in c if c[word] > min_word_freq]

    i2w = {}
    w2i = {}

    for index, word in enumerate(c_filtered):
        i2w[index] = word
        w2i[word] = index

    assert w2i[PAD] == 0, 'Padding has nonzero index in a dictionary'
    return w2i, i2w
