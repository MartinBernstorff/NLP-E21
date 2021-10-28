from typing import List
import gensim.downloader as api
from datasets import load_dataset
import pandas as pd
import os
import pprint as pprint
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# DATASET
dataset = load_dataset("conllpp")

train = dataset["train"]

# Inspect the dataset
train_inspect = pd.DataFrame(train)
train_inspect["pos_tags"].head()

train["tokens"][0]
train["ner_tags"][0]
train["chunk_tags"][0]

# The dataset consists of 5 columns. Each row contains: 
#   tokens: a sentence (represented as a series), varying length
#   pos_tags: POS true labels
#   chunk_tags
#   ner_tags: NER true labels

num_classes = train.features["ner_tags"].feature.num_classes

# CONVERTING EMBEDDINGS
import numpy as np
import torch

model = api.load("glove-wiki-gigaword-50")

from embedding import gensim_to_torch_embedding

# Convert gensim word embedding to torch word embedding
# Outputs an "embedding layer" and a vocabulary
embedding_layer, vocab = gensim_to_torch_embedding(model)


print(vocab) 
# Appears vocab is a dict with words as keys and an int as val
# The ints are ordered, so maybe they're IDs? 

print(model.key_to_index)


# PREPARING A BATCH
def tokens_to_idx(tokens, vocab=model.key_to_index):
    """
    Converts tokens to model idx

    Parameters:
       tokens: An iterable of tokens
       vocab: A dictionary with words as keys and model-indeces as values
    
    Returns:
        A list of model-indeces for each token # TODO: Beware, is it really a list? Also does something with "UNK"

    Ideas to understand this function:
    - Write documentation for this function including type hints for each arguement and return statement
    - What does the .get method do?
    - Why lowercase?
    """
    return [vocab.get(t.lower(), vocab["UNK"]) for t in tokens]


# sample batch of 10 sentences
batch_tokens = train["tokens"][:10]
batch_tags = train["ner_tags"][:10]
batch_tok_idx = [tokens_to_idx(sent) for sent in batch_tokens]
batch_size = len(batch_tokens)

# compute length of longest sentence in batch
batch_max_len = max([len(s) for s in batch_tok_idx])

# prepare a numpy array with the data, initializing the data with 'PAD'
# and all labels with -1; initializing labels to -1 differentiates tokens
# with tags from 'PAD' tokens
batch_input = vocab["PAD"] * np.ones((batch_size, batch_max_len))
batch_labels = -1 * np.ones((batch_size, batch_max_len))

# copy the data to the numpy array
for i in range(batch_size):
    tok_idx = batch_tok_idx[i]
    tags = batch_tags[i]
    size = len(tok_idx)

    batch_input[i][:size] = tok_idx
    batch_labels[i][:size] = tags


# since all data are indices, we convert them to torch LongTensors
batch_input, batch_labels = torch.LongTensor(batch_input), torch.LongTensor(
    batch_labels
)

# CREATE MODEL
from LSTM import RNN

model = RNN(
    embedding_layer=embedding_layer, output_dim=num_classes + 1, hidden_dim_size=256
)

# FORWARD PASS
X = batch_input
y = model(X)

loss = model.loss_fn(outputs=y, labels=batch_labels)
# loss.backward()