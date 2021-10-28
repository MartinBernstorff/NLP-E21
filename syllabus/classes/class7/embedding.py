import torch
from torch import nn
import numpy as np

def gensim_to_torch_embedding(gensim_wv: gensim.models.Word2Vec, add_padding = True, add_unkwnown = True) -> nn.Embedding, nn.Embedding:
    """
    Converts a gensmim embedding to a pytorch embedding

    Parameters:
        gensim_wv, a gensim word2vec embedding

    Returns:
        embedding layer, pytorch embedding
        vocab, a gensim vocab
    """
    embedding_size = gensim_wv.vectors.shape[1]

    # create unknown and padding embedding
    unk_emb = np.mean(gensim_wv.vectors, axis=0).reshape((1, embedding_size))
    pad_emb = np.zeros((1, gensim_wv.vectors.shape[1]))

    # add the new embedding
    embeddings = np.vstack([gensim_wv.vectors, unk_emb, pad_emb])

    weights = torch.FloatTensor(embeddings)

    emb_layer = nn.Embedding.from_pretrained(embeddings=weights, padding_idx=-1)

    # creating vocabulary
    vocab = gensim_wv.key_to_index

    if add_padding == True:
        vocab["PAD"] = emb_layer.padding_idx

    if add_unknown == True:
        vocab["UNK"] = weights.shape[0] - 2

    return emb_layer, vocab