import os
import torch

from xlm.utils import AttrDict
from xlm.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from xlm.model.transformer import TransformerModel


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large')
# xlmr.eval()

en_sent = 'Hello World!'
de_sent = 'Hallo Welt! Jungs'

# en_tokens = xlmr.encode(en_sent)
# de_tokens = xlmr.encode(de_sent)
#
# en_features = xlmr.extract_features(en_tokens)
# de_features = xlmr.extract_features(de_tokens)
#
# print(cosine_similarity(np.array(en_tokens.tolist()).reshape(1, -1), np.array(de_tokens.tolist()).reshape(1, -1)))