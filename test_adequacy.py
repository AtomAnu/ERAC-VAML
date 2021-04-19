import os
import torch
import torch.nn as nn

from xlm.utils import AttrDict
from xlm.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from xlm.model.transformer import TransformerModel
import fastBPE

# Reload a pre-trained model
model_path = 'mlm_100_1280.pth'
reloaded = torch.load(model_path)
params = AttrDict(reloaded['params'])
print("Supported languages: %s" % ", ".join(params.lang2id.keys()))

# Build dictionary
dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
params.n_words = len(dico)
params.bos_index = dico.index(BOS_WORD)
params.eos_index = dico.index(EOS_WORD)
params.pad_index = dico.index(PAD_WORD)
params.unk_index = dico.index(UNK_WORD)
params.mask_index = dico.index(MASK_WORD)

# Build model
model = TransformerModel(params, dico, True, True)
model.eval()
model.load_state_dict(reloaded['model'])

sentences = ['Cat', 'Katze']

codes_path = 'codes_xnli_100'
vocab_path = 'vocab_xnli_100'

bpe = fastBPE.fastBPE(codes_path, vocab_path)
sentences = bpe.apply(sentences)

# # bpe-ize sentences
# sentences = to_bpe(sentences)
print('\n\n'.join(sentences))

# check how many tokens are OOV
n_w = len([w for w in ' '.join(sentences).split()])
n_oov = len([w for w in ' '.join(sentences).split() if w not in dico.word2id])
print('Number of out-of-vocab words: %s/%s' % (n_oov, n_w))

# add </s> sentence delimiters
sentences = [(('</s> %s </s>' % sent.strip()).split()) for sent in sentences]

bs = len(sentences)
slen = max([len(sent) for sent in sentences])

word_ids = torch.LongTensor(slen, bs).fill_(params.pad_index)
for i in range(len(sentences)):
    sent = torch.LongTensor([dico.index(w) for w in sentences[i]])
    word_ids[:len(sent), i] = sent

lengths = torch.LongTensor([len(sent) for sent in sentences])

langs = None

tensor = model('fwd', x=word_ids, lengths=lengths, langs=langs, causal=False).contiguous()
print(tensor.size())
print(tensor[0].size())

embeddings = tensor[0]
en_tensor = embeddings[0].unsqueeze(0)
de_tensor = embeddings[1].unsqueeze(0)

cos = nn.CosineSimilarity()
sim = cos(en_tensor, de_tensor)
print(sim)