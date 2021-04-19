import os
import torch

from xlm.utils import AttrDict
from xlm.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from xlm.model.transformer import TransformerModel
import fastBPE

# from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large')
# xlmr.eval()

# en_sent = 'Hello World!'
# de_sent = 'Hallo Welt! Jungs'

# en_tokens = xlmr.encode(en_sent)
# de_tokens = xlmr.encode(de_sent)
#
# en_features = xlmr.extract_features(en_tokens)
# de_features = xlmr.extract_features(de_tokens)
#
# print(cosine_similarity(np.array(en_tokens.tolist()).reshape(1, -1), np.array(de_tokens.tolist()).reshape(1, -1)))

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

# Below is one way to bpe-ize sentences
# codes = "codes_xnli_100 vocab_xnli_100"  # path to the codes of the model
# codes = "codes_xnli_100"  # path to the codes of the model
# fastbpe = os.path.join(os.getcwd(), 'tools/fastBPE/fast')


# list of (sentences, lang)
sentences = [
    'once he had worn trendy italian leather shoes and jeans from paris that had cost three hundred euros .', # en
    'Le français est la seule langue étrangère proposée dans le système éducatif .', # fr
    'El cadmio produce efectos tóxicos en los organismos vivos , aun en concentraciones muy pequeñas .', # es
    'Nach dem Zweiten Weltkrieg verbreitete sich Bonsai als Hobby in der ganzen Welt .', # de
    'وقد فاز في الانتخابات في الجولة الثانية من التصويت من قبل سيدي ولد الشيخ عبد الله ، مع أحمد ولد داداه في المرتبة الثانية .', # ar
    '羅伯特 · 皮爾 斯 生於 1863年 , 在 英國 曼徹斯特 學習 而 成為 一 位 工程師 . 1933年 , 皮爾斯 在 直布羅陀去世 .', # zh
]

codes_path = 'codes_xnli_100'
vocab_path = 'vocab_xnli_100'

bpe = fastBPE.fastBPE(codes_path, vocab_path)
sentences = bpe.apply(sentences)

# def to_bpe(sentences):
#     # write sentences to tmp file
#     with open('/tmp/sentences.txt', 'w') as fwrite:
#         for sent in sentences:
#             fwrite.write(sent + '\n')
#
#     # apply bpe to tmp file
#     os.system('%s applybpe /tmp/sentences.txt /tmp/sentences.txt %s' % (fastbpe, codes))
#
#     # load bpe-ized sentences
#     sentences_bpe = []
#     with open('/tmp/sentences.txt') as f:
#         for line in f:
#             sentences_bpe.append(line.rstrip())
#
#     return sentences_bpe

# Below are already BPE-ized sentences

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

# NOTE: No more language id (removed it in a later version)
# langs = torch.LongTensor([params.lang2id[lang] for _, lang in sentences]).unsqueeze(0).expand(slen, bs) if params.n_langs > 1 else None
langs = None

tensor = model('fwd', x=word_ids, lengths=lengths, langs=langs, causal=False).contiguous()
print(tensor.size())
print(tensor[0].size())