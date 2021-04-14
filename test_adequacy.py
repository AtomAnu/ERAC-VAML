import torch
xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large')
xlmr.eval()

en_sent = 'Hello World!'
de_sent = 'Hallo Welt!'

en_tokens = xlmr.encode(en_sent)
de_tokens = xlmr.encode(de_sent)

print(en_tokens.tolist())
print(de_tokens.tolist())

def cosine_similarity():
    pass