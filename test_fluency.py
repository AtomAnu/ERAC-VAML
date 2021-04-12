import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import pandas as pd
import math

bertMaskedLM = BertForMaskedLM.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_score(sentence):
    tokenized_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_input)])
    predictions = bertMaskedLM(tensor_input)
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(predictions.squeeze(), tensor_input.squeeze()).data
    return math.exp(loss)

sample_inputs = ['I just want to go to sleep',
                 'I just want to go to cow',
                 'Cow animal cat nugget',
                 'Hello']

for input in sample_inputs:
    print('Sentence: {}, Fluency: {}'.format(input, str(get_score(input))))