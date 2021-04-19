import torch
# from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
import math

# bertMaskedLM = BertForMaskedLM.from_pretrained('bert-base-uncased')
# bertMaskedLM.eval()
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
# def get_score(sentence):
#     tokenized_input = tokenizer.tokenize(sentence)
#     tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_input)])
#     predictions = bertMaskedLM(tensor_input)
#     print(tokenized_input)
#     print(tensor_input)
#     print(predictions)
#     print(tensor_input.squeeze().size())
#     print(predictions.squeeze().size())
#     loss_func = torch.nn.CrossEntropyLoss()
#     loss = loss_func(predictions.squeeze(), tensor_input.squeeze()).data
#     return 1/math.exp(loss)
#
# sample_inputs = ['I just want to go to sleep',
#                  'I just want to go to cow',
#                  'Cow animal cat nugget',
#                  'Hello World']
#
# for input in sample_inputs:
#     print('Sentence: {}, Fluency: {}'.format(input, str(get_score(input))))

# Load pre-trained model (weights)
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
model.eval()
# Load pre-trained model tokenizer (vocabulary)
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

def score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss=model(tensor_input, lm_labels=tensor_input)
    return math.exp(loss)


sample_inputs = ['I am happy',
                 'I just want to go to sleep',
                 'I just want to go to cow',
                 'Cow animal cat nugget',
                 'Hello World']
print([score(i) for i in sample_inputs])
