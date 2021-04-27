import torch
# from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
import math
import time
import html
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

def calculate_fluency(sentence):
    sentence = html.unescape(sentence)
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss=model(tensor_input, lm_labels=tensor_input)

    return math.exp(loss)


# sample_inputs = ["I'm happy",
#                  "I &apos;m happy",
#                  'I just want to go to sleep',
#                  'I just want to go to cow',
#                  'Cow animal cat nugget',
#                  'Hello World']
#
# sample_inputs = ["I 'd like to say that you 're very special",
#                  "I &apos;d like to say that you &apos;re very special"]

# sample = "I &apos;d like to say that you &apos;re very special"
sample = 'i had a voyage to tens of thousands of of canopy attacks , ' \
         'one place that went , was a little bit more , didn &apos;t have that small typing ' \
         '-- that tube -- had caught all 200 feet in the little fish .'

################ Current Implementation ######################

start = time.time()

sample_words = sample.strip().lower().split()

fluency_scores = []

for i in range(1, len(sample_words)):

    sent = ' '.join([word for word in sample_words[:i+1]])
    fluency_scores.append(calculate_fluency(sent))

end = time.time()
print(end-start)

print(fluency_scores)

################ New Implementation ######################

# sample = "I &apos;d like to say that you &apos;re very special"
sample = 'i had a voyage to tens of thousands of of canopy attacks , ' \
         'one place that went , was a little bit more , didn &apos;t have that small typing ' \
         '-- that tube -- had caught all 200 feet in the little fish .'

def calculate_fluency(tensor_input):

    loss=model(tensor_input, lm_labels=tensor_input)

    return math.exp(loss)

start = time.time()

sample = html.unescape(sample)
sample_words = sample.strip().lower().split()

sent_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample_words[0]))
fluency_scores = []

for i in range(1, len(sample_words)):

    sent_ids += tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample_words[i]))
    tensor_input = torch.tensor([sent_ids])
    fluency_scores.append(calculate_fluency(tensor_input))

end = time.time()
print(end-start)

print(fluency_scores)