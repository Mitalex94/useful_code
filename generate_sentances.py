from transformers import GPT2Tokenizer
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
from scipy.special import softmax
import random

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def clear_last(posibilities, count):
    for _ in range(count):
        posibilities[posibilities.argmax()]=0

def generate_sentances(our_sentance, amount, length):
    for j in range (amount):   
        sentance = our_sentance
        for i in range(length):
            inputs = tokenizer(sentance, return_tensors="pt")
            outputs = model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits.detach().numpy()
            x = softmax(logits[0, -1])
            clear_last(x, random.randint(1,3))
            sentance += tokenizer.decode(int(x.argmax()))
        print(sentance)
        
sentence = input()

generate_sentances(sentence, 15, 10)
