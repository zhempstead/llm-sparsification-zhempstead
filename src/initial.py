import json
from urllib.request import urlopen

from transformers import GPT2LMHeadModel, GPT2Tokenizer

ds_url = 'https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json'
ds = json.loads(urlopen(ds_url).read())['data']

model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')

for example in ds[:10]:
    text = example['story']
    for question in example['questions']:
        text += "\n\nQ: " + question['input_text'] + "\nA:"
        generated = model.generate(tokenizer.encode(text, return_tensors='pt'))
        import pdb; pdb.set_trace()

    

import pdb; pdb.set_trace()
