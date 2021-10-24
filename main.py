import torch.nn.functional as F
from transformers import pipeline
from transformers import AutoTokenizer , AutoModelForSequenceClassification
import torch

import torch.nn.functional as f

model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline(task='sentiment-analysis',model=model, tokenizer= tokenizer)
results = classifier('hate it')
                  
print(results)

tokens = tokenizer.tokenize('happy to learn')

token_ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids = tokenizer('happy to learn')

print('tokens : {}'.format(tokens))
print('token_ids : {}'.format(token_ids))
print('input ids : {}'.format(input_ids))


X_train= ['happy to learn','to learning pytorch']

batch = tokenizer(X_train,padding=True, truncation=True, max_length=512,return_tensors='pt')
print(batch)

with torch.no_grad():
    outputs = model(**batch, labels = torch.tensor([1,0]))
    print(outputs)
    predictions = F.softmax(outputs.logits,dim=1)
    print(predictions)
    labels = torch.argmax(predictions,dim= 1)
    print(labels)
    labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
    print(labels)