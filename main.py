import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from dataloader import NERDataset, NERDataLoader
from model.ner.bilstm_cnn import BiLSTMConv

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

epochs = 5
batch_size = 9

dataset = NERDataset('data/ner_dataset.csv', embedding='glove.6B.50d')
loader = NERDataLoader(dataset, batch_size=batch_size)

model = BiLSTMConv(embeddings=dataset.vocab.vectors, output_size=dataset.labels_count,
                   batch_size=batch_size, char_count=dataset.char_count,
                   max_word_length=dataset.max_word_length, char_embedding_size=25)
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

for epoch in range(epochs):
    correct = 0
    all_tags = 0
    true_tags = []
    pred_tags = []
    for i, (sentences, labels, chars) in enumerate(loader):
        optimizer.zero_grad()
        sentences = sentences.to(device='cuda')
        labels = labels.to(device='cuda')
        chars = chars.to(device='cuda')

        prediction = model(sentences, chars)
        loss = criterion(prediction.view(-1, dataset.labels_count), labels.view(-1))
        loss.backward()
        optimizer.step()

        tags = torch.argmax(prediction, dim=-1)
        for prediction, tag in zip(tags.t(), labels.t()):
            pred_tags.append(prediction.cpu().numpy())
            true_tags.append(tag.cpu().numpy())

        correct += (tags == labels).sum().item()
        all_tags += tags.shape[0] * tags.shape[1]

        if i % 100 == 1:
            true_tags = np.concatenate(true_tags, axis=0)
            pred_tags = np.concatenate(pred_tags, axis=0)
            score = f1_score(true_tags, pred_tags, average='weighted')
            print(f'Epoch: {epoch} Iter: {i}/{len(loader)} Loss: {loss}, Accuracy: {correct / all_tags},'
                  f' F1 score: {score}')
            true_tags = []
            pred_tags = []

torch.save(model.state_dict(), 'models/model')
