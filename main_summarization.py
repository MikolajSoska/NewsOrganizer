import torch
import torch.nn as nn

import model.utils as utils
from model.text_summarization.dataloader import SummarizationDataset, SummarizationDataLoader
from model.text_summarization.pointer_generator import PointerGeneratorNetwork
from utils import set_random_seed

device = 'cuda'
load_checkpoint = False
set_random_seed(0)
epochs = 1
batch_size = 16
dataset = SummarizationDataset('cnn_dailymail', max_article_length=400, max_summary_length=100)
loader = SummarizationDataLoader(dataset, batch_size=batch_size)
model = PointerGeneratorNetwork(dataset.get_vocab_size(), batch_size)
model.to(device)

criterion = nn.NLLLoss()
criterion_coverage = utils.CoverageLoss()
optimizer = torch.optim.Adam(model.parameters())

if load_checkpoint:
    checkpoint = torch.load('data/weights/summarization-model-cpu.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

for epoch in range(epochs):
    for i, (texts, texts_lengths, summaries, summaries_lengths, targets) in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        texts = texts.to(device=device)
        summaries = summaries.to(device=device)

        output = model(texts, texts_lengths, summaries)
        step_losses = []
        for prediction, attention, coverage, summary in zip(*output, summaries):
            step_loss = criterion(prediction, summary) + criterion_coverage(attention, coverage)
            step_losses.append(step_loss)

        loss = torch.sum(torch.stack(step_losses))
        loss.backward()
        optimizer.step()

        if i % 10 == 1:
            print(f'Epoch: {epoch} Iter: {i}/{len(loader)} Loss: {loss}')
            torch.save({
                'epoch': epoch,
                'model_state_dict':
                    model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, 'data/weights/summarization-model.pt')
