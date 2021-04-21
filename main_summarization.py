import time

import torch

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

criterion = utils.SummarizationLoss()
criterion_coverage = utils.CoverageLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.15, initial_accumulator_value=0.1)

if load_checkpoint:
    checkpoint = torch.load('data/weights/summarization-model-cpu.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

time_start = time.time()
for epoch in range(epochs):
    for i, (texts, texts_lengths, summaries, summaries_lengths, targets) in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        texts = texts.to(device=device)
        texts_lengths = texts_lengths.to(device=device)
        summaries = summaries.to(device=device)
        summaries_lengths = summaries_lengths.to(device=device)
        targets = targets.to(device=device)

        output, attention, coverage = model(texts, texts_lengths, summaries)
        loss = criterion(output, targets, summaries_lengths) + criterion_coverage(attention, coverage, targets)
        loss.backward()
        optimizer.step()

        if i % 10 == 1:
            time_iter = round(time.time() - time_start, 2)
            memory = round(torch.cuda.memory_reserved(0) / (1024 ** 3), 2)  # To GB
            print(f'Epoch: {epoch} Iter: {i}/{len(loader)} Loss: {loss}, Time: {time_iter} seconds, '
                  f'Memory: {memory} GB')
            torch.save({
                'epoch': epoch,
                'model_state_dict':
                    model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, 'data/weights/summarization-model.pt')
            time_start = time.time()
