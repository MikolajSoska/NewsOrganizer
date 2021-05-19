import gc
import time

import torch

import neural.model.utils as utils
from neural.model.summarization.dataloader import SummarizationDataset, SummarizationDataLoader
from neural.model.summarization.pointer_generator import PointerGeneratorNetwork
from utils.general import set_random_seed

device = 'cuda'
load_checkpoint = False
set_random_seed(0)
epochs = 13
batch_size = 4
coverage_iterations = 48000
vocab_size = 50000

dataset = SummarizationDataset('cnn_dailymail', max_article_length=400, max_summary_length=100,
                               vocab_size=vocab_size, get_oov=True)
loader = SummarizationDataLoader(dataset, batch_size=batch_size)
model = PointerGeneratorNetwork(len(dataset.vocab))
model.to(device)

criterion = utils.SummarizationLoss()
criterion_coverage = utils.CoverageLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.15, initial_accumulator_value=0.1)

if load_checkpoint:
    print('Loading checkpoint...')
    checkpoint = torch.load('../data/weights/summarization-model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_start = checkpoint['epoch']
    previous_batch_size = checkpoint['batch_size']
    iteration = checkpoint['iteration'] * previous_batch_size // batch_size
    print(f'Epoch set to {epoch_start}. Iteration set to {iteration}.')
    del checkpoint
else:
    iteration = 0
    epoch_start = 0

gc.collect()
time_start = time.time()
running_loss = []
iterations_without_coverage = len(loader) * batch_size * epochs
iterations_count = (epoch_start + 1) * iteration

for epoch in range(epoch_start, epochs):
    for i, (texts, texts_lengths, summaries, summaries_lengths, texts_extended, targets, oov_list) in enumerate(loader):
        if i < iteration:
            continue
        else:
            iteration = 0

        if iterations_count >= iterations_without_coverage and not model.with_coverage:
            print(f'Iteration {iterations_count}. Activated coverage mechanism.')
            model.activate_coverage()

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        texts = texts.to(device=device)
        texts_lengths = texts_lengths.to(device=device)
        summaries = summaries.to(device=device)
        summaries_lengths = summaries_lengths.to(device=device)
        texts_extended = texts_extended.to(device=device)
        targets = targets.to(device=device)

        oov_size = len(max(oov_list, key=lambda x: len(x)))
        output, attention, coverage = model(texts, texts_lengths, texts_extended, oov_size, summaries)
        loss = criterion(output, targets, summaries_lengths)
        if coverage is not None:
            loss = loss + criterion_coverage(attention, coverage, targets)

        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
        iterations_count += 1

        if i % 50 == 1:
            time_iter = round(time.time() - time_start, 2)
            torch.save({
                'epoch': epoch,
                'iteration': i,
                'batch_size': batch_size,
                'model_state_dict':
                    model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, '../data/weights/summarization-model.pt')
            time_start = time.time()
            memory = round(torch.cuda.memory_reserved(0) / (1024 ** 3), 2)  # To GB
            loss = sum(running_loss) / len(running_loss)
            print(f'Epoch: {epoch} Iter: {i}/{len(loader)} Loss: {loss}, Time: {time_iter} seconds, '
                  f'Memory: {memory} GB')

            # article_tokens = texts[:, 0]
            # article = []
            # for token in article_tokens:
            #     if token == 0:
            #         continue
            #     article.append(dataset.get_vocab().itos[token])
            # print('Article:')
            # print(' '.join(article))
            # predictions = output
            # text = []
            # for prediction in predictions:
            #     index = torch.argmax(prediction[0])
            #     text.append(dataset.get_vocab().itos[index])
            # print('Summary:')
            # print(' '.join(text))
