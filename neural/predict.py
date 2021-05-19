import torch
from torchtext.data.utils import get_tokenizer

from neural.model.summarization.dataloader import SummarizationDataset, SpecialTokens
from neural.model.summarization.pointer_generator import PointerGeneratorNetwork
from news.article import NewsArticle
from utils.general import set_random_seed


class NewsPredictor:
    def __init__(self):
        set_random_seed(0)

        self.__summarization_dataset = SummarizationDataset('cnn_dailymail', max_article_length=400,
                                                            max_summary_length=100, vocab_size=150000, get_oov=False,
                                                            vocab_dir='data/vocabs', data_dir='data/datasets')
        bos_index = self.__summarization_dataset.vocab.stoi[SpecialTokens.BOS.value]
        eos_index = self.__summarization_dataset.vocab.stoi[SpecialTokens.EOS.value]
        self.__summarization = PointerGeneratorNetwork(len(self.__summarization_dataset.vocab), bos_index, eos_index)
        self.__load_summarization_model()
        self.__tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

    def __load_summarization_model(self, path_to_weights: str = 'data/weights/summarization-model-2.pt') -> None:
        weights = torch.load(path_to_weights)
        self.__summarization.load_state_dict(weights['model_state_dict'])
        self.__summarization.eval()
        del weights

    def __create_summarization(self, article_content: str) -> str:
        tokens = self.__tokenizer(article_content)
        tokens = [SpecialTokens.BOS.value] + tokens + [SpecialTokens.EOS.value]
        article_tensor = torch.tensor([self.__summarization_dataset.vocab.stoi[token.lower()] for token in tokens])
        article_tensor = article_tensor.unsqueeze(1)
        article_length = torch.tensor(len(tokens)).unsqueeze(0)

        summary_out, _, _ = self.__summarization(article_tensor, article_length, article_tensor, 0)
        summary = []
        for output in summary_out:
            token = torch.argmax(output).item()
            summary.append(self.__summarization_dataset.vocab.itos[token])

        return ' '.join(summary)

    def process_article(self, article: NewsArticle) -> NewsArticle:
        summary = self.__create_summarization(article.content)
        article.summary = summary

        return article
