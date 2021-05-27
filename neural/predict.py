import torch
from nltk.tokenize import sent_tokenize
from torchtext.data.utils import get_tokenizer

from neural.model.ner.bilstm_cnn import BiLSTMConv
from neural.model.ner.dataloader import NERDataset
from neural.model.summarization.dataloader import SummarizationDataset, SpecialTokens
from neural.model.summarization.pointer_generator import PointerGeneratorNetwork
from news.article import NewsArticle
from utils.general import set_random_seed


class NewsPredictor:
    def __init__(self, use_cuda: bool):
        set_random_seed(0)
        self.__device = 'cuda' if use_cuda else 'cpu'
        self.__ner_dataset = NERDataset('../data/ner_dataset.csv', embedding='glove.6B.50d')
        self.__ner = BiLSTMConv(self.__ner_dataset.vocab.vectors, output_size=self.__ner_dataset.labels_count,
                                batch_size=9, char_count=self.__ner_dataset.char_count,
                                max_word_length=self.__ner_dataset.max_word_length, char_embedding_size=25)
        self.__load_ner_model()
        self.__summarization_dataset = SummarizationDataset('cnn_dailymail', max_article_length=400,
                                                            max_summary_length=100, vocab_size=50000, get_oov=True,
                                                            vocab_dir='../data/vocabs', data_dir='../data/datasets')

        bos_index = self.__summarization_dataset.token_to_index(SpecialTokens.BOS)
        self.__summarization = PointerGeneratorNetwork(50000 + len(SpecialTokens), bos_index)
        self.__load_summarization_model()
        self.__tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

    def __load_ner_model(self, path_to_weights: str = '../data/weights/model') -> None:
        weights = torch.load(path_to_weights)
        self.__ner.load_state_dict(weights)
        self.__ner.to(self.__device)
        self.__ner.eval()
        del weights

    def __load_summarization_model(self, path_to_weights: str = '../data/weights/summarization-model-2.pt') -> None:
        weights = torch.load(path_to_weights)
        self.__summarization.load_state_dict(weights['model_state_dict'])
        self.__summarization.to(self.__device)
        self.__summarization.eval()
        del weights

    def __create_summarization(self, article_content: str) -> str:
        tokens = self.__tokenizer(article_content)
        tokens = [SpecialTokens.BOS.value] + tokens + [SpecialTokens.EOS.value]
        article_tensor = torch.tensor([self.__summarization_dataset.token_to_index(token.lower()) for token in tokens],
                                      device=self.__device)
        article_tensor = article_tensor.unsqueeze(1)
        article_length = torch.tensor(len(tokens)).unsqueeze(0)

        summary_out, _, _ = self.__summarization(article_tensor, article_length)
        summary = []
        for output in summary_out:
            token = torch.argmax(output).item()
            summary.append(self.__summarization_dataset.index_to_token(token))

        return ' '.join(summary)

    def __set_named_entities(self, article: NewsArticle) -> NewsArticle:
        index = 0
        for sentence in sent_tokenize(article.content):
            tokens = self.__tokenizer(sentence)
            tokens = [token for token in tokens if token.strip()]
            chars = []
            for token in tokens:
                chars.append([self.__ner_dataset.char_to_index[char] if char in self.__ner_dataset.char_to_index else 0
                              for char in token])
            chars = self.pad_chars_sequence(chars, len(tokens))
            chars = chars.repeat((1, 9, 1))
            article_tensor = torch.tensor([self.__ner_dataset.vocab.stoi[token] for token in tokens],
                                          device=self.__device)
            article_tensor = article_tensor.repeat((9, 1)).permute(1, 0)

            tags = self.__ner(article_tensor, chars)
            tags = torch.argmax(tags, dim=-1)
            tags = tags[:, 0]

            for tag, word in zip(tags, self.__tokenizer(sentence)):
                tag = self.__ner_dataset.get_label_name(tag.item())
                if tag not in ['<pad>', 'O']:
                    tag = tag.split('-')[1]
                    article.named_entities[index] = tag
                index += 1

        return article

    def pad_chars_sequence(self, chars_sequence, sentence_max_length: int):
        padded_sequences = []
        for chars in chars_sequence:
            padded_sequences.append(self.pad_single_sequence(chars))
        for _ in range(sentence_max_length - len(padded_sequences)):
            padded_sequences.append(torch.zeros(self.__ner_dataset.max_word_length, dtype=int, device=self.__device))
        return torch.stack(padded_sequences, dim=1).unsqueeze(0).permute(2, 0, 1)

    def pad_single_sequence(self, chars):
        forward_pad_length = (self.__ner_dataset.max_word_length - len(chars)) // 2
        backward_pad_length = forward_pad_length + (self.__ner_dataset.max_word_length - len(chars)) % 2

        return torch.cat([torch.zeros(forward_pad_length, dtype=int, device=self.__device),
                          torch.tensor(chars, device=self.__device),
                          torch.zeros(backward_pad_length, dtype=int, device=self.__device)], dim=0)

    def process_article(self, article: NewsArticle) -> NewsArticle:
        summary = self.__create_summarization(article.content)
        article.summary = summary
        article = self.__set_named_entities(article)

        return article
