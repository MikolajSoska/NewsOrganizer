import torch
import torch.nn as nn


class NamedEntityRecognition(nn.Module):
    def __init__(self, embeddings: torch.Tensor, output_size: int, batch_size: int, char_count: int,
                 max_word_length: int, char_embedding_size: int):
        super().__init__()
        self.char_features_size = 50
        self.hidden_size = 50
        self.batch_size = batch_size

        char_conv = nn.Conv1d(char_embedding_size, self.char_features_size, kernel_size=3)
        conv_len_out = self.__get_conv_length_out(max_word_length, char_conv)
        self.char_network = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=1),
            nn.Embedding(char_count, char_embedding_size, padding_idx=0),
            Permute(0, 2, 1),
            char_conv,
            nn.MaxPool1d(conv_len_out),
            View(-1, self.batch_size, self.char_features_size)
        )

        self.word_embedding = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(input_size=50 + self.char_features_size, hidden_size=self.hidden_size, num_layers=1,
                            bidirectional=True)

        self.forward_out = TimeDistributed(nn.Sequential(
            nn.Linear(self.hidden_size, output_size),
            nn.LogSoftmax(dim=1)
        ))
        self.backward_out = TimeDistributed(nn.Sequential(
            nn.Linear(self.hidden_size, output_size),
            nn.LogSoftmax(dim=1)
        ))

        self.hidden = self.__init_hidden()

    def __init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_size, device='cuda'),
                torch.randn(2, self.batch_size, self.hidden_size, device='cuda'))

    @staticmethod
    def __get_conv_length_out(input_length: int, conv: nn.Conv1d) -> int:
        return int((input_length + 2 * conv.padding[0] - conv.dilation[0] *
                    (conv.kernel_size[0] - 1) - 1) / conv.stride[0] + 1)

    def forward(self, sentences_in: torch.Tensor, chars_in: torch.Tensor) -> torch.Tensor:
        x_chars = self.char_network(chars_in)

        self.hidden = self.__init_hidden()
        x_sentences = self.word_embedding(sentences_in)
        x_sentences = torch.cat([x_sentences, x_chars], dim=-1)

        x_sentences, self.hidden = self.lstm(x_sentences, self.hidden)
        x_forward = x_sentences[:, :, :self.hidden_size]
        x_backward = x_sentences[:, :, self.hidden_size:]

        x_forward = self.forward_out(x_forward)
        x_backward = self.backward_out(x_backward)

        return x_forward + x_backward


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        x_reshape = x.contiguous().view(-1, x.size(-1))

        y = self.module(x_reshape)
        y = y.view(-1, x.size(1), y.size(-1))

        return y


class View(nn.Module):
    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        return x_in.view(self.shape)


class Permute(nn.Module):
    def __init__(self, *dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        return x_in.permute(self.dimensions)
