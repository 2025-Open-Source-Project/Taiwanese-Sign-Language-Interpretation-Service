import torch
from torch.utils.data import Dataset

from ckiptagger import WS
ws_driver = WS("./data")

# 中文和手語句子轉換為數字化的張量，並提供索引操作

class TranslationDataset(Dataset):
    def __init__(self, data, input_vocab, output_vocab):
        self.data = data
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        
    def tokenize(self, sentence):
        return ws_driver([sentence])[0]

    def encode_input(self, sentence, input_vocab):
        """中文句子：逐字編碼"""
        tokens = self.tokenize(sentence)
        ids = [self.input_vocab.get(tok, self.input_vocab['<UNK>']) for tok in tokens if tok]
        return [self.input_vocab['<SOS>']] + ids + [self.input_vocab['<EOS>']]

    def encode_output(self, sentence, output_vocab):
        """手語句子：以/分詞編碼"""
        tokens = sentence.split('/')
        ids = [self.output_vocab.get(tok, self.output_vocab['<UNK>']) for tok in tokens if tok]
        return [self.output_vocab['<SOS>']] + ids + [self.output_vocab['<EOS>']]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chinese, sign = self.data.iloc[idx]
        x = self.encode_input(chinese, self.input_vocab)
        y = self.encode_output(sign, self.output_vocab)
        return torch.tensor(x), torch.tensor(y)
