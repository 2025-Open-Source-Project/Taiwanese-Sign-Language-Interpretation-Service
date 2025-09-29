import pickle
from ckiptagger import WS
ws_driver = WS("./data")  # 注意路徑要對

def tokenize(sentence):
    return ws_driver([sentence])[0]  # 回傳詞的 list

def build_input_vocab(sentences):
    """建中文輸入的詞級詞表"""
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    idx = 4
    for sentence in sentences:
        tokens = tokenize(sentence)  # 斷詞
        for word in tokens:  # 每一個字
            if word and word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

def build_output_vocab(sentences):
    """建手語輸出的詞級詞表"""
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    idx = 4
    for sentence in sentences:
        for word in sentence.split('/'):  # 用 / 分詞
            if word and word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

def save_vocab(vocab, path):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)

def load_vocab(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
