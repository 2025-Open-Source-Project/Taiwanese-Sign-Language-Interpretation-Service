import torch
from models.encoder import Encoder
from models.decoder import Decoder
from utils.vocab_builder import load_vocab
from config import *

from ckiptagger import WS
ws_driver = WS("./data")

def encode_input(sentence, vocab):
    tokens = ws_driver([sentence])[0]
    print(tokens)
    ids = [vocab.get(tok, vocab['<UNK>']) for tok in tokens if tok]
    return [vocab['<SOS>']] + ids + [vocab['<EOS>']]

def translate(model_path, sentence):
    input_vocab = load_vocab('vocab/input_vocab.pkl')
    output_vocab = load_vocab('vocab/output_vocab.pkl')
    idx2word = {v: k for k, v in output_vocab.items()}

    input_ids = encode_input(sentence, input_vocab)
    input_tensor = torch.tensor(input_ids).unsqueeze(0)

    encoder = Encoder(len(input_vocab), EMBED_SIZE, HIDDEN_SIZE)
    decoder = Decoder(len(output_vocab), EMBED_SIZE, HIDDEN_SIZE)

    encoder.load_state_dict(torch.load(f"{model_path}_encoder.pt"))
    decoder.load_state_dict(torch.load(f"{model_path}_decoder.pt"))

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        hidden, cell = encoder(input_tensor)
        dec_input = torch.tensor([output_vocab['<SOS>']])
        outputs = []

        for _ in range(20):
            pred, hidden, cell = decoder(dec_input, hidden, cell)
            pred_id = pred.argmax(1).item()
            if pred_id == output_vocab['<EOS>']:
                break
            outputs.append(idx2word.get(pred_id, '?'))
            dec_input = torch.tensor([pred_id])
        
    print("預測結果（手語語句）：", '/'.join(outputs))

if __name__ == "__main__":
    
    while True:
        sentence = input("請輸入中文句子（輸入 'exit' 退出）：")
        if sentence == "exit":
            break
        # 使用者輸入完整自然中文
        translate("saved_model", sentence)
