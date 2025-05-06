import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
from tqdm import tqdm

from models.encoder import Encoder
from models.decoder import Decoder
from utils.vocab_builder import build_input_vocab,build_output_vocab, save_vocab
from utils.dataset import TranslationDataset
from config import *

# Step 1: Load data
df = pd.read_csv('data/translation_dataset.csv')
train_df, val_df = train_test_split(df, test_size=0.2)

# Step 2: Build vocab
input_vocab = build_input_vocab(train_df['chinese'])  # 中文句子
output_vocab = build_output_vocab(train_df['sign'])   # 手語句子
save_vocab(input_vocab, 'vocab/input_vocab.pkl')
save_vocab(output_vocab, 'vocab/output_vocab.pkl')

# Step 3: Create dataset

# 中文句子：逐字編碼
# 手語句子：以/分詞編碼
train_ds = TranslationDataset(train_df, input_vocab, output_vocab)
val_ds = TranslationDataset(val_df, input_vocab, output_vocab)

# 文和手語句子轉換為數字化的張量
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 4: Model
# 模型構建
encoder = Encoder(len(input_vocab), EMBED_SIZE, HIDDEN_SIZE).to(device)
decoder = Decoder(len(output_vocab), EMBED_SIZE, HIDDEN_SIZE).to(device)

# 自適應學習率的優化方法，能夠快速收斂，適合深度學習模型
criterion = nn.CrossEntropyLoss(ignore_index=0)
# 優化器會根據這些參數計算梯度並更新
# 模型初始化
enc_opt = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
dec_opt = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

# Step 5: Train
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    # 每個批次的訓練
    for batch in tqdm(train_loader):
        src_batch, trg_batch = zip(*batch)

        src = nn.utils.rnn.pad_sequence(src_batch, batch_first=True).to(device)
        trg = nn.utils.rnn.pad_sequence(trg_batch, batch_first=True).to(device)

        # 編碼器前向傳遞
        hidden, cell = encoder(src)
        
        # 解碼器逐步生成輸出
        dec_input = trg[:, 0]
        loss = 0

        for t in range(1, trg.size(1)):
            pred, hidden, cell = decoder(dec_input, hidden, cell)
            loss += criterion(pred, trg[:, t])
            dec_input = trg[:, t]  # Teacher forcing

        #  損失反向傳播與參數更新
        loss.backward()
        enc_opt.step()
        dec_opt.step()
        enc_opt.zero_grad()
        dec_opt.zero_grad()

        total_loss += loss.item() / trg.size(1)

    print(f"[Epoch {epoch+1}] Loss: {total_loss/len(train_loader):.4f}")

# 儲存模型
torch.save(encoder.state_dict(), "saved_model_encoder.pt")
torch.save(decoder.state_dict(), "saved_model_decoder.pt")
print("✅ 模型已經儲存成功！")

