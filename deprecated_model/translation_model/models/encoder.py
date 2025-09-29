import torch.nn as nn

# 輸入序列轉換為隱藏狀態和記憶單元
class Encoder(nn.Module):
    # input_dim: 輸入詞彙表的大小, embed_dim: 嵌入層的維度(將離散的詞或字（例如文字、符號）轉換為連續的數值向量的方式)（每個詞或字的嵌入向量大小), hidden_dim: LSTM隱藏層的維度 
    def __init__(self, input_dim, embed_dim, hidden_dim):
        super().__init__()
        # 將輸入序列中的詞或字轉換為嵌入向量
        self.embedding = nn.Embedding(input_dim, embed_dim)
        # 提取序列的上下文資訊 
        # batch_first=True表示輸入的形狀是(batch_size, seq_len, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True) # , dropout=0.3

    def forward(self, src):
        embedded = self.embedding(src)
        # STM 的隱藏狀態和記憶單元會在每個時間步更新，並累積之前所有時間步的資訊
        # 最後一個時間步的隱藏狀態和記憶單元包含了整個序列的上下文資訊，因此可以作為輸入序列的壓縮表示
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: LSTM的輸出序列, hidden: 最後一個時間步的隱藏狀態(當前時間步的輸出，包含了輸入序列的上下文資訊), cell: 最後一個時間步的記憶單元(STM 的內部記憶，用於捕捉長期依賴資訊)
        return hidden, cell
