import torch.nn as nn

# 根據Encoder提供的隱藏狀態和記憶單元，逐步生成輸出序列
class Decoder(nn.Module):
    # output_dim: 輸出詞彙表的大小（即可能的輸出詞或字的總數）, embed_dim: 嵌入層的維度（每個詞或字的嵌入向量大小), hidden_dim: LSTM隱藏層的維度
    # 這裡的hidden_dim應該和Encoder的hidden_dim相同，因為Decoder需要使用Encoder的隱藏狀態和記憶單元來生成輸出序列 
    def __init__(self, output_dim, embed_dim, hidden_dim):
        super().__init__()
        # 使用嵌入層、LSTM 和全連接層來生成輸出序列
        # 將輸出序列中的詞或字轉換為嵌入向量
        # 解碼器的輸入是詞的索引（整數），這些索引需要轉換為嵌入向量才能被 LSTM 處理
        self.embedding = nn.Embedding(output_dim, embed_dim)
        # 生成序列的上下文資訊
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True) #, dropout=0.3
        # 將 LSTM 的隱藏狀態映射到輸出詞彙表的大小，生成每個詞的概率分佈
        self.fc = nn.Linear(hidden_dim, output_dim)

    # input形狀為 (batch_size,)，即每個批次的詞索引, hidden: 編碼器或前一時間步的隱藏狀態, cell: 編碼器或前一時間步的記憶單元
    def forward(self, input, hidden, cell):
        # 形狀從 (batch_size,) 變為 (batch_size, 1)
        input = input.unsqueeze(1)
        # 輸入詞轉換為嵌入向量，輸出形狀為 (batch_size, 1, embed_dim)
        embedded = self.embedding(input)
        # 將嵌入向量和隱藏狀態、記憶單元輸入到 LSTM 中
        # output: 當前時間步的隱藏狀態, (hidden, cell): 更新後的隱藏狀態和記憶單元
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # prediction: 當前時間步的詞概率分佈形狀為 (batch_size, output_dim)
        prediction = self.fc(output.squeeze(1))
        # 回傳 當前時間步的詞概率分佈、更新後的隱藏狀態和記憶單元
        return prediction, hidden, cell
