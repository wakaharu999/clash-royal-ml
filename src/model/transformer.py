import torch
import torch.nn as nn
import math

class DeckTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1):
        super(DeckTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 1. Embedding層（カード番号をAIが理解できる多次元ベクトルに変換）
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # ★クラロワのデッキは「カードの順番」に意味がないため位置エンコーディングは使いません
        
        # 2. Transformer Encoder層（カード同士の相性や組み合わせを学習）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True # 入力データを (バッチサイズ, 8枚, ベクトル次元) の形にする
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 出力層（各ポジションに入るカードが「全121種類のうちどれか」を確率で出力）
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # xの形: (batch_size, 8)
        
        # ベクトル化
        embedded = self.embedding(x) * math.sqrt(self.d_model) # 形: (batch_size, 8, d_model)
        
        # Attention層でカード同士の関連性を計算
        out = self.transformer_encoder(embedded) # 形: (batch_size, 8, d_model)
        
        # 各ポジションの予測結果（スコア）を出力
        logits = self.fc_out(out) # 形: (batch_size, 8, vocab_size)
        
        return logits

# --- 動作確認用 ---
if __name__ == "__main__":
    # 先ほど確認した語彙サイズ「121」を指定してモデルを召喚
    model = DeckTransformer(vocab_size=121)
    
    # ダミーデータ（バッチサイズ4、8枚のカード）を作成して流し込んでみる
    # 0〜120のランダムな整数を生成
    dummy_input = torch.randint(0, 121, (4, 8))
    
    # モデルの推論を実行！
    output = model(dummy_input)
    
    print("--- ネットワーク通過後のテンソルサイズ ---")
    print(output.shape) 
    print("↑ [バッチ数(4), デッキの枚数(8), 語彙サイズ(121)] になっていれば成功です！")