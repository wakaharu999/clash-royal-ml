import torch
import torch.nn as nn

class DeckEncoder(nn.Module):
    def __init__(self, num_cards, embed_size=64, num_heads=4, num_layers=2):
        super().__init__()
        # カードID（0〜150程度に圧縮されたもの）をベクトルに変換
        self.embedding = nn.Embedding(num_cards, embed_size)
        
        # Transformerで「8枚のカード間のシナジー」を学習
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, 
            nhead=num_heads, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x: (batch_size, 8)
        emb = self.embedding(x)              # (batch_size, 8, embed_size)
        out = self.transformer(emb)          # (batch_size, 8, embed_size)
        
        # 8枚のカードの特徴量を平均して、1つの「デッキ全体の特徴ベクトル」にする
        deck_vec = out.mean(dim=1)           # (batch_size, embed_size)
        return deck_vec


class MatchupPredictor(nn.Module):
    def __init__(self, num_cards=200, embed_size=64):
        super().__init__()
        # 1. 双子ネットワークの「脳（共有パラメータ）」
        self.encoder = DeckEncoder(num_cards, embed_size)
        
        # 2. 2つのデッキベクトルを比較して相性スコアを出す層
        # 結合入力 [my_vec, op_vec, 差, 積] を想定するため embed_size * 4
        self.fc = nn.Sequential(
            nn.Linear(embed_size * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh() # 出力を -1 (圧倒的不利) から 1 (圧倒的有利) の範囲に収める
        )

    def forward(self, my_deck, op_deck):
        # それぞれのデッキをベクトル化（Siamese Networkの心臓部）
        my_vec = self.encoder(my_deck)
        op_vec = self.encoder(op_deck)
        
        # ベクトル同士の相互作用（相性）を計算
        diff = my_vec - op_vec
        mult = my_vec * op_vec
        
        # 特徴量をすべて結合して全結合層（MLP）へ
        interaction = torch.cat([my_vec, op_vec, diff, mult], dim=1)
        score = self.fc(interaction)
        
        return score