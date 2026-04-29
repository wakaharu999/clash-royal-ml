# 事前学習用のデータセットとTransformerモデルの定義

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import random
import torch.nn as nn
import math

class ClashRoyaleDataset(Dataset):
    def __init__(self, csv_file, json_file):
        # データの読み込み
        self.df = pd.read_csv(csv_file, header=None)
        with open(json_file, 'r', encoding='utf-8') as f:
            self.cards = json.load(f)
            
        # 生のカードIDを0からの連番（インデックス）に変換する辞書を作成
        self.raw_ids = list(self.cards.keys())
        self.id_to_idx = {int(raw_id): i for i, raw_id in enumerate(self.raw_ids)}
        
        # MASK（隠しカード）用の特殊トークン番号を最後に割り当てる
        self.mask_idx = len(self.id_to_idx)
        
        # 全カード種類 + MASKトークンで語彙サイズ（モデルが扱う単語の種類数）を決定
        self.vocab_size = len(self.id_to_idx) + 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1行取得
        row = self.df.iloc[idx].values
        
        # 最初の2列(名前, タグ)を飛ばして、カードID(8枚)を取得し連番に変換
        deck_raw = row[2:10]
        deck_idx = [self.id_to_idx[int(c)] for c in deck_raw]
        
        # ランダムに1枚選んで隠す（穴埋め問題の作成）
        mask_pos = random.randint(0, 7)
        target = deck_idx[mask_pos] # これが予測すべき「正解」
        
        input_deck = deck_idx.copy()
        input_deck[mask_pos] = self.mask_idx # 選んだ場所をMASK番号で上書き
        
        # PyTorchのテンソルに変換して返す (入力デッキ, 正解のカード)
        return torch.tensor(input_deck, dtype=torch.long), torch.tensor(target, dtype=torch.long)

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
        
        # 3. 出力層（各ポジションに入るカードが「全122種類のうちどれか」を確率で出力）
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