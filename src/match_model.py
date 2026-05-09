# matchup_model.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# ==========================================
# データセットの定義 (PyTorch用)
# ==========================================
class BattleDataset(Dataset):
    def __init__(self, decks_A, decks_B, labels):
        # 整数(raw_id)ならlong型、小数(multi-hot)ならfloat32型に自動調整
        dt = torch.long if decks_A.dtype == np.int64 else torch.float32
        self.decks_A = torch.tensor(decks_A, dtype=dt)
        self.decks_B = torch.tensor(decks_B, dtype=dt)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.decks_A[idx], self.decks_B[idx], self.labels[idx]


# ==========================================
# データ一括処理関数（読み込み〜水増し〜ローダー化）
# ==========================================
def prepare_dataloaders(csv_path, encoder_type="multi-hot", test_size=0.2, batch_size=64):
    df_battles = pd.read_csv(csv_path)
    my_cols = [f'my_{i}' for i in range(8)]
    op_cols = [f'op_{i}' for i in range(8)]
    decks1 = df_battles[my_cols].values.tolist()
    decks2 = df_battles[op_cols].values.tolist()
    winners = (df_battles['result'] == 1).astype(np.float32)

    all_cards = np.unique(df_battles[my_cols + op_cols].values).tolist()

    if encoder_type == "multi-hot":
        mlb = MultiLabelBinarizer(classes=all_cards)
        vec_decks1 = np.asarray(mlb.fit_transform(decks1), dtype=np.float32)
        vec_decks2 = np.asarray(mlb.fit_transform(decks2), dtype=np.float32)
        vector_dim = len(all_cards)

    elif encoder_type == "raw_id":
        card_to_idx = {card: i for i, card in enumerate(all_cards)}
        vec_decks1 = np.array([[card_to_idx[c] for c in deck] for deck in decks1], dtype=np.int64)
        vec_decks2 = np.array([[card_to_idx[c] for c in deck] for deck in decks2], dtype=np.int64)
        vector_dim = len(all_cards) # カードの種類数（約122）
    else:
        raise ValueError("不明なエンコーダ")

    X_A = np.vstack((vec_decks1, vec_decks2))
    X_B = np.vstack((vec_decks2, vec_decks1))
    Y = np.concatenate((winners, 1.0 - winners))

    X_A_train, X_A_test, X_B_train, X_B_test, Y_train, Y_test = train_test_split(
        X_A, X_B, Y, test_size=test_size, random_state=42
    )

    train_dataset = BattleDataset(X_A_train, X_B_train, Y_train)
    test_dataset = BattleDataset(X_A_test, X_B_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, vector_dim, len(Y)

# ==========================================
# 予測モデル本体 (ベースモデル)
# ==========================================
class MatchupPredictor(nn.Module):
    def __init__(self, vector_dim):
        super().__init__()
        self.fc1 = nn.Linear(vector_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, deck_A, deck_B):
        x = torch.cat([deck_A, deck_B], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)
    
# ==========================================
# 新モデル: Cross-Attention Predictor
# ==========================================
class CrossAttentionPredictor(nn.Module):
    def __init__(self, num_cards, embed_dim=64,pretrained_embeddings=None):
        super().__init__()
        # カードをベクトル空間に配置する層
        self.embedding = nn.Embedding(num_cards, embed_dim)
        
        # 相手のデッキを見て、警戒すべきカードを見つけるAttention機構
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        
        if pretrained_embeddings is not None:
            # 渡された重みをコピーして上書きする
            self.embedding.weight.data.copy_(pretrained_embeddings)
            # ※ここで requires_grad = True のままにしておくことで、
            # 勝敗予測タスクに合わせて「シナジー」から「カウンター」へと重みが微調整（再学習）されます。
            print("✨ 事前学習済みのEmbedding（重み）をロードしました！")

        # 最終判定ネットワーク
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, deck_A, deck_B):
        # 1. 8枚のカードIDをベクトルに変換
        emb_A = self.embedding(deck_A) # (batch, 8, embed_dim)
        emb_B = self.embedding(deck_B)
        
        # 2. Cross Attention (AはBを警戒し、BはAを警戒する)
        attn_A2B, _ = self.cross_attn(query=emb_A, key=emb_B, value=emb_B) 
        attn_B2A, _ = self.cross_attn(query=emb_B, key=emb_A, value=emb_A) 
        
        # 3. 8枚の情報を「デッキ総合力」に圧縮
        pool_A = attn_A2B.mean(dim=1)
        pool_B = attn_B2A.mean(dim=1)
        
        # 4. ガッチャンコして勝敗判定
        x = torch.cat([pool_A, pool_B], dim=1)
        return self.fc(x)
    
# ==========================================
# モデル2: 平均値プーリング -> 重要度スコアを算出するAttentionプーリング版
# ==========================================
class AttentionPoolingPredictor(nn.Module):
    def __init__(self, num_cards, embed_dim=64, pretrained_embeddings=None):
        super().__init__()
        # 1. カードの埋め込み層
        self.embedding = nn.Embedding(num_cards, embed_dim)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            print("✨ 事前学習済みのEmbedding（重み）を AttentionPoolingPredictor にロードしました！")

        # 2. 相手デッキへの警戒（Cross-Attention）
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        
        # 🌟 3. アテンションプーリング層（重要度スコア算出用）
        self.attention_pooling = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(), # 非線形な関係を捉える
            nn.Linear(embed_dim // 2, 1)
        )

        # 4. 最終勝敗判定ネットワーク
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, deck_A, deck_B):
        # [Step 1] 8枚のカードIDをベクトルに変換
        emb_A = self.embedding(deck_A) # (batch, 8, embed_dim)
        emb_B = self.embedding(deck_B)
        
        # [Step 2] Cross Attention (相手のカードを踏まえた状態へ更新)
        attn_A2B, _ = self.cross_attn(query=emb_A, key=emb_B, value=emb_B) 
        attn_B2A, _ = self.cross_attn(query=emb_B, key=emb_A, value=emb_A) 
        
        # 🌟 [Step 3] アテンションプーリング 🌟
        # ① 各カードが「この試合においてどれくらい重要か」のスコアを計算 -> (batch, 8, 1)
        score_A = self.attention_pooling(attn_A2B)
        score_B = self.attention_pooling(attn_B2A)
        
        # ② Softmax関数で、8枚の重要度の合計が1.0（100%）になるよう変換（重み付け）
        weight_A = torch.softmax(score_A, dim=1)
        weight_B = torch.softmax(score_B, dim=1)
        
        # ③ 各カードのベクトルに重要度の重みを掛けて、足し合わせる（デッキ総合力へ圧縮） -> (batch, embed_dim)
        pool_A = torch.sum(attn_A2B * weight_A, dim=1)
        pool_B = torch.sum(attn_B2A * weight_B, dim=1)
        
        # [Step 4] 自分と相手のデッキ総合力を結合して判定
        x = torch.cat([pool_A, pool_B], dim=1)
        return self.fc(x)