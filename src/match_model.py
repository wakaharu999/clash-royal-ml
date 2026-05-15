import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer


# ==========================================
# データセットの定義 
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
# データ一括処理関数
# ==========================================
def prepare_dataloaders(train_csv_path, test_csv_path, encoder_type="multi-hot", batch_size=64):
    # 1. 訓練用とテスト用のCSVをそれぞれ読み込む
    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)
    
    my_cols = [f'my_{i}' for i in range(8)]
    op_cols = [f'op_{i}' for i in range(8)]

    # 2. 全カードのリストを作成（TrainとTestの全カードを合体させて語彙を作る）
    # これにより、テストデータにだけ存在するマイナーカードによるエラーを防ぎます
    all_cards = np.unique(np.concatenate((
        df_train[my_cols + op_cols].values.flatten(),
        df_test[my_cols + op_cols].values.flatten()
    ))).tolist()

    # 3. DataFrameをTensor用の配列に変換する内部関数
    def process_dataframe(df):
        decks1 = df[my_cols].values.tolist()
        decks2 = df[op_cols].values.tolist()
        winners = (df['result'] == 1).astype(np.float32)

        if encoder_type == "multi-hot":
            mlb = MultiLabelBinarizer(classes=all_cards)
            vec_decks1 = np.asarray(mlb.fit_transform(decks1), dtype=np.float32)
            vec_decks2 = np.asarray(mlb.fit_transform(decks2), dtype=np.float32)
            
        elif encoder_type == "raw_id":
            card_to_idx = {card: i for i, card in enumerate(all_cards)}
            vec_decks1 = np.array([[card_to_idx[c] for c in deck] for deck in decks1], dtype=np.int64)
            vec_decks2 = np.array([[card_to_idx[c] for c in deck] for deck in decks2], dtype=np.int64)
            
        else:
            raise ValueError("不明なエンコーダ")

        # データ水増し（A vs B と B vs A の両方を作成して2倍にする）
        X_A = np.vstack((vec_decks1, vec_decks2))
        X_B = np.vstack((vec_decks2, vec_decks1))
        Y = np.concatenate((winners, 1.0 - winners))
        
        return X_A, X_B, Y

    # 4. 変換処理の実行
    X_A_train, X_B_train, Y_train = process_dataframe(df_train)
    X_A_test, X_B_test, Y_test = process_dataframe(df_test)

    # 5. データローダー化
    train_dataset = BattleDataset(X_A_train, X_B_train, Y_train)
    test_dataset = BattleDataset(X_A_test, X_B_test, Y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    vector_dim = len(all_cards)
    
    return train_loader, test_loader, vector_dim, len(Y_train)

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
