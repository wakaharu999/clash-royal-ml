# matchup_model.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric.nn import RGCNConv
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
# 追加：グローバルグラフからカードの「環境理解ベクトル」を生成するGNN
# ==========================================
class GlobalRGCNEncoder(nn.Module):
    def __init__(self, num_cards, num_roles=4, role_dim=16, embed_dim=128):
        super().__init__()
        # 1. IDベースの埋め込み (128 - 16 = 112次元)
        self.id_emb = nn.Embedding(num_cards, embed_dim - role_dim)
        
        # 2. 役割(Role)ベースの埋め込み (16次元)
        self.role_emb = nn.Embedding(num_roles, role_dim)
        
        # 3. マルチリレーショナルGCN層 (3つの関係性を処理)
        # 0:シナジー, 1:カウンター, 2:重量競合
        self.conv1 = RGCNConv(embed_dim, embed_dim, num_relations=3)
        self.conv2 = RGCNConv(embed_dim, embed_dim, num_relations=3)
        
        self.relu = nn.ReLU()

    def forward(self, edge_index, edge_type, node_roles):
        # [Step 1] IDベクトルと役割ベクトルをガッチャンコして128次元の初期値を作る
        x_id = self.id_emb.weight # 全カードのIDベクトル [num_cards, 112]
        x_role = self.role_emb(node_roles) # 全カードの役割ベクトル [num_cards, 16]
        x = torch.cat([x_id, x_role], dim=-1) # -> [num_cards, 128]
        
        # [Step 2] グラフ上での情報伝達 (環境のメタを学習)
        x = self.conv1(x, edge_index, edge_type)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        
        # 全カードの「最新の環境理解ベクトル」を返す
        return x
    
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
    
class GNNPredictor(nn.Module):
    # 🌟 pretrained_embeddings の代わりに、rgcn_encoder を受け取るように変更
    def __init__(self, rgcn_encoder, embed_dim=128):
        super().__init__()
        # GNNエンコーダを内部に保持
        self.rgcn = rgcn_encoder
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        
        self.attention_pooling = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    # 🌟 グラフデータもforwardの引数に追加
    def forward(self, deck_A, deck_B, edge_index, edge_type, node_roles):
        # [Step 1] まずGNNを走らせて、全120枚の「最新の環境ベクトル」を生成する！
        global_card_embeddings = self.rgcn(edge_index, edge_type, node_roles)
        
        # [Step 2] その辞書の中から、いま対戦している8枚のベクトルだけを引っ張ってくる
        emb_A = global_card_embeddings[deck_A] # (batch, 8, embed_dim)
        emb_B = global_card_embeddings[deck_B]
        
        # --- これ以降 (Cross Attention と Pooling) は前回と全く同じ！ ---
        attn_A2B, _ = self.cross_attn(query=emb_A, key=emb_B, value=emb_B) 
        attn_B2A, _ = self.cross_attn(query=emb_B, key=emb_A, value=emb_A) 
        
        score_A = self.attention_pooling(attn_A2B)
        score_B = self.attention_pooling(attn_B2A)
        weight_A = torch.softmax(score_A, dim=1)
        weight_B = torch.softmax(score_B, dim=1)
        
        pool_A = torch.sum(attn_A2B * weight_A, dim=1)
        pool_B = torch.sum(attn_B2A * weight_B, dim=1)
        
        x = torch.cat([pool_A, pool_B], dim=1)
        return self.fc(x)