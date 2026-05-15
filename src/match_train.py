import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import ast
import json

# 作成したモデルをインポート
from match_model import MagNetEncoder, CrossAttentionPredictor

# --- 設定パラメータ ---
GRAPH_PATH = "models/directed_graph.pt"
MATCHES_CSV_PATH = "data/matches.csv"
CARDS_JSON_PATH = "data/cards.json"

HIDDEN_DIM = 128
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-4

def load_card_mapping():
    with open(CARDS_JSON_PATH, 'r', encoding='utf-8') as f:
        cards_data = json.load(f)
    
    # JSONのキー（"26000020"などの文字列）を数値(int)に変換してソート
    card_ids = sorted([int(card_id) for card_id in cards_data.keys()])
    
    id_to_idx = {cid: i for i, cid in enumerate(card_ids)}
    return id_to_idx, len(card_ids)

def prepare_dataset(id_to_idx):
    """対戦データをPyTorchのDatasetに変換"""
    df = pd.read_csv(MATCHES_CSV_PATH)
    
    my_decks = []
    op_decks = []
    labels = []
    
    for _, row in df.iterrows():
        my_deck = [int(row[f'my_{i}']) for i in range(8)]
        op_deck = [int(row[f'op_{i}']) for i in range(8)]
        
        try:
            m_idx = [id_to_idx[c] for c in my_deck]
            o_idx = [id_to_idx[c] for c in op_deck]
        except KeyError:
            continue
            
        result_val = float(row['result'])
        
        # 引き分け(0)などのノイズは除外する（勝敗が明確なデータのみ学習させる）
        if result_val == 0.0:
            continue
            
        my_decks.append(m_idx)
        op_decks.append(o_idx)
        
        # 勝ち(1) なら 1.0、負け(-1) なら 0.0 をラベルとして保存
        label = 1.0 if result_val > 0 else 0.0
        labels.append(label)

    my_decks_tensor = torch.tensor(my_decks, dtype=torch.long)
    op_decks_tensor = torch.tensor(op_decks, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    return TensorDataset(my_decks_tensor, op_decks_tensor, labels_tensor)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. データとグラフの読み込み
    id_to_idx, num_cards = load_card_mapping()
    dataset = prepare_dataset(id_to_idx)
    
    # 学習用と検証用に分割 (8:2)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # MagNet用に構築した有向グラフをロード
    graph_data = torch.load(GRAPH_PATH, weights_only=False).to(device)
    edge_index = graph_data.edge_index
    edge_weight = graph_data.edge_attr

    # 2. モデルの初期化
    encoder = MagNetEncoder(num_cards=num_cards, hidden_dim=HIDDEN_DIM).to(device)
    predictor = CrossAttentionPredictor(hidden_dim=HIDDEN_DIM).to(device)

    # BCEWithLogitsLoss は内部でSigmoidをかけるので、Predictorの出力(Logit)をそのまま渡せる
    criterion = nn.BCEWithLogitsLoss()
    
    # 2つのモデルのパラメータを同時に最適化 (End-to-End学習)
    optimizer = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=LEARNING_RATE)

    # 3. 学習ループ
    print("--- 学習開始 ---")
    for epoch in range(EPOCHS):
        encoder.train()
        predictor.train()
        total_loss = 0
        
        for my_decks, op_decks, labels in train_loader:
            my_decks, op_decks, labels = my_decks.to(device), op_decks.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # (A) Encoderで全カードの複素数ベクトル(実部・虚部)を生成
            x_real, x_imag = encoder(edge_index, edge_weight)
            
            # (B) Predictorで特定のバッチのデッキを抽出し、勝敗判定
            logits = predictor(x_real, x_imag, my_decks, op_decks)
            
            # 損失計算と逆伝播
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # 4. 検証ループ (Validation)
        encoder.eval()
        predictor.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for my_decks, op_decks, labels in val_loader:
                my_decks, op_decks, labels = my_decks.to(device), op_decks.to(device), labels.to(device)
                
                x_real, x_imag = encoder(edge_index, edge_weight)
                logits = predictor(x_real, x_imag, my_decks, op_decks)
                
                # 0より大きければ my_deck の勝ちと予測
                preds = (logits > 0).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

if __name__ == "__main__":
    train()