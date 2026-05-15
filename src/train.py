import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import json
from sklearn.metrics import accuracy_score

from model import MagNetEncoder, SimpleSumPredictor, MagNetCrossAttentionPredictor

# --- 1. 設定 ---
GRAPH_PATH = "models/directed_graph.pt"
CARDS_JSON_PATH = "data/cards.json"
TRAIN_CSV = 'data/train_matches.csv'
TEST_CSV  = 'data/test_matches.csv'
SAVE_PATH = 'models/best_model.pth' 

HIDDEN_DIM = 128
BATCH_SIZE = 256
EPOCHS = 100      
LR = 5e-5  # MagNetは学習率が低め(1e-4)の方が安定します
PATIENCE = 10

def load_card_mapping():
    """cards.jsonからIDマッピングを作成（グラフのノード番号と合わせるため必須）"""
    with open(CARDS_JSON_PATH, 'r', encoding='utf-8') as f:
        cards_data = json.load(f)
    card_ids = sorted([int(card_id) for card_id in cards_data.keys()])
    id_to_idx = {cid: i for i, cid in enumerate(card_ids)}
    return id_to_idx, len(card_ids)

def prepare_dataset(csv_path, id_to_idx, is_train=True):
    """CSVを読み込み、PyTorchのDatasetに変換する"""
    df = pd.read_csv(csv_path)
    my_decks, op_decks, labels = [], [], []
    
    for _, row in df.iterrows():
        my_deck = [int(row[f'my_{i}']) for i in range(8)]
        op_deck = [int(row[f'op_{i}']) for i in range(8)]
        
        try:
            m_idx = [id_to_idx[c] for c in my_deck]
            o_idx = [id_to_idx[c] for c in op_deck]
        except KeyError:
            continue
            
        result_val = float(row['result'])
        if result_val == 0.0:
            continue # 引き分けなどのノイズは除外
            
        is_my_win = 1.0 if result_val > 0 else 0.0
        
        # 訓練時のみ、50%の確率で自分と相手をひっくり返してデータ拡張
        if is_train and torch.rand(1).item() > 0.5:
            my_decks.append(o_idx)
            op_decks.append(m_idx)
            labels.append(1.0 - is_my_win)
        else:
            my_decks.append(m_idx)
            op_decks.append(o_idx)
            labels.append(is_my_win)

    return TensorDataset(
        torch.tensor(my_decks, dtype=torch.long),
        torch.tensor(op_decks, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32)
    )

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 MagNet 勝敗予測モデルの学習を開始します (Device: {device})")

    # --- 2. データの準備 ---
    print("📦 データを読み込み、前処理を行っています...")
    id_to_idx, num_cards = load_card_mapping()
    
    train_dataset = prepare_dataset(TRAIN_CSV, id_to_idx, is_train=True)
    test_dataset = prepare_dataset(TEST_CSV, id_to_idx, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 有向グラフのロード (PyTorch 2.6のセキュリティ対策 weights_only=False)
    graph_data = torch.load(GRAPH_PATH, weights_only=False).to(device)
    edge_index = graph_data.edge_index
    edge_weight = graph_data.edge_attr

    # --- 3. モデルの初期化 ---
    encoder = MagNetEncoder(num_cards=num_cards, hidden_dim=HIDDEN_DIM).to(device)
    
    # 🌟 事前学習済みの MagNet 重みをロードする
    PRETRAINED_PATH = "models/pretrained_magnet.pth"
    try:
        encoder.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device, weights_only=True))
        print("✨ 事前学習済みの MagNetEncoder (相性辞書) をロードしました！")
        
        # 🌟 Encoderの重みを完全に固定（フリーズ）する
        for param in encoder.parameters():
            param.requires_grad = False
        print("🔒 Encoderの重みを固定しました。Predictorのみを学習させます。")
            
    except FileNotFoundError:
        print(f"⚠️ 警告: '{PRETRAINED_PATH}' が見つかりません。ランダム初期化で進めます。")

    # Predictor は最強の Cross-Attention モデルを使用
    #predictor = SimpleSumPredictor(hidden_dim=HIDDEN_DIM).to(device)
    predictor = MagNetCrossAttentionPredictor(hidden_dim=HIDDEN_DIM).to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    # 🌟 Encoderは固定したので、Optimizerには Predictor のパラメータだけを渡す
    optimizer = optim.Adam(predictor.parameters(), lr=LR)
    # --- 4. 学習ループ (Early Stopping搭載) ---
    print(f"🔥 学習スタート (最大 {EPOCHS} epochs)...")
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        # [A] Train Phase
        encoder.train()
        predictor.train()
        train_loss = 0
        
        for dA, dB, labels in train_loader:
            dA, dB, labels = dA.to(device), dB.to(device), labels.to(device)
            optimizer.zero_grad()
            
            x_real, x_imag = encoder(edge_index, edge_weight)
            logits = predictor(x_real, x_imag, dA, dB)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)

        # [B] Validation Phase
        encoder.eval()
        predictor.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for dA, dB, labels in test_loader:
                dA, dB, labels = dA.to(device), dB.to(device), labels.to(device)
                
                x_real, x_imag = encoder(edge_index, edge_weight)
                logits = predictor(x_real, x_imag, dA, dB)
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(test_loader)
        
        # 正答率の計算
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        binary_preds = (all_preds >= 0.5).astype(int)
        val_acc = accuracy_score(all_labels, binary_preds)

        print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%", end="")

        # [C] Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # EncoderとPredictor両方の重みを保存
            torch.save({
                'encoder': encoder.state_dict(),
                'predictor': predictor.state_dict()
            }, SAVE_PATH)
            print(" 🌟 記録更新！モデルを保存しました")
        else:
            patience_counter += 1
            print(f" ⚠️ 更新なし ({patience_counter}/{PATIENCE})")
            
            if patience_counter >= PATIENCE:
                print(f"\n🛑 {PATIENCE}エポック連続で改善が見られなかったため、早期終了します！")
                break

    print("\n" + "="*40)
    print(f"🏆 学習完了！ ベストテスト正答率: {best_val_acc*100:.2f}%")
    print("="*40)