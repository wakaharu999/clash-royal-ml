import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, accuracy_score
import os
import json

# match_model.py から MagNetEncoder のみをインポート
from model import MagNetEncoder

# --- 1. 設定 ---
GRAPH_PATH = "models/directed_graph.pt"
CARDS_JSON_PATH = "data/cards.json"
SAVE_DIR = "models"
SAVE_PATH = os.path.join(SAVE_DIR, "pretrained_magnet.pth")

HIDDEN_DIM = 128
EPOCHS = 200
LR = 0.001

# --- 2. リンク予測用の一時的なモデル ---
class LinkPredictor(nn.Module):
    """
    2つのカードのベクトルを受け取り、その間に「有利・不利」の矢印(エッジ)が存在するかを予測する
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # u(実・虚) と v(実・虚) を結合するため次元数は4倍
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_real, x_imag, edge_index):
        # 矢印の根本(u) と 先端(v)
        row, col = edge_index
        
        u_real, u_imag = x_real[row], x_imag[row]
        v_real, v_imag = x_real[col], x_imag[col]
        
        # 2つのカードの特徴量をガッチャンコ
        features = torch.cat([u_real, u_imag, v_real, v_imag], dim=-1)
        return self.mlp(features).squeeze(-1)

# --- 3. 補助関数 ---
def load_card_mapping():
    with open(CARDS_JSON_PATH, 'r', encoding='utf-8') as f:
        cards_data = json.load(f)
    card_ids = sorted([int(card_id) for card_id in cards_data.keys()])
    return len(card_ids)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 MagNet 事前学習 (Link Prediction) を開始します - Device: {device}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- データ準備 ---
    num_cards = load_card_mapping()
    graph_data = torch.load(GRAPH_PATH, weights_only=False).to(device)
    edge_index = graph_data.edge_index
    edge_weight = graph_data.edge_attr

    # --- モデル初期化 ---
    encoder = MagNetEncoder(num_cards=num_cards, hidden_dim=HIDDEN_DIM).to(device)
    predictor = LinkPredictor(hidden_dim=HIDDEN_DIM).to(device)
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0

    print(f"🔥 学習スタート (最大 {EPOCHS} epochs)...")
    
    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        predictor.train()
        optimizer.zero_grad()

        # 1. グラフ構造から各カードの複素数ベクトルを生成
        x_real, x_imag = encoder(edge_index, edge_weight)

        # 2. 正例エッジ（実際に存在する有利不利の関係）の予測
        pos_logits = predictor(x_real, x_imag, edge_index)
        pos_labels = torch.ones_like(pos_logits)

        # 3. 負例エッジ（相性関係がないランダムなペア）を生成して予測
        neg_edge_index = negative_sampling(
            edge_index=edge_index, 
            num_nodes=num_cards,
            num_neg_samples=edge_index.size(1) # 正例と同じ数だけ生成
        )
        neg_logits = predictor(x_real, x_imag, neg_edge_index)
        neg_labels = torch.zeros_like(neg_logits)

        # 4. ロスの計算（正例は1、負例は0になるように学習）
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([pos_labels, neg_labels])
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # 5. 評価指標 (AUCと正答率) の計算
        with torch.no_grad():
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            labels_np = labels.cpu().numpy()
            
            acc = accuracy_score(labels_np, preds)
            auc = roc_auc_score(labels_np, probs)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Acc: {acc*100:.2f}% | AUC: {auc:.4f}")

        # AUCが更新されたらEncoderのみを保存！
        if auc > best_auc:
            best_auc = auc
            # 注意: Predictorは捨てて、Encoder(相性辞書)だけを保存します
            torch.save(encoder.state_dict(), SAVE_PATH)

    print("\n" + "="*40)
    print(f"🏆 事前学習完了！ ベストAUC: {best_auc:.4f}")
    print(f"💾 学習済みMagNetEncoderは '{SAVE_PATH}' に保存されました！")
    print("="*40)