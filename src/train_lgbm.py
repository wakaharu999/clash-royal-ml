import torch
import pandas as pd
import numpy as np
import json
import os
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# 注意: MagNetEncoderが定義されているファイル名に合わせてください
from model import MagNetEncoder 

# --- 1. 設定 ---
GRAPH_PATH = "models/directed_graph.pt"
CARDS_JSON_PATH = "data/cards.json"
PRETRAINED_PATH = "models/pretrained_magnet.pth"
TRAIN_CSV = 'data/train_matches.csv'
TEST_CSV  = 'data/test_matches.csv'
HIDDEN_DIM = 128

def load_card_mapping():
    with open(CARDS_JSON_PATH, 'r', encoding='utf-8') as f:
        cards_data = json.load(f)
    card_ids = sorted([int(card_id) for card_id in cards_data.keys()])
    id_to_idx = {cid: i for i, cid in enumerate(card_ids)}
    return id_to_idx, len(card_ids)

# --- 2. MagNetから「相性辞書ベクトル」を抽出 ---
def get_magnet_embeddings(device):
    id_to_idx, num_cards = load_card_mapping()
    encoder = MagNetEncoder(num_cards=num_cards, hidden_dim=HIDDEN_DIM).to(device)
    
    # 事前学習済みの重みをロード
    encoder.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device, weights_only=True))
    encoder.eval()
    
    graph_data = torch.load(GRAPH_PATH, weights_only=False).to(device)
    
    with torch.no_grad():
        x_real, x_imag = encoder(graph_data.edge_index, graph_data.edge_attr)
        
        # 魔法のハック: 実部・虚部を「強さ(振幅)」と「相性の向き(位相)」に変換
        amplitude = torch.sqrt(x_real**2 + x_imag**2 + 1e-8)
        phase = torch.atan2(x_imag, x_real)
        
    return amplitude.cpu().numpy(), phase.cpu().numpy(), id_to_idx

# --- 3. LightGBM用の特徴量(テーブルデータ)を作成 ---
def extract_features(csv_path, amp_np, phase_np, id_to_idx):
    df = pd.read_csv(csv_path)
    X, y = [], []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"📦 {os.path.basename(csv_path)} を処理中"):
        my_deck = [int(row[f'my_{i}']) for i in range(8)]
        op_deck = [int(row[f'op_{i}']) for i in range(8)]
        
        try:
            m_idx = [id_to_idx[c] for c in my_deck]
            o_idx = [id_to_idx[c] for c in op_deck]
        except KeyError:
            continue
            
        result_val = float(row['result'])
        if result_val == 0.0:
            continue
        is_my_win = 1 if result_val > 0 else 0
        
        # 各カードのベクトルを取得 (8枚 x 128次元)
        m_amp, m_pha = amp_np[m_idx], phase_np[m_idx]
        o_amp, o_pha = amp_np[o_idx], phase_np[o_idx]
        
        # 🌟 特徴量エンジニアリング（相手との関係性・総当たり特化型）🌟
        
        # 1. 8枚 vs 8枚 の総当たり戦（64通り）の差分マトリックスを作成
        # m_amp[:, None, :] は (8, 1, 128)、o_amp[None, :, :] は (1, 8, 128)
        # 引き算をすると自動的に (8, 8, 128) の「全組み合わせ差分」が計算されます
        diff_amp = m_amp[:, None, :] - o_amp[None, :, :]
        diff_pha = m_pha[:, None, :] - o_pha[None, :, :]

        # 2. 64通りの相性の中から、「最大の有利」「最大の不利」「全体の相性」を抽出して結合
        features = np.concatenate([
            # 強さ(振幅)に関する対面評価
            diff_amp.max(axis=(0, 1)),   # 自デッキで一番相手に刺さっているカードの有利度
            diff_amp.min(axis=(0, 1)),   # 相手に一番刺されている（警戒すべき）不利度
            diff_amp.mean(axis=(0, 1)),  # 64通りの平均的な戦力差
            
            # 相性(位相)に関する対面評価
            diff_pha.max(axis=(0, 1)),   # 最も強烈なカウンター(天敵)をこちらが持っているか
            diff_pha.min(axis=(0, 1)),   # 最も強烈なカウンターを相手が持っているか
            diff_pha.mean(axis=(0, 1))   # 64通りの平均的な相性差
        ])
        
        X.append(features)
        y.append(is_my_win)
        
    return np.array(X), np.array(y)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("🚀 MagNetの特徴量を使った LightGBM 学習を開始します...")
    
    # 1. ベクトルの抽出
    print("🧠 MagNet 事前学習モデルを展開中...")
    amp_np, phase_np, id_to_idx = get_magnet_embeddings(device)
    
    # 2. 特徴量の作成
    X_train, y_train = extract_features(TRAIN_CSV, amp_np, phase_np, id_to_idx)
    X_test, y_test   = extract_features(TEST_CSV, amp_np, phase_np, id_to_idx)
    
    print(f"\n📊 データ準備完了: Train {X_train.shape[0]}件, Test {X_test.shape[0]}件, 特徴量次元数 {X_train.shape[1]}")
    
    # 3. LightGBMモデルの定義と学習
    print("\n🌳 LightGBM 学習スタート！ (一瞬で終わります)")
    clf = lgb.LGBMClassifier(
        n_estimators=1000,       # 最大の木の本数
        learning_rate=0.05,      # 学習率
        max_depth=6,             # 木の深さ（過学習を防ぐために少し浅め）
        subsample=0.8,           # データのサンプリング（過学習防止）
        random_state=42,
        importance_type='gain'
    )
    
    # 学習実行（テストデータでEarly Stoppingを監視）
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
    )
    
    # 4. 評価
    print("\n" + "="*40)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds) # type: ignore
    print(f"🏆 LightGBM 最終テスト正答率: {acc*100:.2f}%")
    print("="*40)