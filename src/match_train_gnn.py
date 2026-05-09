import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.metrics import accuracy_score

# 先ほど作成・修正したモデルたちをインポート
from match_model import GlobalRGCNEncoder, GNNPredictor, prepare_dataloaders

if __name__ == "__main__":
    print("🚀 RGCN + AttentionPooling モデルのEnd-to-End学習を開始します...")

    # --- 1. 設定 ---
    CSV_PATH = 'data/matches.csv'
    GRAPH_PATH = 'models/global_graph.pt'
    EPOCHS = 100      
    LR = 0.001
    PATIENCE = 5       # Early Stoppingの我慢回数
    SAVE_PATH = 'best_rgcn_model.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用デバイス: {device}")

    # ==========================================
    # 🌟 2. グラフデータ（環境地図）の読み込み
    # ==========================================
    print(f"📦 グラフデータ '{GRAPH_PATH}' を読み込んでいます...")
    try:
        # weights_only=False は、テンソル以外の辞書データも読み込むために必要です
        graph_data = torch.load(GRAPH_PATH, map_location=device, weights_only=False)
        
        # GNNに渡すための3つのテンソルを準備
        edge_index = graph_data['edge_index'].to(device)
        edge_type = graph_data['edge_type'].to(device)
        node_roles = graph_data['node_roles'].to(device)
        
        num_cards = len(graph_data['card_to_idx'])
        print(f"✨ 成功: {num_cards}枚のカードと {edge_index.shape[1]}本のエッジをロードしました。")
        
    except FileNotFoundError:
        print(f"❌ エラー: '{GRAPH_PATH}' が見つかりません。先に build_graph_data.py を実行してください。")
        exit()

    # --- 3. 対戦データの準備 ---
    print("📦 対戦履歴を読み込み、データローダーを作成中...")
    train_loader, test_loader, vector_dim, total_samples = prepare_dataloaders(
        csv_path=CSV_PATH, 
        encoder_type="raw_id"
    )
    
    # 万が一のバグを防ぐためのチェック
    assert num_cards == vector_dim, f"⚠️ グラフのカード数({num_cards})と対戦データのカード数({vector_dim})が一致しません！"

    # ==========================================
    # 🌟 4. モデルの構築 (バケツリレー構造)
    # ==========================================
    print("🧠 モデルを構築中...")
    
    # [A] RGCNエンコーダ (全カードの環境ベクトルを生成するモデル)
    rgcn_encoder = GlobalRGCNEncoder(
        num_cards=num_cards,
        num_roles=4,        # Spell, Building, Support, Win_Condition
        role_dim=16,
        embed_dim=128       # 最終的に128次元のリッチなベクトルにする
    ).to(device)

    # [B] 勝敗予測モデル (RGCNエンコーダを内包する)
    model = GNNPredictor(
        rgcn_encoder=rgcn_encoder,
        embed_dim=128
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    # --- 5. 学習ループ (Early Stopping搭載) ---
    print(f"\n🔥 学習スタート (最大 {EPOCHS} epochs)...")
    
    best_val_acc = 0.0 
    patience_counter = 0 
    
    for epoch in range(1, EPOCHS + 1):
        # -------------------------
        # [Train] 訓練フェーズ
        # -------------------------
        model.train()
        train_loss = 0
        for dA, dB, labels in train_loader:
            dA, dB, labels = dA.to(device), dB.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # 🌟 ポイント: 対戦カードと一緒に、グラフ情報も毎回渡す！
            # 内部で「グラフ更新 -> ベクトル抽出 -> 勝敗予測」が走る
            outputs = model(dA, dB, edge_index, edge_type, node_roles)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)

        # -------------------------
        # [Eval] 評価フェーズ
        # -------------------------
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for dA, dB, labels in test_loader:
                dA, dB, labels = dA.to(device), dB.to(device), labels.to(device)
                outputs = model(dA, dB, edge_index, edge_type, node_roles)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(test_loader)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        binary_preds = (all_preds >= 0.5).astype(int)
        val_acc = accuracy_score(all_labels, binary_preds)

        # ログ出力
        print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%", end="")

        # -------------------------
        # Early Stopping 判定
        # -------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0 
            torch.save(model.state_dict(), SAVE_PATH)
            print(" 🌟 記録更新！モデルを保存しました")
        else:
            patience_counter += 1
            print(f" ⚠️ 更新なし ({patience_counter}/{PATIENCE})")
            
            if patience_counter >= PATIENCE:
                print(f"\n🛑 {PATIENCE}エポック連続で改善が見られなかったため、学習を早期終了(Early Stopping)します！")
                break

    # --- 6. 最終結果 ---
    print("\n" + "="*40)
    print(f"🏆 学習完了！")
    print(f"✨ ベストなテスト正答率: {best_val_acc*100:.2f}%")
    print(f"💾 ベストモデルは '{SAVE_PATH}' に保存されています")
    print("="*40)