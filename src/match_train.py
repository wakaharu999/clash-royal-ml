import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.metrics import accuracy_score, roc_auc_score
from match_model import MatchupPredictor, CrossAttentionPredictor, prepare_dataloaders

if __name__ == "__main__":
    print(" 勝敗予測モデルの学習を開始します")

    # --- 1. 設定 ---
    TRAIN_CSV = 'data/train_matches.csv'
    TEST_CSV  = 'data/test_matches.csv'
    
    #ENCODER_TYPE: "multi-hot,  "raw_id" のいずれかを選択
    ENCODER_TYPE = "raw_id"
    EPOCHS = 100      
    LR = 0.001
    PATIENCE = 5       
    SAVE_PATH = 'best_model.pth' 

    # --- 2. データの準備 ---
    print(f" データを読み込み、前処理を行っています (モード: {ENCODER_TYPE})...")
    

    train_loader, test_loader, vector_dim, total_samples = prepare_dataloaders(
        train_csv_path=TRAIN_CSV, 
        test_csv_path=TEST_CSV,
        encoder_type=ENCODER_TYPE
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    EMBED_DIM = 128
    pretrained_path = 'models/pretrained_model.pth'

    print(f" 事前学習ファイル {pretrained_path} を読み込んでいます...")
    try:
        state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
        target_key = 'embedding.weight' 
        
        if target_key in state_dict:
            raw_weights = state_dict[target_key]
            pretrained_weights = raw_weights[:vector_dim, :].detach().clone()
            
            print(f"✨ 成功: '{target_key}' から {pretrained_weights.shape[0]}枚分のカードの記憶を抽出しました！")
        else:
            print(f"⚠️ エラー: '{target_key}' が見つかりません。ランダム初期化で進めます。")
            pretrained_weights = None
            
    except FileNotFoundError:
        print(f"⚠️ エラー: ファイル '{pretrained_path}' が見つかりません。ランダム初期化で進めます。")
        pretrained_weights = None

    model = CrossAttentionPredictor(num_cards=vector_dim, embed_dim= EMBED_DIM, pretrained_embeddings=pretrained_weights).to(device)
    #model = MatchupPredictor(vector_dim=vector_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    # --- 3. 学習ループ (Early Stopping搭載) ---
    print(f" 学習スタート (最大 {EPOCHS} epochs)...")
    
    best_val_acc = 0.0 # 最高正答率を記録する変数
    patience_counter = 0 # 改善が見られないエポック数をカウントする変数
    
    for epoch in range(1, EPOCHS + 1):
        # -------------------------
        # [A] 訓練フェーズ (Train)
        # -------------------------
        model.train()
        train_loss = 0
        for dA, dB, labels in train_loader:
            dA, dB, labels = dA.to(device), dB.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(dA, dB)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)

        # -------------------------
        # [B] 評価フェーズ (Validation)
        # -------------------------
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for dA, dB, labels in test_loader:
                dA, dB, labels = dA.to(device), dB.to(device), labels.to(device)
                outputs = model(dA, dB)
                
                # TestデータでのLossも計算
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # 正答率計算のための準備
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(test_loader)
        
        # Testデータでの正答率を計算
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        binary_preds = (all_preds >= 0.5).astype(int)
        val_acc = accuracy_score(all_labels, binary_preds)

        # ログ出力 (TrainとTestを横並びで比較！)
        print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%", end="")

        # -------------------------
        # [C] Early Stopping の判定
        # -------------------------
        if val_acc > best_val_acc:
            # 最高記録を更新した場合！
            best_val_acc = val_acc
            patience_counter = 0 # カウンターをリセット
            # 現在のモデルの状態(重み)をファイルに保存
            torch.save(model.state_dict(), SAVE_PATH)
            print(" 記録更新！モデルを保存しました")
        else:
            # 更新できなかった場合
            patience_counter += 1
            print(f" 更新なし ({patience_counter}/{PATIENCE})")
            
            if patience_counter >= PATIENCE:
                print(f"\n {PATIENCE}エポック連続で改善が見られなかったため、学習を早期終了(Early Stopping)します！")
                break

    # --- 4. 最終結果の表示 ---
    print("\n" + "="*40)
    print(f"🏆 学習完了！")
    print(f"✨ ベストなテスト正答率: {best_val_acc*100:.2f}%")
    print(f"💾 ベストモデルは '{SAVE_PATH}' に保存されています")
    print("="*40)