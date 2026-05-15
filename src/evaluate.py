import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

from match_model import CrossAttentionPredictor, MatchupPredictor, prepare_dataloaders

if __name__ == "__main__":
    print("📊 モデルの詳細評価を開始します...")

    # ==========================================
    # 1. 評価設定（ここで3形態を切り替えます）
    # ==========================================
    TRAIN_CSV = 'data/train_matches.csv'
    TEST_CSV  = 'data/test_matches.csv'
    
    # 🌟 評価したいモデルの組み合わせを指定してください
    # MODEL_TYPE: "base" (MatchupPredictor) または "attention" (CrossAttentionPredictor)
    MODEL_TYPE = "base"      
    
    # ENCODER_TYPE: "raw_id" または "multi-hot" (※attentionの場合は必ずraw_idにしてください)
    ENCODER_TYPE = "raw_id"  
    
    # 評価するモデルのファイル名（trainで保存したものに合わせてください）
    MODEL_PATH = 'best_model.pth' 
    EMBED_DIM = 128
    
    OUTPUT_CSV = 'evaluation_details.csv' 
    OUTPUT_IMG = 'confusion_matrix.png'   

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==========================================
    # 2. データの準備
    # ==========================================
    print(f"📦 テストデータを読み込んでいます... (モデル: {MODEL_TYPE}, エンコード: {ENCODER_TYPE})")
    _, test_loader, vector_dim, _ = prepare_dataloaders(
        train_csv_path=TRAIN_CSV, 
        test_csv_path=TEST_CSV,
        encoder_type=ENCODER_TYPE
    )

    # ==========================================
    # 3. 学習済みモデルのロード
    # ==========================================
    print(f"🧠 保存されたモデル '{MODEL_PATH}' をロード中...")
    
    # 🌟 3形態の切り替え
    if MODEL_TYPE == "attention":
        # 形態1: CrossAttention (評価時は pretrained_embeddings=None でOK)
        model = CrossAttentionPredictor(
            num_cards=vector_dim, 
            embed_dim=EMBED_DIM
        ).to(device)
    else:
        # 形態2 & 3: MatchupPredictor (raw_id / multi-hot 自動対応)
        model = MatchupPredictor(
            num_cards=vector_dim, 
            embed_dim=EMBED_DIM,
            encoder_type=ENCODER_TYPE
        ).to(device)

    # 学習済みの重みを読み込んで上書き（これで事前学習の知識も復元されます）
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ ロード完了")

    # ==========================================
    # 4. 予測の実行
    # ==========================================
    all_preds = []
    all_labels = []
    all_probs = []
    deck_A_list = []
    deck_B_list = []

    print("🔍 テストデータで予測を実行中...")
    with torch.no_grad():
        for deck_A, deck_B, labels in test_loader:
            deck_A, deck_B = deck_A.to(device), deck_B.to(device)
            labels = labels.to(device)

            outputs = model(deck_A, deck_B)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend((probs >= 0.5).astype(int))
            all_labels.extend(labels.cpu().numpy())
            
            deck_A_list.extend(deck_A.cpu().numpy())
            deck_B_list.extend(deck_B.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    all_probs = np.array(all_probs).flatten()

    # ==========================================
    # 5. 評価結果の出力
    # ==========================================
    print("\n" + "="*40)
    print("🎯 【予測結果レポート】")
    print("="*40)
    cm = confusion_matrix(all_labels, all_preds)
    print(classification_report(all_labels, all_preds, target_names=["Lose", "Win"]))
    
    accuracy = (all_preds == all_labels).mean()
    print(f"✨ 最終正答率 (Accuracy): {accuracy * 100:.2f}%")
    print(f"   - 実際の勝ち試合を当てた数: {cm[1, 1]} 🙆‍♂️")
    print(f"   - 実際の負け試合を当てた数: {cm[0, 0]} 🙅‍♂️")
    print(f"   - 勝ちと予測して負けた数 (False Positive): {cm[0, 1]} 🤦‍♂️")
    print(f"   - 負けと予測して勝った数 (False Negative): {cm[1, 0]} 🙆‍♂️")

    # --- 6. 混同行列の画像保存 ---
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Predict: Lose", "Predict: Win"], 
                yticklabels=["Actual: Lose", "Actual: Win"])
    plt.title('Confusion Matrix: Clash Royale Matchup')
    plt.ylabel('Actual Result')
    plt.xlabel('Predicted Result')
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f"\n🖼️ 混同行列の画像を保存しました: {OUTPUT_IMG}")

    # --- 7. 詳細な予測結果のCSV書き出し ---
    df_results = pd.DataFrame({
        'Actual_Result': all_labels,
        'Predicted_Label': all_preds,
        'Win_Probability': np.round(all_probs, 4),
        'Is_Correct': all_labels == all_preds,
        'Deck_A_Predicting': [str(d.tolist()) for d in deck_A_list],
        'Deck_B_Opponent': [str(d.tolist()) for d in deck_B_list]
    })

    df_results['Confidence'] = np.abs(df_results['Win_Probability'] - 0.5) * 2
    df_results = df_results.sort_values(by='Confidence', ascending=False)
    
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"📄 詳細な予測結果をCSVに保存しました: {OUTPUT_CSV}")