import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 注意: モデルが定義されているファイル名を正確に指定してください ---
from match_model import CrossAttentionPredictor, prepare_dataloaders

if __name__ == "__main__":
    print("📊 モデルの詳細評価を開始します...")

    # --- 1. 設定 ---
    TRAIN_CSV = 'data/train_matches.csv' # 語彙を揃えるためにTrainも読み込みます
    TEST_CSV  = 'data/test_matches.csv'
    ENCODER_TYPE = "raw_id"
    MODEL_PATH = 'models/attention_model.pth'
    EMBED_DIM = 128
    
    OUTPUT_CSV = 'evaluation_details.csv' # 予測結果を書き出すCSVファイル
    OUTPUT_IMG = 'confusion_matrix.png'   # 混同行列の画像ファイル

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 2. データの準備 ---
    print("📦 テストデータを読み込んでいます...")
    _, test_loader, vector_dim, _ = prepare_dataloaders(
        train_csv_path=TRAIN_CSV, 
        test_csv_path=TEST_CSV,
        encoder_type=ENCODER_TYPE
    )

    # --- 3. 学習済みモデルのロード ---
    print(f"🧠 保存されたモデル '{MODEL_PATH}' をロード中...")
    model = CrossAttentionPredictor(num_cards=vector_dim, embed_dim=EMBED_DIM).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"⚠️ エラー: '{MODEL_PATH}' が見つかりません。先に学習を実行してください。")
        exit()
        
    model.eval()

    # --- 4. 推論の実行 ---
    all_preds = []
    all_probs = []
    all_labels = []
    deck_A_list = []
    deck_B_list = []

    print("🔍 テストデータに対して推論を実行中...")
    with torch.no_grad():
        for dA, dB, labels in test_loader:
            dA, dB = dA.to(device), dB.to(device)
            outputs = model(dA, dB)
            
            # Sigmoidで 0.0 ~ 1.0 の確率（Win_Probability）に変換
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int) # 0.5以上なら「勝ち(1)」と予測
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().flatten())
            
            # デッキの中身(カードID配列)も保存しておく
            deck_A_list.extend(dA.cpu().numpy())
            deck_B_list.extend(dB.cpu().numpy())

    # --- 5. 評価と混同行列の計算 ---
    all_labels = np.array(all_labels).astype(int)
    all_preds = np.array(all_preds)

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["Lose (0)", "Win (1)"])

    print("\n" + "="*40)
    print("📈 評価レポート (Classification Report)")
    print("="*40)
    print(report)

    print("\n" + "="*40)
    print("🧮 混同行列 (Confusion Matrix)")
    print("="*40)
    print(f"True Negative  (正解:負け, 予測:負け) : {cm[0][0]} 🙆‍♂️")
    print(f"False Positive (正解:負け, 予測:勝ち) : {cm[0][1]} 💥(致命的な勘違い)")
    print(f"False Negative (正解:勝ち, 予測:負け) : {cm[1][0]} 💧(見逃し)")
    print(f"True Positive  (正解:勝ち, 予測:勝ち) : {cm[1][1]} 🙆‍♂️")

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
        'Actual_Result': all_labels,               # 実際の勝敗 (1=勝ち, 0=負け)
        'Predicted_Label': all_preds,              # モデルの予測 (1=勝ち, 0=負け)
        'Win_Probability': np.round(all_probs, 4), # モデルが弾き出した勝率 (0.0000 ~ 1.0000)
        'Is_Correct': all_labels == all_preds,     # 正解したかどうか (True/False)
        'Deck_A_Predicting': [str(d.tolist()) for d in deck_A_list], # 予測側のデッキ
        'Deck_B_Opponent': [str(d.tolist()) for d in deck_B_list]    # 相手側のデッキ
    })

    df_results['Confidence'] = abs(df_results['Win_Probability'] - 0.5)
    df_results = df_results.sort_values(by='Confidence', ascending=False).drop(columns=['Confidence'])

    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"📝 予測詳細のCSVを保存しました: {OUTPUT_CSV}")

    # --- 8. エラー分析のヒント ---
    wrong_high_conf = df_results[(df_results['Is_Correct'] == False) & (df_results['Win_Probability'] > 0.8)]
    print(f"\n🚨 [分析] モデルが80%以上の自信を持って『絶対勝てる！』と予測したのに負けた試合数: {len(wrong_high_conf)}件")