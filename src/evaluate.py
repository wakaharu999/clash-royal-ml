import torch
from dataset.dataset import ClashRoyaleDataset
from model.transformer import DeckTransformer

# デバイスの設定
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# 1. テストデータとカタログの読み込み
test_dataset = ClashRoyaleDataset('../data/test.csv', '../data/cards.json')

# IDからカード名に変換するための辞書（逆引き辞書）
idx_to_name = {idx: test_dataset.cards[str(raw_id)]['name'] for raw_id, idx in test_dataset.id_to_idx.items()}
idx_to_name[test_dataset.mask_idx] = "[MASK (隠されたカード)]"

# 2. 学習済みモデルの読み込み
model = DeckTransformer(vocab_size=test_dataset.vocab_size).to(device)
model_path = "/Users/haru/Documents/GitHub/clash-royal-ml/learned_models/deck_transformer.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval() # 推論モードに切り替え

print("=== AIの実力テスト（Top5予測） ===")
# 試しに最初の5個のテストデッキを解かせてみる
for i in range(5):
    # テストデータから1問取得
    x, y = test_dataset[i]
    x_input = x.unsqueeze(0).to(device) # バッチ次元(1)を追加してモデルに入れる
    
    with torch.no_grad(): # 推論時は学習しないので勾配計算をオフ
        logits = model(x_input)
    
    # MASKされている場所の出力を取得
    mask_pos = (x == test_dataset.mask_idx).nonzero(as_tuple=True)[0].item()
    mask_logits = logits[0, mask_pos, :]
    
    # 確率が高い上位5枚のカードを取得
    probs = torch.softmax(mask_logits, dim=0)
    top5_prob, top5_idx = torch.topk(probs, 5)
    
    # --- 結果の表示 ---
    print(f"\n【テスト {i+1}】")
    deck_names = [idx_to_name[int(idx.item())] for idx in x]
    correct_name = idx_to_name[int(y.item())]
    
    print(f"入力デッキ: {deck_names}")
    print(f"★ 実際の正解: {correct_name}")
    print("AIの予測（Top 5）:")
    for rank in range(5):
        pred_name = idx_to_name[int(top5_idx[rank].item())]
        pred_prob = top5_prob[rank].item() * 100
        # 正解が含まれていたら分かりやすく色をつける（ターミナル用）
        match_mark = "🎯 正解!" if pred_name == correct_name else ""
        print(f"  {rank+1}位: {pred_name} ({pred_prob:.1f}%) {match_mark}")