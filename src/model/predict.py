import os
import sys
import json
import torch

sys.path.append('/Users/haru/Documents/GitHub/clash-royal-ml/src')
from model import MatchupPredictor # type: ignore

def predict_matchup():
    base_dir = '/Users/haru/Documents/GitHub/clash-royal-ml'
    json_path = os.path.join(base_dir, 'data/cards.json')
    model_path = os.path.join(base_dir, 'learned_models/matchup_model.pth')
    
    # カード辞書の読み込み
    with open(json_path, 'r', encoding='utf-8') as f:
        cards = json.load(f)
        
    raw_ids = list(cards.keys())
    id_to_idx = {int(raw_id): i for i, raw_id in enumerate(raw_ids)}
    unk_idx = len(id_to_idx)
    vocab_size = len(id_to_idx) + 1

    # モデルの準備
    model = MatchupPredictor(num_cards=vocab_size, embed_size=128)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval() # 評価モード（学習しない設定）にする

    # 🚀 ここにテストしたいデッキのカードID（8枚）を入れる！
    # 例として「2.6ホグ」と「ペッカ攻城」などのIDを後でここに入力します

   # 自分のデッキ: 三銃士（3M、ポンプ、ハンター、ユーノ、ゴースト、アイゴレ、アイスピ、ババ樽）
    my_deck_raw = [26000028, 27000007, 26000044, 26000046, 26000050, 26000038, 26000030, 28000015]
    # 相手のデッキ: 2.9高回転バルーン（バルーン、ディガー、マスケット、ボムタワー、コウモリ、アイゴレ、雪玉、ババ樽）
    op_deck_raw = [26000006, 26000032, 26000014, 27000004, 26000049, 26000038, 28000017, 28000015]

    my_deck_idx = [id_to_idx.get(c, unk_idx) for c in my_deck_raw]
    op_deck_idx = [id_to_idx.get(c, unk_idx) for c in op_deck_raw]

    # テンソルに変換してバッチ次元（1試合分）を追加
    my_tensor = torch.tensor([my_deck_idx], dtype=torch.long)
    op_tensor = torch.tensor([op_deck_idx], dtype=torch.long)

    # 予測の実行
    with torch.no_grad():
        score = model(my_tensor, op_tensor).item()

    print("\n⚔️ 予測結果 ⚔️")
    print(f"相性スコア: {score:+.3f} (-1.0 ～ +1.0)")
    
    if score > 0.3:
        print("判定: 🟢 自分のデッキがかなり有利！")
    elif score > 0.1:
        print("判定: 🟡 自分のデッキが微有利（PS勝負）")
    elif score > -0.1:
        print("判定: ⚪️ 完全な互角")
    elif score > -0.3:
        print("判定: 🟠 相手のデッキが微有利（PS勝負）")
    else:
        print("判定: 🔴 相手のデッキがかなり有利（キツい）")

if __name__ == "__main__":
    predict_matchup()