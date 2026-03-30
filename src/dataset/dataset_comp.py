import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json


class ClashRoyaleMatchupDataset(Dataset):
    def __init__(self, csv_file, json_file):
        # 1. ヘッダー付きで新しい match_data.csv を読み込む
        self.df = pd.read_csv(csv_file)
        
        with open(json_file, 'r', encoding='utf-8') as f:
            self.cards = json.load(f)
            
        # 2. 生のカードIDを0からの連番（インデックス）に変換する辞書
        self.raw_ids = list(self.cards.keys())
        self.id_to_idx = {int(raw_id): i for i, raw_id in enumerate(self.raw_ids)}
        
        # cards.jsonにない新カードが来た時のための「未知トークン(UNK)」を用意
        self.unk_idx = len(self.id_to_idx)
        self.vocab_size = len(self.id_to_idx) + 1
        
        # 3. 相性スコアの事前計算
        # (自分のクラウン - 相手のクラウン) / 3.0 を計算し、-1.0 〜 1.0 の連続値にする
        def calc_matchup_score(row):
            diff = row['my_crowns'] - row['op_crowns']
            if diff >= 3:
                return 1.0   # 3-0: 圧倒的相性有利
            elif diff == 2:
                return 0.7   # 2-0: かなり有利
            elif diff == 1:
                return 0.1   # 1-0: 微有利（ほぼPS勝負）
            elif diff == 0:
                return 0.0   # 引き分け: 完全互角
            elif diff == -1:
                return -0.1  # 0-1: 微不利（ほぼPS勝負）
            elif diff == -2:
                return -0.5  # 0-2: かなり不利
            else:
                return -1.0  # 0-3: 圧倒的相性不利（無理ゲー）

        self.df['score'] = self.df.apply(calc_matchup_score, axis=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1試合分のデータを取得
        row = self.df.iloc[idx]
        
        # 自分のデッキ8枚 (my_0 ~ my_7) を連番に変換
        my_deck_raw = [row[f'my_{i}'] for i in range(8)]
        my_deck_idx = [self.id_to_idx.get(int(c), self.unk_idx) for c in my_deck_raw]
        
        # 相手のデッキ8枚 (op_0 ~ op_7) を連番に変換
        op_deck_raw = [row[f'op_{i}'] for i in range(8)]
        op_deck_idx = [self.id_to_idx.get(int(c), self.unk_idx) for c in op_deck_raw]
        
        # スコアを取得
        score = row['score']
        
        # PyTorchのテンソルに変換して返す (自分デッキ, 相手デッキ, 相性スコア)
        # スコアはニューラルネットのLoss計算用に次元を持たせるため [score] とする
        return (
            torch.tensor(my_deck_idx, dtype=torch.long), 
            torch.tensor(op_deck_idx, dtype=torch.long), 
            torch.tensor([score], dtype=torch.float32)
        )

if __name__ == "__main__":
    # 動作確認 (パスは環境に合わせて調整してください)
    csv_path = '/Users/haru/Documents/GitHub/clash-royal-ml/data/match_data.csv'
    json_path = '/Users/haru/Documents/GitHub/clash-royal-ml/data/cards.json'
    
    dataset = ClashRoyaleMatchupDataset(csv_path, json_path)
    
    # DataLoaderでバッチサイズ(今回は4試合ずつ)にまとめる
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for my_deck, op_deck, score in dataloader:
        print("語彙サイズ (全カード種類 + UNK):", dataset.vocab_size)
        print("\n--- 自分のデッキ (my_deck) ---")
        print(my_deck)
        print("\n--- 相手のデッキ (op_deck) ---")
        print(op_deck)
        print("\n--- 相性スコア (score: -1.0 ~ 1.0) ---")
        print(score)
        break