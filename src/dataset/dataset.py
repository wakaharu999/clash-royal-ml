import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import random

class ClashRoyaleDataset(Dataset):
    def __init__(self, csv_file, json_file):
        # データの読み込み
        self.df = pd.read_csv(csv_file, header=None)
        with open(json_file, 'r', encoding='utf-8') as f:
            self.cards = json.load(f)
            
        # 生のカードIDを0からの連番（インデックス）に変換する辞書を作成
        self.raw_ids = list(self.cards.keys())
        self.id_to_idx = {int(raw_id): i for i, raw_id in enumerate(self.raw_ids)}
        
        # MASK（隠しカード）用の特殊トークン番号を最後に割り当てる
        self.mask_idx = len(self.id_to_idx)
        
        # 全カード種類 + MASKトークンで語彙サイズ（モデルが扱う単語の種類数）を決定
        self.vocab_size = len(self.id_to_idx) + 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1行取得
        row = self.df.iloc[idx].values
        
        # 最初の2列(名前, タグ)を飛ばして、カードID(8枚)を取得し連番に変換
        deck_raw = row[2:10]
        deck_idx = [self.id_to_idx[int(c)] for c in deck_raw]
        
        # ランダムに1枚選んで隠す（穴埋め問題の作成）
        mask_pos = random.randint(0, 7)
        target = deck_idx[mask_pos] # これが予測すべき「正解」
        
        input_deck = deck_idx.copy()
        input_deck[mask_pos] = self.mask_idx # 選んだ場所をMASK番号で上書き
        
        # PyTorchのテンソルに変換して返す (入力デッキ, 正解のカード)
        return torch.tensor(input_deck, dtype=torch.long), torch.tensor(target, dtype=torch.long)

if __name__ == "__main__":
    # 動作確認 (src/dataset/ディレクトリから実行する場合、dataフォルダは2階層上)
    dataset = ClashRoyaleDataset('/Users/haru/Documents/GitHub/clash-royal-ml/data/train.csv', '/Users/haru/Documents/GitHub/clash-royal-ml/data/cards.json')
    
    # DataLoaderでバッチサイズごとにまとめる
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for x, y in dataloader:
        print("語彙サイズ (全カード種類 + MASK):", dataset.vocab_size)
        print("--- 入力データ (x): 各行のどこか1つがMASK番号になっている ---")
        print(x)
        print("--- 正解データ (y): 隠されたカードの番号 ---")
        print(y)
        break