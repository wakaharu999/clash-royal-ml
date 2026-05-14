import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from collections import defaultdict
import ast
import json

# --- 設定パラメータ ---
MATCHES_CSV_PATH = "data/matches.csv"
CARDS_JSON_PATH = "data/cards.json"
OUTPUT_GRAPH_PATH = "models/directed_graph.pt"

MIN_MATCHES = 50          # 偶然を弾くため、最低この回数以上マッチングしたペアのみ対象
WIN_RATE_THRESHOLD = 0.53 # 勝率が53%以上のペアを「明確な有利（有向エッジ）」とする

def load_card_mapping():
    """カードIDを 0 ~ N-1 の連続したインデックスに変換するマッピングを作成"""
    with open(CARDS_JSON_PATH, 'r', encoding='utf-8') as f:
        cards_data = json.load(f)
    
    # JSONのキー（"26000020"などの文字列）を取得し、
    # 後のCSVデータと型を合わせるために数値(int)に変換してソートする
    card_ids = sorted([int(card_id) for card_id in cards_data.keys()])
    
    id_to_idx = {cid: i for i, cid in enumerate(card_ids)}
    idx_to_id = {i: cid for i, cid in enumerate(card_ids)}
    return id_to_idx, idx_to_id

def build_directed_graph():
    print("1. カードマッピングと対戦データの読み込み中...")
    id_to_idx, _ = load_card_mapping()
    num_nodes = len(id_to_idx)
    
    df = pd.read_csv(MATCHES_CSV_PATH)
    
    # (Card_A, Card_B) のペアに対する集計辞書
    # AがBに勝った回数と、AとBが対戦した総回数
    win_counts = defaultdict(int)
    match_counts = defaultdict(int)
    
    print(f"2. {len(df)}件の対戦データから勝敗行列を集計中...")
    for _, row in df.iterrows():
        # 1. 各列からカードIDを取り出してデッキリストを復元する
        my_deck = [int(row[f'my_{i}']) for i in range(8)]
        op_deck = [int(row[f'op_{i}']) for i in range(8)]
        
        # 2. result列（1=my_deck勝利, 0=op_deck勝利）を見て勝者と敗者を決定
        if row['result'] == 1:
            winner_deck = my_deck
            loser_deck = op_deck
        else:
            winner_deck = op_deck
            loser_deck = my_deck
        
        # 3. 勝者の各カードと、敗者の各カードのペアを「1勝」として記録
        for w_card in winner_deck:
            for l_card in loser_deck:
                if w_card == l_card:
                    continue # ミラー（同じカード同士）はエッジを張らない
                
                # 未知のカードID（アップデート直後など）はスキップ
                if w_card not in id_to_idx or l_card not in id_to_idx:
                    continue
                
                w_idx = id_to_idx[w_card]
                l_idx = id_to_idx[l_card]
                
                win_counts[(w_idx, l_idx)] += 1
                match_counts[(w_idx, l_idx)] += 1
                match_counts[(l_idx, w_idx)] += 1 # 逆から見ても対戦回数は同じ

    print("3. 有向エッジ（勝者 -> 敗者）の抽出中...")
    source_nodes = []
    target_nodes = []
    edge_weights = []
    
    for (card_a, card_b), wins in win_counts.items():
        total_matches = match_counts[(card_a, card_b)]
        
        # 対戦回数が少なすぎるペアはノイズになるので除外
        if total_matches < MIN_MATCHES:
            continue
            
        win_rate = wins / total_matches
        
        # 勝率が閾値（例: 53%）を超えていれば、「AはBの天敵である」として有向エッジを張る
        if win_rate >= WIN_RATE_THRESHOLD:
            source_nodes.append(card_a)
            target_nodes.append(card_b)
            
            # 50%からの上振れ分を重みとする (例: 勝率60%なら 0.60 - 0.50 = 0.10)
            # これにより、より強烈なカウンターほどエッジが太くなる
            weight = win_rate - 0.50 
            edge_weights.append(weight)

    # PyTorchテンソルへの変換
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)

    # グラフデータオブジェクトの作成
    # x (ノード特徴量) はMagNetEncoder内で生成するため、ここでは構造だけ保存
    graph_data = Data(edge_index=edge_index, edge_attr=edge_weight, num_nodes=num_nodes)
    
    print(f"4. グラフ構築完了! ノード数: {num_nodes}, 有向エッジ数: {edge_index.size(1)}")
    
    # 保存
    torch.save(graph_data, OUTPUT_GRAPH_PATH)
    print(f"-> {OUTPUT_GRAPH_PATH} に保存しました。")

if __name__ == "__main__":
    build_directed_graph()