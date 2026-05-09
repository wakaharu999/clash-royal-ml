import os
import json
import pandas as pd
import numpy as np
import torch
import itertools

# ==========================================
# 🌟 ハイブリッド方式の定義エリア（一番上に書く）
# ==========================================
ROLE_MAP = {"Spell": 0, "Building": 1, "Support": 2, "Win_Condition": 3}

# 主軸（Win Condition）となるユニットのカードIDリスト（代表的なものを仮置き）
# ※ 必要に応じてご自身の基準で追加・削除してください
TARGET_MAIN_CARDS = [
    "Golem", "Lava Hound", "Giant", "Royal Giant", "Goblin Giant", 
    "Electro Giant", "Elixir Golem", "Goblin Barrel", "Goblin Drill", 
    "Wall Breakers", "Mortar", "X-Bow", "Balloon", "Royal Hogs", 
    "Hog Rider", "Ram Rider", "Three Musketeers", "Royal Recruits", 
    "P.E.K.K.A", "Giant Skeleton", "Mega Knight", "Boss Bandit", 
    "Graveyard", "Miner"
]

def build_global_graph():
    print("🌐 グローバルグラフの構築を開始します...")

    # --- 1. データの読み込み ---
    csv_path = 'data/matches.csv'
    json_path = 'data/cards.json'
    
    df = pd.read_csv(csv_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        cards_data = json.load(f)

    # 🌟 追加：名前リストから動的にIDリスト(整数)を生成する処理
    win_condition_ids = []
    for card_id_str, info in cards_data.items():
        if info.get("name") in TARGET_MAIN_CARDS:
            win_condition_ids.append(int(card_id_str))
            
    print(f"🎯 主軸(Win Condition)として {len(win_condition_ids)} 枚のカードを認識しました。")

    my_cols = [f'my_{i}' for i in range(8)]
    op_cols = [f'op_{i}' for i in range(8)]
    all_cards = np.unique(df[my_cols + op_cols].values)
    num_cards = len(all_cards)
    
    card_to_idx = {card: i for i, card in enumerate(all_cards)}

    # ==========================================
    #  1.5 ハイブリッド方式での「役割（Role）」ベクトル作成
    # ==========================================
    card_roles = []
    for card_id in all_cards:
        card_id_int = int(card_id)
        card_id_str = str(card_id)
        
        # 1. 動的生成したIDリストに含まれていれば Win_Condition
        if card_id_int in win_condition_ids:
            role = "Win_Condition"
        # 2. それ以外はIDの先頭2文字で判定
        elif card_id_str.startswith("28"):
            role = "Spell"
        elif card_id_str.startswith("27"):
            role = "Building"
        elif card_id_str.startswith("26"):
            role = "Support"
        else:
            role = "Support"
            
        card_roles.append(ROLE_MAP[role])
    
    node_roles = torch.tensor(card_roles, dtype=torch.long)

    # --- 2. 行列の初期化 ---
    co_occur = np.zeros((num_cards, num_cards))
    degree = np.zeros(num_cards)
    
    match_count = np.zeros((num_cards, num_cards))
    # 🌟 勝率ではなく「相性スコア（クラウン差を加味）」を貯める行列
    counter_score = np.zeros((num_cards, num_cards))

    # --- 3. 対戦データの集計 ---
    print("📊 対戦履歴を集計中... (クラウン差を考慮しています)")
    for _, row in df.iterrows():
        my_deck = [card_to_idx[row[col]] for col in my_cols]
        op_deck = [card_to_idx[row[col]] for col in op_cols]
        
        is_win = (row['result'] == 1)
        my_crowns = row['my_crowns']
        op_crowns = row['op_crowns']
        
        # 🌟 クラウン差を用いた勝者の圧倒度スコア計算
        # 基本勝利点1.0 ＋ クラウン差1つにつき0.5のボーナス
        # 例：3-0なら 1.0 + (3 * 0.5) = 2.5点
        # 例：1-0なら 1.0 + (1 * 0.5) = 1.5点
        if is_win:
            crown_diff = my_crowns - op_crowns
            win_points = 1.0 + (crown_diff * 0.5)
            lose_points = 0.0
        else:
            crown_diff = op_crowns - my_crowns
            lose_points = 1.0 + (crown_diff * 0.5)
            win_points = 0.0

        # [A] シナジー集計
        for deck in [my_deck, op_deck]:
            for c in deck:
                degree[c] += 1
            for c1, c2 in itertools.combinations(deck, 2):
                co_occur[c1, c2] += 1
                co_occur[c2, c1] += 1

        # [B] カウンター集計 (自分 vs 相手)
        for m_card in my_deck:
            for o_card in op_deck:
                match_count[m_card, o_card] += 1
                match_count[o_card, m_card] += 1
                
                # 自分カードから見ての相性スコア加算
                counter_score[m_card, o_card] += win_points
                # 相手カードから見ての相性スコア加算
                counter_score[o_card, m_card] += lose_points

    # --- 4. エッジの生成 ---
    source_nodes, target_nodes, edge_types = [], [], []

    # Relation 0: シナジー (閾値: 正規化スコアが 0.05 以上)
    SYNERGY_THRESHOLD = 0.05
    for i in range(num_cards):
        for j in range(num_cards):
            if i != j and co_occur[i, j] > 0:
                norm_weight = co_occur[i, j] / (np.sqrt(degree[i]) * np.sqrt(degree[j]))
                if norm_weight >= SYNERGY_THRESHOLD:
                    source_nodes.append(i)
                    target_nodes.append(j)
                    edge_types.append(0)

    # Relation 1: カウンター (閾値: 平均相性スコアが 0.7 以上)
    # ※クラウンボーナスがあるので、単なる勝率(0.55など)より閾値を少し高めに設定
    COUNTER_THRESHOLD = 0.7 
    for i in range(num_cards):
        for j in range(num_cards):
            if i != j and match_count[i, j] >= 50:
                avg_score = counter_score[i, j] / match_count[i, j]
                if avg_score >= COUNTER_THRESHOLD:
                    source_nodes.append(i)
                    target_nodes.append(j)
                    edge_types.append(1)

    # Relation 2: 重量競合 (コスト6以上)
    heavy_cards_idx = []
    for card_id, info in cards_data.items():
        card_id = int(card_id)
        if card_id in card_to_idx:
            cost = info.get('elixirCost', 0)
            if cost >= 6:
                heavy_cards_idx.append(card_to_idx[card_id])
                
    for i, j in itertools.permutations(heavy_cards_idx, 2):
        source_nodes.append(i)
        target_nodes.append(j)
        edge_types.append(2)

    # --- 5. 保存 ---
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)

    os.makedirs('data', exist_ok=True)
    save_path = 'data/global_graph.pt'
    torch.save({
        'edge_index': edge_index,
        'edge_type': edge_type,
        'node_roles': node_roles,  # 🌟 ノード特徴量（役割）も一緒に保存！
        'card_to_idx': card_to_idx
    }, save_path)

    print("\n✅ グラフの構築が完了しました！")
    print(f"📁 保存先: {save_path}")
    print(f"   総エッジ数: {edge_index.shape[1]}")
    print(f"     - シナジー: {(edge_type == 0).sum().item()}")
    print(f"     - カウンター: {(edge_type == 1).sum().item()}")
    print(f"     - 重量競合: {(edge_type == 2).sum().item()}")

if __name__ == "__main__":
    build_global_graph()