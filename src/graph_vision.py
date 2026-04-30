import os
import json
import pandas as pd
import networkx as nx
from itertools import combinations
from pyvis.network import Network

if __name__ == "__main__":
    print("🚀 グラフの可視化を開始します...")

    # --- Step 1: データの読み込みとグラフ構築 ---
    csv_path = 'data/ranking_train.csv'
    json_path = 'data/cards.json'
    
    if not os.path.exists(csv_path):
        print(f"⚠️ ファイルが見つかりません: {csv_path}")
        exit()

    df = pd.read_csv(csv_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        cards_data = json.load(f)
        
    id_to_name = {int(k): v["name"] if isinstance(v, dict) else str(v) for k, v in cards_data.items()}
    card_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'card7', 'card8']
    
    co_occurrence = {}
    for row in df[card_cols].values:
        clean_deck = [id_to_name.get(c, f"Unknown({c})") for c in row if pd.notna(c)]
        for c1, c2 in combinations(sorted(clean_deck), 2):
            pair = (c1, c2)
            co_occurrence[pair] = co_occurrence.get(pair, 0) + 1

    G = nx.Graph()
    
    # 🌟 【重要】表示する線の太さの「閾値」
    # 7000本すべて描画すると画面が真っ黒になるため、
    # ここでは「100回以上一緒に使われた、本当に太い繋がり」だけを描画します。
    # 自分の見たい粒度に合わせて、この数字を 50 や 200 に調整してみてください。
    threshold = 100 
    
    for (c1, c2), w in co_occurrence.items():
        if w >= threshold:
            G.add_edge(c1, c2, value=w) # PyVisでは線の太さを'value'で指定します

    print(f"✅ 地図の作成完了: {G.number_of_nodes()}枚のカード, {G.number_of_edges()}本の繋がり")

    # --- Step 2: PyVisによるインタラクティブなHTML出力 ---
    print("🌐 ブラウザ用のHTMLファイルを生成中...")
    
    # キャンバスの設定（背景はダークモードが見やすいです）
    net = Network(height='800px', width='100%', bgcolor='#222222', font_color='white', select_menu=True)
    
    # NetworkXのグラフをPyVisに流し込む
    net.from_nx(G)
    
    # 物理演算（フォースダイナミクス）の設定
    # ノード同士が反発しあい、美しいネットワークの形に自動整理されます
    net.repulsion(node_distance=200, central_gravity=0.1, spring_length=200, spring_strength=0.05, damping=0.09)

    # HTMLファイルとして保存
    output_html = "clash_royale_network.html"
    net.write_html(output_html)
    
    print(f"🎉 可視化完了！\n 👉 フォルダ内に生成された '{output_html}' をダブルクリックしてブラウザで開いてください。")