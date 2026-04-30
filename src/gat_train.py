import os
import json
import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import negative_sampling

# ==========================================
# 1. GATモデルの定義 (アテンション機構付きGNN)
# ==========================================
class CardGAT(nn.Module):
    def __init__(self, num_nodes, in_channels=128, out_channels=64):
        super().__init__()
        # カードの初期ベクトルを学習する層
        self.embedding = nn.Embedding(num_nodes, in_channels)
        
        # 第1層: 4つのマルチヘッドアテンションで相性を多角的に分析
        # edge_dim=1 を指定して、正規化した共起回数を考慮させる
        self.gat1 = GATConv(in_channels, out_channels, heads=4, concat=True, edge_dim=1)
        
        # 第2層: 各ヘッドの情報を統合して最終的なベクトル(64次元)にする
        self.gat2 = GATConv(out_channels * 4, out_channels, heads=1, concat=False, edge_dim=1)

    def encode(self, x_idx, edge_index, edge_attr):
        # 1. 初期埋め込みの取得
        x = self.embedding(x_idx)
        
        # 2. 第1層 (アテンション + 非線形活性化)
        h = self.gat1(x, edge_index, edge_attr=edge_attr)
        h = F.elu(h)
        
        # 3. 第2層 (最終的な特徴抽出)
        h = self.gat2(h, edge_index, edge_attr=edge_attr)
        return h

    def decode(self, z, edge_label_index):
        # 2つのノードのベクトルの内積を計算して「繋がりの強さ」を出す
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

# ==========================================
# 2. メイン実行プロセス
# ==========================================
if __name__ == "__main__":
    print(" GNN (GAT) による高精度埋め込み生成を開始します...")

    # --- Step 1: データ読み込み ---
    csv_path = os.path.join('data/ranking_train.csv')
    json_path = os.path.join('data/cards.json')
    
    if not os.path.exists(csv_path):
        print(f" ファイルが見つかりません: {csv_path}")
        exit()

    df = pd.read_csv(csv_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        cards_data = json.load(f)
        
    id_to_name = {int(k): v["name"] if isinstance(v, dict) else str(v) for k, v in cards_data.items()}
    card_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'card7', 'card8']
    
    # --- Step 2: グラフ構築と汎用カード対策 ---
    print(" グラフを構築し、汎用カードの重みを補正しています...")
    co_occurrence = {}
    for row in df[card_cols].values:
        clean_deck = [id_to_name.get(c, f"Unknown({c})") for c in row if pd.notna(c)]
        for c1, c2 in combinations(sorted(clean_deck), 2):
            pair = (c1, c2)
            co_occurrence[pair] = co_occurrence.get(pair, 0) + 1

    G = nx.Graph()
    threshold = 20
    for (c1, c2), w in co_occurrence.items():
        if w >= threshold:
            G.add_edge(c1, c2, weight=w)

    node_list = list(G.nodes())
    num_nodes = len(node_list)
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}

    # ハブノード（汎用カード）の影響を数学的に抑制する重み計算
    degrees = {node: val for node, val in G.degree(weight='weight')}
    src_list, dst_list, attr_list = [], [], []
    for u, v, d in G.edges(data=True):
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        # 正規化した重み: sqrt(次数の積)で割ることでハブ経由のリンクを細くする
        norm_w = d['weight'] / (np.sqrt(degrees[u]) * np.sqrt(degrees[v]))
        
        src_list.extend([u_idx, v_idx])
        dst_list.extend([v_idx, u_idx])
        attr_list.extend([[norm_w], [norm_w]])

    # PyTorchテンソルへ変換
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(attr_list, dtype=torch.float)
    x_idx = torch.arange(num_nodes, dtype=torch.long)

    # --- Step 3: 学習設定 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用デバイス: {device}")
    
    model = CardGAT(num_nodes=num_nodes).to(device)
    x_idx = x_idx.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # --- Step 4: 学習ループ ---
    epochs = 600
    print(f" 学習スタート ({epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # ベクトルの生成
        z = model.encode(x_idx, edge_index, edge_attr)
        
        # ポジティブサンプル（実際の繋がり）のスコア
        pos_out = model.decode(z, edge_index)
        
        # ネガティブサンプル（ランダムな繋がり）のスコア
        neg_edge_index = negative_sampling(edge_index, num_nodes=num_nodes)
        neg_out = model.decode(z, neg_edge_index)
        
        # 損失計算 (BCE Loss)
        pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones(pos_out.size(0)).to(device))
        neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros(neg_out.size(0)).to(device))
        loss = pos_loss + neg_loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"   Epoch {epoch:3d} | Loss: {loss.item():.4f}")

    # --- Step 5: 保存 ---
    print(" 結果を保存しています...")
    model.eval()
    with torch.no_grad():
        final_embeddings = model.encode(x_idx, edge_index, edge_attr).cpu().numpy()

    save_dir = 'models'
    os.makedirs(save_dir, exist_ok=True)
    
    # JSON用辞書の作成
    embeddings_dict = {idx_to_node[i]: final_embeddings[i].tolist() for i in range(num_nodes)}
    
    json_path = os.path.join(save_dir, 'gat_embeddings.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings_dict, f, ensure_ascii=False, indent=2)
        
    pt_path = os.path.join(save_dir, 'gat_embeddings.pt')
    torch.save({
        'node_to_idx': node_to_idx,
        'idx_to_node': idx_to_node,
        'embeddings': final_embeddings
    }, pt_path)

    print(f" 完了！\n - {json_path}\n - {pt_path}")