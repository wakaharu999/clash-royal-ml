import os
import json
import pandas as pd
import networkx as nx
from itertools import combinations
import torch
import torch.nn as nn
import torch.optim as optim

# PyTorch Geometricのインポート
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

# ==========================================
# 1. GCNモデルの定義
# ==========================================
class CardGCN(nn.Module):
    def __init__(self, num_nodes, embedding_dim=64):
        super().__init__()
        self.initial_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, 64)
        self.conv2 = GCNConv(64, 64)

    def encode(self, edge_index):
        x = self.initial_embedding.weight
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

# ==========================================
# 1. 埋め込みモデルの定義（Shallow Embedding）
# ==========================================
class CardEmbedding(nn.Module):
    def __init__(self, num_nodes, embedding_dim=64):
        super().__init__()
        # 最初はランダムな辞書
        self.embedding = nn.Embedding(num_nodes, embedding_dim)

    def encode(self, edge_index):
        # 余計な情報交換（GCNConv）はせず、純粋な辞書の中身だけをそのまま出力する
        return self.embedding.weight

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)
    
# ==========================================
# メイン実行ブロック
# ==========================================
if __name__ == "__main__":
    print("GCN埋め込みベクトルの生成スクリプトを起動しました...")

    # ------------------------------------------------
    # Step 1: ランキングデータからグラフ（地図）を作成
    # ------------------------------------------------
    csv_path = os.path.join('data/ranking_train.csv')
    json_path = os.path.join('data/cards.json')
    
    print(f"データの読み込み中: {csv_path}")
    df = pd.read_csv(csv_path)

    with open(json_path, 'r', encoding='utf-8') as f:
        cards_data = json.load(f)
        
    # ID -> 名前 のマッピング辞書
    id_to_name = {}
    for card_id_str, name_info in cards_data.items():
        if isinstance(name_info, dict) and "name" in name_info:
            id_to_name[int(card_id_str)] = name_info["name"]
        else:
            id_to_name[int(card_id_str)] = str(name_info)

    # card1 ~ card8 の列だけを抽出してリスト化
    card_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'card7', 'card8']
    decks = df[card_cols].values.tolist()

    # IDを名前に変換
    decks_names = []
    for deck in decks:
        decks_names.append([id_to_name.get(card_id, f"Unknown({card_id})") for card_id in deck])

    print(f"{len(decks_names)} 個のデッキデータから共起行列を作成します...")

    co_occurrence = {}
    for deck in decks_names:
        for card1, card2 in combinations(sorted(deck), 2):
            pair = (card1, card2)
            co_occurrence[pair] = co_occurrence.get(pair, 0) + 1

    G = nx.Graph()
    threshold = 20 # ノイズ除外の閾値
    for (card1, card2), weight in co_occurrence.items():
        if weight >= threshold:
            G.add_edge(card1, card2, weight=weight)

    # PyTorch用のノードID変換（カード名 -> 連番インデックス）
    node_list = list(G.nodes())
    num_nodes = len(node_list)
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}

    print(f"グラフ構築完了: {num_nodes}枚のカード、{G.number_of_edges()}本の繋がり")

    # PyTorch Geometric用の形式 (edge_index) に変換
    source_nodes, target_nodes = [], []
    for u, v in G.edges():
        source_nodes.extend([node_to_idx[u], node_to_idx[v]])
        target_nodes.extend([node_to_idx[v], node_to_idx[u]])
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

    # ------------------------------------------------
    # Step 2: GCNの学習
    # ------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    model = CardEmbedding(num_nodes=num_nodes, embedding_dim=64).to(device)
    edge_index = edge_index.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 1000
    print("\n 学習スタート...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        z = model.encode(edge_index)
        
        # 正解の繋がり
        pos_out = model.decode(z, edge_index)
        pos_label = torch.ones(pos_out.size(0)).to(device)
        
        # 偽物の繋がり（ネガティブサンプリング）
        neg_edge_index = negative_sampling(
            edge_index=edge_index, num_nodes=num_nodes,
            num_neg_samples=edge_index.size(1)
        )
        neg_out = model.decode(z, neg_edge_index)
        neg_label = torch.zeros(neg_out.size(0)).to(device)
        
        # 損失の計算と重み更新
        out = torch.cat([pos_out, neg_out])
        labels = torch.cat([pos_label, neg_label])
        loss = criterion(out, labels)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"   Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.4f}")

    # ------------------------------------------------
    # Step 3: 結果の抽出と保存
    # ------------------------------------------------
    model.eval()
    with torch.no_grad():
        final_embeddings = model.encode(edge_index).cpu().numpy()

    # カード名 と 128次元ベクトル(リスト化) の辞書を作成
    embeddings_dict = {}
    for idx, card_name in idx_to_node.items():
        embeddings_dict[card_name] = final_embeddings[idx].tolist()

    save_dir = 'models'

    pt_path = os.path.join(save_dir, 'gcn_embeddings_shallow.pt')
    json_path = os.path.join(save_dir, 'gcn_embeddings_shallow.json')

    # 1. PyTorchのテンソルとして保存（モデルに読み込ませる用）
    torch.save({
        'node_to_idx': node_to_idx,
        'idx_to_node': idx_to_node,
        'embeddings': final_embeddings
    }, pt_path)
    
    # 2. JSONとしても保存（中身を人間が確認したり、他の言語で使う用）
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings_dict, f, ensure_ascii=False, indent=2)

    print("\nすべて完了しました！")
    print("以下のファイルを保存しました:")
    print("   - gcn_embeddings_shallow.pt   (PyTorch用バイナリデータ)")
    print("   - gcn_embeddings_shallow.json (人間も読めるテキストデータ)")