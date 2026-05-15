import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_signed_directed.nn.directed import MagNetConv

class MagNetEncoder(nn.Module):
    """
    全カードの有向グラフ（カウンター関係）を読み込み、
    各カードの「繋がり（実部）」と「相性の方向（虚部）」を複素数ベクトルとしてエンコードするモデル
    """
    def __init__(self, num_cards, hidden_dim, q=0.25, K=1):
        super().__init__()
        self.card_embedding = nn.Embedding(num_cards, hidden_dim)
        
        # MagNetConv層
        self.conv1 = MagNetConv(in_channels=hidden_dim, out_channels=hidden_dim, K=K, q=q, trainable_q=False)
        self.conv2 = MagNetConv(in_channels=hidden_dim, out_channels=hidden_dim, K=K, q=q, trainable_q=False)

    def forward(self, edge_index, edge_weight=None):
        device = self.card_embedding.weight.device
        num_cards = self.card_embedding.num_embeddings
        
        x = torch.arange(num_cards, device=device)

        # 初期状態（実数のみ、虚部はゼロ）
        x_real = self.card_embedding(x)
        x_imag = torch.zeros_like(x_real)

        # MagNet第1層
        x_real, x_imag = self.conv1(x_real, x_imag, edge_index, edge_weight)
        x_real = F.relu(x_real)
        x_imag = F.relu(x_imag)

        # MagNet第2層
        x_real, x_imag = self.conv2(x_real, x_imag, edge_index, edge_weight)

        return x_real, x_imag


class SimpleSumPredictor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # 入力次元数を変更: 自分(2) + 相手(2) + 差分(2) + 掛け算(2) = 8倍
        input_dim = hidden_dim * 8

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_real, x_imag, my_decks, op_decks):
        # 1. 抽出
        my_real = x_real[my_decks]
        my_imag = x_imag[my_decks]
        op_real = x_real[op_decks]
        op_imag = x_imag[op_decks]

        # 2. 単純加算
        my_real_sum = my_real.sum(dim=1)
        my_imag_sum = my_imag.sum(dim=1)
        op_real_sum = op_real.sum(dim=1)
        op_imag_sum = op_imag.sum(dim=1)

        # 🌟 3. 差分と相互作用を明示的に計算する（これが突破口！）
        diff_real = my_real_sum - op_real_sum
        diff_imag = my_imag_sum - op_imag_sum
        mult_real = my_real_sum * op_real_sum
        mult_imag = my_imag_sum * op_imag_sum

        # 4. すべて結合してMLPへ
        features = torch.cat([
            my_real_sum, my_imag_sum, 
            op_real_sum, op_imag_sum, 
            diff_real, diff_imag,     # 相手との戦力差
            mult_real, mult_imag      # 相手との相互作用
        ], dim=-1)

        logits = self.mlp(features)
        
        return logits.squeeze(-1)
    
class MagNetCrossAttentionPredictor(nn.Module):
    """
    MagNetが生成した「環境全体の相性ベクトル」を受け取り、
    試合の盤面（8枚 vs 8枚）でCross-Attentionをかけて局所的な有利不利を判定する最終キメラモデル
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.3):
        super().__init__()
        # 実部(繋がり)と虚部(相性の方向)を結合するため、カード1枚あたりの次元数は hidden_dim * 2 になる
        self.embed_dim = hidden_dim * 2

        # 相手のデッキを見て、自分のどのカードが刺さるか（警戒すべきか）を計算する層
        self.cross_attn_my = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_op = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        # 最終勝敗判定用MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_real, x_imag, my_decks, op_decks):
        amplitude = torch.sqrt(x_real**2 + x_imag**2 + 1e-8)
        phase = torch.atan2(x_imag, x_real)

        # 実部・虚部の代わりに、振幅と位相を結合して渡す [batch_size, 8, embed_dim]
        my_emb = torch.cat([amplitude[my_decks], phase[my_decks]], dim=-1)
        op_emb = torch.cat([amplitude[op_decks], phase[op_decks]], dim=-1)

        # 2. クロスアテンション計算 (以下は今のコードのままでOK)
        attn_my, _ = self.cross_attn_my(query=my_emb, key=op_emb, value=op_emb)
        attn_op, _ = self.cross_attn_op(query=op_emb, key=my_emb, value=my_emb)

        pool_my = attn_my.mean(dim=1)
        pool_op = attn_op.mean(dim=1)

        features = torch.cat([pool_my, pool_op], dim=-1)
        logits = self.mlp(features)

        return logits.squeeze(-1)