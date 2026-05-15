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
        # カードの初期ベクトル（実数のみを想定）
        self.card_embedding = nn.Embedding(num_cards, hidden_dim)
        
        # MagNetConv層の定義
        # K: チェビシェフ多項式の次数（通常1か2）
        # q: 磁荷（位相の回転角を制御するパラメータ。0.25がよく使われる）
        self.conv1 = MagNetConv(in_channels=hidden_dim, out_channels=hidden_dim, K=K, q=q, trainable_q=False)
        self.conv2 = MagNetConv(in_channels=hidden_dim, out_channels=hidden_dim, K=K, q=q, trainable_q=False)

    def forward(self, edge_index, edge_weight=None):
        device = self.card_embedding.weight.device
        num_cards = self.card_embedding.num_embeddings
        
        # 全ノード（カード）のインデックスを生成
        x = torch.arange(num_cards, device=device)

        # 1. 初期状態の定義（最初は実数のみ、虚部はゼロからスタート）
        x_real = self.card_embedding(x)
        x_imag = torch.zeros_like(x_real)

        # 2. MagNet第1層（ここでペッカとメガナイトの位相がズレ始める）
        x_real, x_imag = self.conv1(x_real, x_imag, edge_index, edge_weight)
        x_real = F.relu(x_real)
        x_imag = F.relu(x_imag) # 実部と虚部をそれぞれ活性化

        # 3. MagNet第2層（より深い間接的な相性を学習）
        x_real, x_imag = self.conv2(x_real, x_imag, edge_index, edge_weight)

        # 最終的な全カードの複素数表現（実部と虚部）を返す
        return x_real, x_imag


class SimpleSumPredictor(nn.Module):
    """
    MagNetでエンコードされた複素数ベクトルを受け取り、
    デッキ8枚をSum Poolingして勝敗を予測する超軽量モデル
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # 入力次元数の計算:
        # 実部と虚部で2倍、自分と相手のデッキでさらに2倍 -> hidden_dim * 4
        input_dim = hidden_dim * 4

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # 最終的な勝敗ロジット（>0なら勝ち、<0なら負け）
        )

    def forward(self, x_real, x_imag, my_decks, op_decks):
        """
        my_decks, op_decks: [batch_size, 8] (各デッキのカードIDリスト)
        """
        # 1. バッチ内の各カードのベクトルを抽出 [batch_size, 8, hidden_dim]
        my_real = x_real[my_decks]
        my_imag = x_imag[my_decks]
        op_real = x_real[op_decks]
        op_imag = x_imag[op_decks]

        # 2. デッキの単純加算（Sum Pooling） [batch_size, hidden_dim]
        my_real_sum = my_real.sum(dim=1)
        my_imag_sum = my_imag.sum(dim=1)
        op_real_sum = op_real.sum(dim=1)
        op_imag_sum = op_imag.sum(dim=1)

        # 3. 特徴量の結合（複素数空間を実数空間のロングベクトルに展開）
        # [batch_size, hidden_dim * 4]
        features = torch.cat([my_real_sum, my_imag_sum, op_real_sum, op_imag_sum], dim=-1)

        # 4. MLPによる勝敗判定
        logits = self.mlp(features)
        
        return logits.squeeze(-1)
    
import torch
import torch.nn as nn
import torch.nn.functional as F

# ... (MagNetEncoderの定義はそのまま) ...

class CrossAttentionPredictor(nn.Module):
    """
    MagNetの出力（8枚のカードベクトル）を受け取り、
    自分と相手のデッキ間でCross-Attentionを計算して勝敗を予測するモデル
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.3):
        super().__init__()
        
        # 実部と虚部を結合するため、カード1枚あたりの次元数は hidden_dim * 2 になる
        self.embed_dim = hidden_dim * 2
        
        # Cross-Attention層 (batch_first=True で [batch, seq, dim] を受け付ける)
        self.cross_attn_my_to_op = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_op_to_my = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # Attention後の特徴量を処理する層
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        
        # 最終判定用MLP (自分の8枚の集約ベクトル + 相手の8枚の集約ベクトル)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim * 2, 256),
            nn.GELU(), # Transformerと相性の良いGELUを採用
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_real, x_imag, my_decks, op_decks):
        # 1. 各カードの実部と虚部を抽出し、結合する [batch_size, 8, hidden_dim * 2]
        my_emb = torch.cat([x_real[my_decks], x_imag[my_decks]], dim=-1)
        op_emb = torch.cat([x_real[op_decks], x_imag[op_decks]], dim=-1)
        
        # 2. Cross-Attentionの計算
        # Q: 自分, K,V: 相手 -> 「自分のカードが、相手のどのカードを警戒すべきか」
        attn_my, _ = self.cross_attn_my_to_op(query=my_emb, key=op_emb, value=op_emb)
        # Q: 相手, K,V: 自分 -> 「相手のカードが、自分のどのカードを警戒しているか」
        attn_op, _ = self.cross_attn_op_to_my(query=op_emb, key=my_emb, value=my_emb)
        
        # 3. 残差接続 (Residual Connection) と Layer Normalization
        my_features = self.layer_norm1(my_emb + attn_my)
        op_features = self.layer_norm2(op_emb + attn_op)
        
        # 4. Pooling (8枚のカードベクトルを1つに集約)
        # Attentionで「相手を踏まえた上での役割」が計算済みなので、ここはSumやMeanでOK
        my_pooled = my_features.mean(dim=1)  # [batch_size, embed_dim]
        op_pooled = op_features.mean(dim=1)  # [batch_size, embed_dim]
        
        # 5. MLPで勝敗判定
        features = torch.cat([my_pooled, op_pooled], dim=-1) # [batch_size, embed_dim * 2]
        logits = self.mlp(features)
        
        return logits.squeeze(-1)