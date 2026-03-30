import streamlit as st
import torch
import json
import os
import sys

# パスを通す（srcフォルダを認識させるため）
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model.model import MatchupPredictor

# ==========================================
# 1. ページの設定
# ==========================================
st.set_page_config(page_title="クラロワ AI相性予測", page_icon="⚔️", layout="centered")
st.title("⚔️ クラロワ デッキ相性予測AI")
st.write("世界トップランカーの対戦データから学習したAIが、デッキの有利・不利を判定します！")

# ==========================================
# 2. モデルとデータの読み込み (キャッシュして高速化)
# ==========================================
@st.cache_resource
def load_model_and_data():
    base_dir = os.path.dirname(__file__)
    json_path = os.path.join(base_dir, 'data', 'cards.json')
    model_path = os.path.join(base_dir, 'learned_models', 'matchup_model.pth')
    
    # cards.json の読み込み（※JSONの構造に合わせて適宜調整してください）
    with open(json_path, 'r', encoding='utf-8') as f:
        cards_data = json.load(f)
    
    # ID -> 名前、名前 -> ID の辞書を作成
    id_to_name = {}
    name_to_id = {}
    
    # ★注意: cards_dataの構造が辞書型かリスト型かで読み込み方が変わります
    # もし {"26000000": "ナイト", ...} のような構造なら以下のようになります
    for card_id_str, name in cards_data.items():
        # もし JSONの中身が複雑な構造なら、ここを適宜書き換えます
        # 例: name = card_info["name"] など
        if isinstance(name, dict) and "name" in name:
            actual_name = name["name"]
        else:
            actual_name = str(name)
            
        id_to_name[int(card_id_str)] = actual_name
        name_to_id[actual_name] = int(card_id_str)
        
    raw_ids = list(id_to_name.keys())
    id_to_idx = {raw_id: i for i, raw_id in enumerate(raw_ids)}
    vocab_size = len(id_to_idx) + 1

    # モデルの準備
    model = MatchupPredictor(num_cards=vocab_size, embed_size=128)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, id_to_idx, name_to_id, list(name_to_id.keys())

model, id_to_idx, name_to_id, card_names = load_model_and_data()
unk_idx = len(id_to_idx)

# ==========================================
# 3. UI: デッキ選択部分
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔵 自分のデッキ (8枚)")
    my_deck_names = st.multiselect("カードを選んでください", options=card_names, max_selections=8, key="my_deck")

with col2:
    st.subheader("🔴 相手のデッキ (8枚)")
    op_deck_names = st.multiselect("カードを選んでください", options=card_names, max_selections=8, key="op_deck")

# ==========================================
# 4. 予測実行ボタン
# ==========================================
if st.button("🚀 相性を予測する！", use_container_width=True):
    if len(my_deck_names) != 8 or len(op_deck_names) != 8:
        st.warning("⚠️ 自分と相手のデッキをそれぞれ8枚ずつ選んでください！")
    else:
        # 名前をIDに、IDをインデックスに変換
        my_idx = [id_to_idx.get(name_to_id[name], unk_idx) for name in my_deck_names]
        op_idx = [id_to_idx.get(name_to_id[name], unk_idx) for name in op_deck_names]
        
        my_tensor = torch.tensor([my_idx], dtype=torch.long)
        op_tensor = torch.tensor([op_idx], dtype=torch.long)
        
        # 推論
        with torch.no_grad():
            score = model(my_tensor, op_tensor).item()
            
        # 結果の表示
        st.divider()
        st.header("📊 予測結果")
        
        if score > 0.3:
            st.success(f"相性スコア: **{score:+.3f}** \n🔵 **自分のデッキがかなり有利！**")
        elif score > 0.1:
            st.info(f"相性スコア: **{score:+.3f}** \n🟡 **自分のデッキが微有利（PS勝負）**")
        elif score > -0.1:
            st.warning(f"相性スコア: **{score:+.3f}** \n⚪️ **完全な互角**")
        elif score > -0.3:
            st.error(f"相性スコア: **{score:+.3f}** \n🟠 **相手のデッキが微有利（PS勝負）**")
        else:
            st.error(f"相性スコア: **{score:+.3f}** \n🔴 **相手のデッキがかなり有利（キツい）**")
            st.snow() # 絶望の演出