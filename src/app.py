import torch
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import CrossAttentionPredictor

app = FastAPI(title="Clash Royale Match API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pth')
CARDS_JSON_PATH = os.path.join(BASE_DIR, 'data', 'cards.json')

# 1. マスターデータのロードとマッピング作成
def load_card_master():
    with open(CARDS_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    id_to_name = {str(k): v['name'] for k, v in data.items()}
    name_to_id = {v['name'].lower(): int(k) for k, v in data.items()}
    
    # 🌟 ここが重要！学習時(np.unique)と同じように、IDを小さい順に並べて 0~120 の連番を振る
    sorted_raw_ids = sorted([int(k) for k in data.keys()])
    raw_id_to_idx = {raw_id: idx for idx, raw_id in enumerate(sorted_raw_ids)}
    
    return id_to_name, name_to_id, raw_id_to_idx

ID_TO_NAME, NAME_TO_ID, RAW_ID_TO_IDX = load_card_master()

# 2. モデルのロード
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CrossAttentionPredictor(num_cards=121, embed_dim=128)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 3. リクエスト形式
class MatchupRequest(BaseModel):
    deck_a: list[str]  
    deck_b: list[str]

@app.post("/predict")
async def predict(request: MatchupRequest):
    def resolve_ids(names):
        raw_ids = []
        indices = []
        for name in names:
            name_lower = name.strip().lower()
            if name_lower in NAME_TO_ID:
                raw_id = NAME_TO_ID[name_lower]
                raw_ids.append(raw_id)
                
                # 🌟 生ID (27000009) を モデル用の連番 (0-120) に変換して渡す！
                if raw_id in RAW_ID_TO_IDX:
                    indices.append(RAW_ID_TO_IDX[raw_id])
                else:
                    # 万が一未知のカードが来たらとりあえず0番（あるいはエラー）にする
                    indices.append(0) 
            else:
                raise HTTPException(status_code=400, detail=f"カード名 '{name}' が見つかりません。")
        return raw_ids, indices

    try:
        # 名前 → 生ID と インデックス(連番) の両方を取得
        raw_ids_a, idx_a = resolve_ids(request.deck_a)
        raw_ids_b, idx_b = resolve_ids(request.deck_b)

        # 🌟 モデル入力用には「インデックス(0-120)」のテンソルを使う
        t_a = torch.tensor([idx_a], dtype=torch.long).to(DEVICE)
        t_b = torch.tensor([idx_b], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            output = model(t_a, t_b)
            probability = torch.sigmoid(output).item()

        return {
            "prediction": {
                "win_rate_a": round(probability * 100, 2),
                "win_rate_b": round((1 - probability) * 100, 2)
            },
            # 返信用の名前復元には「生ID」を使う
            "resolved_names_a": [ID_TO_NAME[str(i)] for i in raw_ids_a],
            "resolved_names_b": [ID_TO_NAME[str(i)] for i in raw_ids_b]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))