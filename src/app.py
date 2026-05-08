import torch
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from match_model import CrossAttentionPredictor

app = FastAPI(title="Clash Royale Match API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pth')
CARDS_JSON_PATH = os.path.join(BASE_DIR, 'cards.json')

# 1. マスターデータのロードとマッピング作成
def load_card_master():
    with open(CARDS_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    id_to_name = {str(k): v['name'] for k, v in data.items()}
    name_to_id = {v['name'].lower(): int(k) for k, v in data.items()}
    return id_to_name, name_to_id

ID_TO_NAME, NAME_TO_ID = load_card_master()

# 2. モデルのロード (以前と同じ)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CrossAttentionPredictor(num_cards=122, embed_dim=128)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 3. リクエスト形式（名前のリストでもIDのリストでも受け取れるようにする）
class MatchupRequest(BaseModel):
    deck_a: list[str]  # カード名のリスト (例: ["Tombstone", "Hog Rider", ...)
    deck_b: list[str]

@app.post("/predict")
async def predict(request: MatchupRequest):
    def resolve_ids(names):
        ids = []
        for name in names:
            name_lower = name.strip().lower()
            if name_lower in NAME_TO_ID:
                ids.append(NAME_TO_ID[name_lower])
            else:
                raise HTTPException(status_code=400, detail=f"カード名 '{name}' が見つかりません。")
        return ids

    try:
        # 名前をIDに変換
        ids_a = resolve_ids(request.deck_a)
        ids_b = resolve_ids(request.deck_b)

        # モデル入力用にテンソル化 (ここでは簡略化していますが、実際は id_to_idx の変換も必要です)
        t_a = torch.tensor([ids_a], dtype=torch.long).to(DEVICE)
        t_b = torch.tensor([ids_b], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            output = model(t_a, t_b)
            probability = torch.sigmoid(output).item()

        return {
            "prediction": {
                "win_rate_a": round(probability * 100, 2),
                "win_rate_b": round((1 - probability) * 100, 2)
            },
            "resolved_names_a": [ID_TO_NAME[str(i)] for i in ids_a],
            "resolved_names_b": [ID_TO_NAME[str(i)] for i in ids_b]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))