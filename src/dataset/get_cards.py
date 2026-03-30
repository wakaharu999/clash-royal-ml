import requests
import json

# --- 設定 ---
API_KEY = "CR_API_KEY"  # 先ほどと同じAPIキーを入力
BASE_URL = "https://api.clashroyale.com/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

# APIから全カード情報を取得
response = requests.get(f"{BASE_URL}/cards", headers=HEADERS)
data = response.json()

# IDをキーにした辞書を作成
card_catalog = {}
for card in data['items']:
    if 'elixirCost' not in card:
        continue
    card_catalog[card['id']] = {
        "name": card['name'],
        "rarity": card['rarity'],
        "iconUrls": card['iconUrls'],
        "elixirCost": card['elixirCost']
    }

# JSONファイルとして保存
with open("cards.json", "w", encoding="utf-8") as f:
    json.dump(card_catalog, f, ensure_ascii=False, indent=4)

print(f"全{len(card_catalog)}枚のカード情報を cards.json に保存しました！")