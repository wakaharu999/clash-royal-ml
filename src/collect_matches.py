import os
import requests
import time
import csv
import urllib.parse
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CR_API_KEY_2")
BASE_URL = "https://api.clashroyale.com/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# ==========================================
# グローバルトッププレイヤーから対戦データを収集
# ==========================================
url_global = f"{BASE_URL}/locations/global/pathoflegend/players?limit=1000"
res_global = requests.get(url_global, headers=HEADERS)
global_data = res_global.json()['items']

match_rows = set()

# ここで何人分集めるか調整（テストなら10、本番なら1000など）
TARGET_PLAYERS = 5000

for p in global_data[:TARGET_PLAYERS]:
    safe_tag = urllib.parse.quote(p['tag'])
    res_log = requests.get(f"{BASE_URL}/players/{safe_tag}/battlelog", headers=HEADERS)
    logs = res_log.json()
    
    for log in logs:
        # PvP または pathOfLegend を対象とする
        if log.get('type') in ['PvP', 'pathOfLegend'] and 'team' in log and 'opponent' in log:
            team = log['team'][0]
            opponent = log['opponent'][0]
            
            my_cards = team.get('cards', [])
            op_cards = opponent.get('cards', [])
            
            if len(my_cards) == 8 and len(op_cards) == 8:
                # デッキをソートしてタプル化（setに入れるため）
                my_deck = tuple(sorted([c['id'] for c in my_cards]))
                op_deck = tuple(sorted([c['id'] for c in op_cards]))
                
                # 平均レベルの計算
                my_lev = sum([14 + c.get('level', 0) for c in my_cards]) / 8
                op_lev = sum([14 + c.get('level', 0) for c in op_cards]) / 8
                
                # 勝敗判定とクラウン数
                my_crowns = team.get('crowns', 0)
                op_crowns = opponent.get('crowns', 0)
                result = 1 if my_crowns > op_crowns else (-1 if my_crowns < op_crowns else 0)
                
                battle_time = log.get('battleTime')
                
                # 行データをタプルにして set に追加（デッキ、勝敗、クラウン数、レベル、日時）
                match_rows.add(my_deck + op_deck + (result, my_crowns, op_crowns, my_lev, op_lev, battle_time))
                
    time.sleep(0.1)

# ==========================================
# CSVへの書き込み
# ==========================================
# 保存先ディレクトリの作成
os.makedirs('/Users/haru/Documents/GitHub/clash-royal-ml/data', exist_ok=True)

with open("/Users/haru/Documents/GitHub/clash-royal-ml/data/match_data.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    
    # ヘッダーの書き込み
    header = [f'my_{i}' for i in range(8)] + [f'op_{i}' for i in range(8)] + \
             ['result', 'my_crowns', 'op_crowns', 'my_lev_avg', 'op_lev_avg', 'battle_time']
    writer.writerow(header)
    
    # データの書き込み
    for r in match_rows:
        writer.writerow(r)

print(f"match_data.csv を作成しました。（{len(match_rows)}件）")