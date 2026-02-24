import requests
import time
import csv
import json
import urllib.parse

API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiIsImtpZCI6IjI4YTMxOGY3LTAwMDAtYTFlYi03ZmExLTJjNzQzM2M2Y2NhNSJ9.eyJpc3MiOiJzdXBlcmNlbGwiLCJhdWQiOiJzdXBlcmNlbGw6Z2FtZWFwaSIsImp0aSI6IjAzMjVlODNkLTI1ZDMtNDlkYy04MWNlLTZiYzBlNThmOGJiYyIsImlhdCI6MTc3MTg0NTA5Niwic3ViIjoiZGV2ZWxvcGVyL2QzOGRkY2RlLTMwZjEtMDFhMi1lYzc3LTkyMmU2OTQ0ZmJjNCIsInNjb3BlcyI6WyJyb3lhbGUiXSwibGltaXRzIjpbeyJ0aWVyIjoiZGV2ZWxvcGVyL3NpbHZlciIsInR5cGUiOiJ0aHJvdHRsaW5nIn0seyJjaWRycyI6WyIxNzUuMTc3LjQzLjEyNyJdLCJ0eXBlIjoiY2xpZW50In1dfQ.R4Z6HeBJpMLnJZvG1T7x6DeNDJOWECq5jdElo5JPrrZYcCfN6VyzA4VAHhVuoDOGd8wViKqCTNseFPQiuv78Jw"
BASE_URL = "https://api.clashroyale.com/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# ==========================================
# 1. test.csv の作成（上位400人のデータ）
# ==========================================
url_global = f"{BASE_URL}/locations/global/pathoflegend/players?limit=1000"
res_global = requests.get(url_global, headers=HEADERS)
global_data = res_global.json()['items']

test_players = global_data[:400]
test_tags = {p['tag'] for p in test_players}

test_rows = set()
for p in test_players:
    safe_tag = urllib.parse.quote(p['tag'])
    res_log = requests.get(f"{BASE_URL}/players/{safe_tag}/battlelog", headers=HEADERS)
    logs = res_log.json()
    
    for log in logs:
        if log.get('type') in ['PvP', 'pathOfLegend'] and 'team' in log:
            cards = tuple(sorted([c['id'] for c in log['team'][0]['cards']]))
            if len(cards) == 8:
                # プレイヤー名、タグ、カード8枚の順番
                test_rows.add((p['name'], p['tag']) + cards)
    time.sleep(0.1)

with open("test.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    for r in test_rows:
        writer.writerow(r)
print(f"test.csv を作成しました。（{len(test_rows)}件）")

# ==========================================
# 2. train.csv の作成（芋づる式収集）
# ==========================================
queue_tags = [p['tag'] for p in global_data[400:]]
visited_tags = set(test_tags)
train_rows = set()

TARGET_PLAYERS = 40000 # 学習データをどこまで広げるかのループ回数

for i in range(TARGET_PLAYERS):
    if not queue_tags:
        break
        
    current_tag = queue_tags.pop(0)
    visited_tags.add(current_tag)
    
    safe_tag = urllib.parse.quote(current_tag)
    res_log = requests.get(f"{BASE_URL}/players/{safe_tag}/battlelog", headers=HEADERS)
    logs = res_log.json()
    
    for log in logs:
        if log.get('type') in ['PvP', 'pathOfLegend'] and 'team' in log:
            # 自分のデータを追加
            team_cards = tuple(sorted([c['id'] for c in log['team'][0]['cards']]))
            if len(team_cards) == 8:
                train_rows.add((log['team'][0]['name'], log['team'][0]['tag']) + team_cards)
            
            # 相手のデータを追加（テスト対象者以外）
            opp_tag = log['opponent'][0]['tag']
            if opp_tag not in test_tags:
                opp_cards = tuple(sorted([c['id'] for c in log['opponent'][0]['cards']]))
                if len(opp_cards) == 8:
                    train_rows.add((log['opponent'][0]['name'], opp_tag) + opp_cards)
            
            # 相手を次の探索対象に追加
            if opp_tag not in visited_tags and opp_tag not in queue_tags:
                queue_tags.append(opp_tag)
                
    time.sleep(0.1)

with open("train.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    for r in train_rows:
        writer.writerow(r)
print(f"train.csv を作成しました。（{len(train_rows)}件）")