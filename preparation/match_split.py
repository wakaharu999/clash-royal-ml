import pandas as pd
from sklearn.model_selection import train_test_split

# --- 設定 ---
INPUT_CSV = 'data/matches.csv'
TRAIN_CSV = 'data/train_matches.csv'
TEST_CSV  = 'data/test_matches.csv'
TEST_SIZE = 0.1  # 10%をテストデータにする
RANDOM_STATE = 42 # ここを固定することで、何度実行しても同じ分割になる

def split_data():
    print(f"📦 データを読み込んでいます: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    # データを分割
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # 保存
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    
    print(f"✨ 分割完了！")
    print(f"  - 訓練用データ ({len(train_df)}件) -> {TRAIN_CSV}")
    print(f"  - テスト用データ ({len(test_df)}件) -> {TEST_CSV}")

if __name__ == "__main__":
    split_data()