# 1. ベースとなるOSとPythonのバージョンを指定 (軽量なslim版を使用)
FROM python:3.10-slim

# 2. コンテナ内の作業ディレクトリを /app に設定
WORKDIR /app

# 3. 必要なシステムライブラリがあればインストール（今回は最小限）
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# 4. パッケージリストを先にコピーしてインストール (キャッシュを利用してビルドを爆速にするため)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. アプリケーションのコードと必要なファイル（モデル、JSON）を丸ごとコピー
# ※ .dockerignore に書かれたファイルはコピーされません
COPY src/ ./src/
COPY models/best_model.pth ./models/
COPY data/cards.json ./data/

# 6. Cloud Runが使用するポート番号を指定（デフォルトは8080）
EXPOSE 8080

# 7. サーバーの起動コマンド (src.appの中のappを起動)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]