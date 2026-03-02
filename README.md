# Clash Royale Deck Transformer (開発中)

## プロジェクト概要
クラッシュ・ロワイヤル（Clash Royale）のデッキ構成を学習し、カード間のシナジーやメタゲーム（環境）を分析するための機械学習プロジェクトです。
公式APIから取得したトッププレイヤーの対戦履歴データを元に、Transformerモデルを用いた自己教師あり学習（Masked Language Modeling）を行い、デッキの欠落カード予測や最適なカードの提案を目指します。

## 技術スタック
* **言語:** Python 3.12
* **機械学習:** PyTorch, scikit-learn
* **データ処理:** pandas, numpy
* **データ収集:** requests (Clash Royale Official API)

## セットアップ
必要なライブラリを一括でインストールします。

```bash
pip3 install -r requirements.txt