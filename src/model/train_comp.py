import os
import sys

# srcフォルダの場所を教える
sys.path.append('/Users/haru/Documents/GitHub/clash-royal-ml/src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.dataset_comp import ClashRoyaleMatchupDataset
from model import MatchupPredictor # type: ignore

def train():
    # パス設定
    base_dir = '/Users/haru/Documents/GitHub/clash-royal-ml'
    csv_path = os.path.join(base_dir, 'data/match_data.csv')
    json_path = os.path.join(base_dir, 'data/cards.json')
    pretrained_path = os.path.join(base_dir, 'learned_models/deck_transformer202602.pth') 
    save_path = os.path.join(base_dir, 'learned_models/matchup_model.pth')

    batch_size = 64
    num_epochs = 100
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    dataset = ClashRoyaleMatchupDataset(csv_path, json_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"データセットのサイズ: {len(dataset)} 試合")

    # モデル構築 (embed_sizeは128)
    model = MatchupPredictor(num_cards=dataset.vocab_size, embed_size=128)
    
    if os.path.exists(pretrained_path):
        print(f"前回の事前学習モデル ({pretrained_path}) を読み込みます...")
        model.encoder.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
        print("脳みその移植が完了しました！")
    
    model.to(device)

    # Lossの計算式（自動平均なし）
    criterion = nn.MSELoss(reduction='none')

    # ==========================================
    # ★ フェーズ1：天才の脳みそを「凍結」する
    # ==========================================
    print("❄️ フェーズ1: Encoderを凍結し、予測層(FC)だけを学習します...")
    for param in model.encoder.parameters():
        param.requires_grad = False  # Encoderの重みを更新しない（バリアを張る）
        
    # 予測パーツ（fc層）だけを学習させる
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    print("\n--- 学習スタート！ ---")
    for epoch in range(num_epochs):
        
        # ==========================================
        # ★ フェーズ2：途中でバリアを解除し、全体を優しく学習（エポック4から）
        # ==========================================
        if epoch == 3:
            print("\n🔥 フェーズ2: Encoderの凍結を解除！全体を微調整(ファインチューニング)します！")
            for param in model.encoder.parameters():
                param.requires_grad = True # バリア解除
            # 全体を対象にしつつ、学習率(lr)を 0.0001 に激減させて優しく学習する
            optimizer = optim.Adam(model.parameters(), lr=0.0001)

        model.train()
        total_loss = 0.0
        
        for i, (my_deck, op_deck, score) in enumerate(dataloader):
            my_deck, op_deck, score = my_deck.to(device), op_deck.to(device), score.to(device)
            
            optimizer.zero_grad()
            predictions = model(my_deck, op_deck)
            
            # Loss計算と重み付け
            base_loss = criterion(predictions, score)
            weights = torch.where(torch.abs(score) == 1.0, 5.0, 1.0).to(device)
            loss = torch.mean(base_loss * weights)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"🌟 Epoch [{epoch+1}/{num_epochs}] 完了 | Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"\n🎉 学習完了！ 相性予測モデルを {save_path} に保存しました！")

if __name__ == "__main__":
    train()