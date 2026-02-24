import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from dataset.dataset import ClashRoyaleDataset
from model.transformer import DeckTransformer

EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# 1. データの準備
# 削減したデータを使用する場合は '../data/train_reduced.csv' に書き換えてください
dataset = ClashRoyaleDataset('/Users/haru/Documents/GitHub/clash-royal-ml/data/train.csv', '/Users/haru/Documents/GitHub/clash-royal-ml/data/cards.json')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 2. モデルの準備
model = DeckTransformer(vocab_size=dataset.vocab_size).to(device)

# 3. 損失関数と最適化手法
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 4. 学習ループ
print("=== 学習スタート ===")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        logits = model(x)
        
        mask_positions = (x == dataset.mask_idx)
        masked_logits = logits[mask_positions]
        
        loss = criterion(masked_logits, y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
            
    avg_loss = total_loss / len(dataloader)
    print(f"--- Epoch {epoch+1} 完了 | 平均Loss: {avg_loss:.4f} ---")

# 5. 学習済みモデルの保存
save_dir = "/Users/haru/Documents/GitHub/clash-royal-ml/learned_models"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "deck_transformer.pth")
torch.save(model.state_dict(), save_path)
print(f"学習済みモデルを '{save_path}' に保存しました！")