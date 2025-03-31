import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import os
import torch.nn.functional as F
from model import SimCSEModel
from dataset import SimCSEDataset
import time
from datetime import timedelta


MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 32
TEMPERATURE = 0.05
CHECKPOINT_PATH = './checkpoint'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = SimCSEModel(MODEL_NAME).to(device)

raw_dataset = load_dataset("daily_dialog")
dialogs = raw_dataset['train']['dialog']
sentences = []
for dialog in dialogs:
    sentences.extend(dialog)

dataset = SimCSEDataset(sentences, tokenizer, max_len=MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

os.makedirs(CHECKPOINT_PATH, exist_ok=True)

best_loss = float('inf')
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    epoch_start = time.time()

    #TODO 
    for input_ids, attention_mask in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # 동일 문장을 두 번 인코딩
        input_ids = torch.cat([input_ids, input_ids], dim=0)
        attention_mask = torch.cat([attention_mask, attention_mask], dim=0)

        embeddings = model(input_ids=input_ids, attention_mask=attention_mask)

        embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 정규화

        similarity_matrix = torch.matmul(embeddings, embeddings.T)  # cosine similarity (normalized이므로)
        labels = torch.arange(0, embeddings.size(0) // 2, device=device)
        labels = torch.cat([labels + embeddings.size(0) // 2, labels])  # Positive pair의 인덱스

        # temperature scaling
        similarity_matrix = similarity_matrix / TEMPERATURE

        # 자기 자신은 제외 (masking)
        mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=device)
        similarity_matrix.masked_fill_(mask, -1e9)

        loss = F.cross_entropy(similarity_matrix, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch + 1} completed. Avg Loss: {avg_loss:.4f}. "
          f"Time taken: {timedelta(seconds=int(epoch_time))}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, "best_model.pth"))
        print(f"Checkpoint saved at epoch {epoch + 1} with loss {best_loss:.4f}")

total_time = time.time() - start_time
print(f"Training finished. Total time: {timedelta(seconds=int(total_time))}")
