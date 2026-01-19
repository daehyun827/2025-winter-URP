import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from MODEL import MyModel
from Data_loader import TestDataset

test_path = r'C:\Users\uuuuu\plm\myproject\data\dataset\data_TEST.pt'

#평가(평균 ensemble)
def evaluation(test_path):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  test_data = TestDataset(test_path)
  test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

  correct = 0
  total = 0
  P = 0
  T = 0
  L = 0
  S = 0

  with torch.no_grad():
    with tqdm(total=len(test_loader)) as pbar:
      for x, adj, y in test_loader:
        x = x[0].to(device)
        adj = adj[0].to(device)
        y = y[0].to(device)

        logits_sum = None
    
        for p in range(4):
          model = MyModel()
          model.eval()
          model.load_state_dict(torch.load(rf'C:\Users\uuuuu\plm\myproject\results\Train_model\model_fold{p}.pth'))
          model.to(device)

          logits = model(x, adj)
          logits_sum = logits if logits_sum is None else (logits_sum + logits)

        logits_avg = logits_sum / 4
        pred = torch.argmax(logits_avg, dim=0)

        if pred == y:
          if pred == 0:
            P += 1
          elif pred == 1:
            T += 1
          elif pred == 2:
            L += 1
          elif pred == 3:
            S += 1
            
        correct += int((pred.item() == y.item()))
        total += 1
        pbar.update(1)

  print(f"Average Test Accuracy: {correct / total}")
  print(f"Total: {total}")
  print(f"Correct: {correct}")
  print(f"Correct Peripheral: {P}")
  print(f"Correct Transmembrane: {T}")
  print(f"Correct LipidAnchor: {L}")
  print(f"Correct Soluble: {S}")

if __name__ == '__main__':
  evaluation(test_path)
