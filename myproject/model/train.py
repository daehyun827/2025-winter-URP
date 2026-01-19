#학습시키기
#DeepLoc 2.1 에서 4 fold cross validation으로 학습을 진행했기에 나의 모델에서도 같은 방식으로 진행해볼 예정
#학습 마친 후 마지막에 test data를 통해서 Accuracy를 측정해보려고 함
import os
import random
import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn
import torch.nn.functional as F
from sklearn.model_selection import PredefinedSplit           #Kfold 용
from tqdm import tqdm
import matplotlib.pyplot as plt
from MODEL import MyModel
from focal_loss import FocalLoss
from Data_loader import TrainDataset

train_path = r'C:\Users\uuuuu\plm\myproject\data\dataset\data_TRAIN.pt'
save_path = r'C:\Users\uuuuu\plm\myproject\results\test_2\model'
graph_path = r'C:\Users\uuuuu\plm\myproject\results\test_2\loss'

#재현성 확보를 위해서 randomness 제어해봄
def set_seed(seed=7):
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

#학습
def model_training():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  train_data = TrainDataset(train_path)

#어떤 partition인지 알아내기
  part_list = []

  for i in range(len(train_data)):
    part = train_data[i][3]
    part_list.append(int(part))

  sp = PredefinedSplit(test_fold = part_list)

  # 4 fold cross validation
  for fold, (train_idx, val_idx) in enumerate(sp.split(range(len(part_list)))):

    set_seed(7)

    train_data_fold = Subset(train_data, train_idx)
    val_data_fold = Subset(train_data, val_idx)

    g = torch.Generator()
    g.manual_seed(7)

    train_loader = DataLoader(train_data_fold, batch_size=1, shuffle=True, num_workers=0, generator=g)
    val_loader = DataLoader(val_data_fold, batch_size=1, shuffle=False)

    model = MyModel()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    model = model.to(device)
    model.train()
    num_epochs = 5                                                                    #epoch 수                                                                   

    #loss 저장용 리스트 만들기
    Train_loss_list = []
    Val_loss_list = []

    #Training 부분 / 20 epoch로 먼저 시도해볼 예정
    for epoch in range(num_epochs):

      model.train()
      total_train_loss = 0.0
      correct = 0
      total = 0

      tepoch = tqdm(train_loader, unit="batch", desc=f"Fold {fold} Training Epoch {epoch+1}/{num_epochs}", leave=False)

      for x, adj, y, part in tepoch:

        x = x[0].to(device)
        adj = adj[0].to(device)
        y = y[0].to(device)
        y_pred = model(x, adj)
        criterion = FocalLoss(y_pred, y).to(device)  

        pred = torch.argmax(y_pred, dim=0)
        loss = criterion(y_pred, y)
        #loss = F.cross_entropy(y_pred, y)                                             #multiclass classification을 위한 loss 함수로 cross entropy를 사용하였음
                                                                                      #논문에서는 focal loss를 사용했는데 두 loss 함수의 차이를 알아보면 좋을듯함+focal loss로도 진행해보기

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        correct += int((pred.item() == y.item()))
        total += 1

      t_loss = total_train_loss / total
      Train_loss_list.append(t_loss)
      train_accuracy = correct / total
      print(f"Train Accuracy: {train_accuracy}\n"
            f"Total: {total}\n"
            f"Correct Answer: {correct}\n")

      #validation 부분
      correct = 0
      total = 0
      total_val_loss = 0
      model.eval()

      with torch.no_grad():
        vepoch = tqdm(val_loader, unit="batch", desc=f"Fold {fold} Validation Epoch {epoch+1}/{num_epochs}", leave=False)
        for x, adj, y, part in vepoch:
          x = x[0].to(device)
          adj = adj[0].to(device)
          y = y[0].to(device)
          y_pred = model(x, adj)

          pred = torch.argmax(y_pred, dim=0)
          loss = criterion(y_pred, y)
          #loss = F.Cross_entropy(y_pred, y)

          correct += int((pred.item() == y.item()))
          total += 1

          total_val_loss += loss.item()
        v_loss = total_val_loss / total
        Val_loss_list.append(v_loss)

        val_accuracy = correct / total
        file_path = os.path.join(save_path, f'model_fold{fold}.pth')
        torch.save(model.state_dict(), file_path)
        print(f"Validation Accuracy: {val_accuracy}\n"
              f"Total: {total}\n"
              f"Correct_Answer: {correct}\n")

  #loss 시각화하기
    plt.figure()
    plt.plot(Train_loss_list, label="train loss")
    plt.plot(Val_loss_list, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Fold {fold} Loss")
    plt.legend()

    pic_path = os.path.join(graph_path, f"Fold {fold} loss.png")
    plt.savefig(pic_path)
    plt.close()

if __name__ == "__main__":
    model_training()


