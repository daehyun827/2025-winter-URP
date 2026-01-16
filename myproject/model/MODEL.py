#GCN으로 앞서 구한 값들을 통해 단백질의 그래프구조 나타내기
#pytorch_geometric으로 간편히 하는 방법도 있지만, 공부 겸 pytorch로만 구현해볼 예정
import torch
import torch.nn as nn

#GCN layer 1개 정의하기
class GCNLayer(nn.Module):
  def __init__(self, in_dim, out_dim, act=None, p=0.25):
    super(GCNLayer, self).__init__()

    self.linear = nn.Linear(in_dim, out_dim)
    self.activation = act
    self.dropout = nn.Dropout(p)

  def forward(self, x, adj):

    out = self.linear(x)
    out = torch.matmul(adj, out)

    if self.activation != None:
      out = self.activation(out)  
      out = self.dropout(out)

    return out, adj

#GCN layer 여러개 쌓기
class GCNBlock(nn.Module):
  def __init__(self, num_layer, in_dim, hidden_dim, out_dim):
    super(GCNBlock, self).__init__()

    self.layers = nn.ModuleList()
    for i in range(num_layer):
      self.layers.append(GCNLayer(in_dim if i == 0 else hidden_dim,
                                  out_dim if i==num_layer-1 else hidden_dim,
                                  nn.ReLU() if i!=num_layer-1 else None))

  def forward(self, x, adj):
    for i, layer in enumerate(self.layers):
      out, adj = layer((x if i==0 else out), adj)
    return out, adj

# Graph정보 하나의 벡터로 만들기(Avg pooling 사용함)
# Attention pooling도 만들어 비교해보고 싶으나, 코딩실력과 Attention 이해의 한계로 인해 아직 못함

class Pooling(nn.Module):
  def __init__(self, in_dim, out_dim, p=0.25):
    super(Pooling, self).__init__()

    self.in_dim = in_dim
    self.out_dim = out_dim
    self.linear = nn.Linear(in_dim, out_dim)
    self.linear_2 = nn.Linear(out_dim, out_dim)
    self.activation = nn.ReLU()
    self.dropout = nn.Dropout(p)


  def forward(self, out):
    out = self.linear(out)
    out = self.activation(out)
    out = self.dropout(out)
    out = out.mean(dim=0)
    return out


# 다중 분류(Periperal/Transmembrane/LipidAnchor/Soluble)
class Classifier(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(Classifier, self).__init__()

    self.in_dim = in_dim
    self.out_dim = out_dim
    self.linear = nn.Linear(in_dim,out_dim)

  def forward(self, out):
    out = self.linear(out)
    return out

#모델 정의
class MyModel(nn.Module):
  def __init__(self, num_layer=2, in_dim=1280, hidden_dim=256, output_dim=256, pool_dim=128, num_class=4):
    super(MyModel, self).__init__()

    self.gcnlayer = GCNBlock(num_layer, in_dim, hidden_dim, out_dim=output_dim)
    self.pooling = Pooling(in_dim=output_dim, out_dim=pool_dim)
    self.classifier = Classifier(in_dim=pool_dim, out_dim=num_class)

  def forward(self, x, adj):
    out, adj = self.gcnlayer(x, adj)
    out = self.pooling(out)
    out = self.classifier(out)
    return out
