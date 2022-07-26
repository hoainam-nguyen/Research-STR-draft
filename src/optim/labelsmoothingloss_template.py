import torch
from torch import nn
from src.model.vocab import Vocab

class ClusterCharacterLoss(nn.Module):
    def __init__(self, config):
        super(ClusterCharacterLoss, self).__init__()
        self.vocab = Vocab(config['vocab'])
        self.cluster_vocab = open('./src/loader/cluster_vocab.txt', 'r', encoding='utf8').readlines()
        self.cluster_list, self.cluster_dict = self.map_cluster()

    def map_cluster(self):
        cluster_dict = {}
        cluster_list = []
        for i, line in enumerate(self.cluster_vocab):
            for s in line[:-1]:
                ids = self.vocab.encode(s)
                cluster_list.append(ids[1])
                cluster_dict[ids[1]] = i 
        return cluster_list, cluster_dict

    def check(self, pred, tgt):
        if pred not in self.cluster_list:
            pred = -1
        else: pred = self.cluster_dict[pred]

        if tgt not in self.cluster_list:
            tgt = -1
        else: tgt = self.cluster_dict[tgt]

        if tgt == pred: return True
        return False

    def create_seq(self, pred_ids):
        pred_list = []
        for d in true_dist:
            pred_list.append(int(torch.argmax(d)))
        pred_tensor = torch.Tensor(pred_list)
        return pred_tensor

    def forward(self, pred, target):
        cluster_list, cluster_dict = self.map_cluster()
        length = min(len(pred), len(target))
        sum_penalty = 0
        for i in range(length):
            if pred[i] == target[i]:
                penalty = 0
            elif self.check(pred[i], target[i]):
                penalty = 0.5
            else:
                penalty = 1
            sum_penalty += penalty
        return sum_penalty/length

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, padding_idx, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.padding_idx = padding_idx

    def create_seq(self, pred_ids):
        pred_list = []
        for d in true_dist:
            pred_list.append(int(torch.argmax(d)))
        pred_tensor = torch.Tensor(pred_list)
        return pred_tensor

    def forward(self, pred, target):
        #pred = pred.log_softmax(dim=self.dim)
        #print(pred)
        #print(self.padding_idx)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 2))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0
            mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)

        # pred_tensor = self.create_seq(pred)
        # target_tensor = self.create_seq(true_dist)
        pred = pred.log_softmax(dim=self.dim)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(self,classes, padding_idx,  gamma=2., reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.CE = LabelSmoothingLoss(classes=classes, padding_idx = padding_idx)

    def forward(self, pred, target):
        CE_loss = self.CE(pred, target)
        pt = torch.exp(-CE_loss)
        F_loss = ((1 - pt)**self.gamma) * CE_loss
        if self.reduction == 'sum':
            return F_loss.sum()
        elif self.reduction == 'mean':
            return F_loss.mean()