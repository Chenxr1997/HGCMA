import torch
import torch.nn as nn
import torch.nn.functional as F
from .contrast import Contrast


class HeCL(nn.Module):
    def __init__(self, h):
        super(HeCL, self).__init__()
        self.type_range = h.type_range
        self.interest_type = h.interest_type
        self.hidden_dim = h.hidden_dim
        self.fc_dict = nn.ModuleDict({k: nn.Linear(h.feats_dim_dict[k], h.hidden_dim, bias=True) 
                                      for k in h.feats_dim_dict})
        for k in self.fc_dict:
            nn.init.xavier_normal_(self.fc_dict[k].weight, gain=1.414)

        if h.feat_drop > 0:
            self.feat_drop = nn.Dropout(h.feat_drop)
        else:
            self.feat_drop = lambda x: x
        
        self.encoder1 = h.encoder1(h)
        self.encoder2 = h.encoder2(h) if h.encoder2 else self.encoder1
        self.contrast = Contrast(h)

    def forward(self, d, full=False):  # p a s
        h_all = []
        for k in self.type_range:
            h_all.append(F.elu(self.feat_drop(self.fc_dict[k](d.feat_dic[k]))))
        d.h = torch.cat(h_all, dim=0)

        z1 = self.encoder1(d, full=False)
        z1 = z1[self.type_range[self.interest_type]]
        z2 = self.encoder2(d, full=False)
        z2 = z2[self.type_range[self.interest_type]]
        d.z1 = z1   
        d.z2 = z2

        loss = self.contrast(d)
        return loss

    def get_embeds(self, d):
        h_all = []
        for k in self.type_range:
            h_all.append(F.elu(self.feat_drop(self.fc_dict[k](d.feat_dic[k]))))
        d.h = torch.cat(h_all, dim=0)

        z1 = self.encoder1(d)
        z1 = z1[self.type_range[self.interest_type]]
        
        return z1.detach()
