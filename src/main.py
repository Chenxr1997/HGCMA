import math
import numpy
import torch
from utils import load_data, set_params, evaluate
import warnings
import pickle as pkl
import os
import random

from module.hecl import HeCL
from module.mp_attn_encoder import Mp_attn_encoder

pid = str(os.getpid())
print(pid)

warnings.filterwarnings('ignore')
args = set_params()
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

## name of intermediate document ##
dataset_name = args.dataset

## random seed ##
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class Para():
    def __init__(self, hyper_dict):
        self.__dict__.update(hyper_dict)
hyper_dict = {}
data_dict = {}


def train():
    feat_dic, type_range, mp_dict, label, idx_train, idx_val, idx_test = \
        load_data(args.dataset, args.ratio, args.type_num)
    classes_num = label.shape[-1]
    feats_dim_dict = {k: feat_dic[k].shape[1] for k in feat_dic}
    print("seed", args.seed)
    print("Dataset:", args.dataset)
    print("The number of meta-paths:", int(len(mp_dict)))
    print("mp_mask:", str(args.mp_mask))
    print("nei_mask:", str(args.nei_mask))
    print("mp_prob:", str(args.mp_prob))
    print("nei_rate:", str(args.nei_rate))
    print("epochs:", str(args.epochs))

    hyper_dict["encoder1"] = Mp_attn_encoder
    hyper_dict["encoder2"] = None

    hyper_dict["hidden_dim"] = args.hidden_dim
    hyper_dict["feats_dim_dict"] = feats_dim_dict
    hyper_dict["feat_drop"] = args.feat_drop
    hyper_dict["attn_drop"] = args.attn_drop
    hyper_dict["mp_name"] = list(mp_dict.keys())
    hyper_dict["type_range"] = type_range
    hyper_dict["tau"] = args.tau
    hyper_dict["lam"] = args.lam
    hyper_dict["interest_type"] = args.interest_type
    hyper_dict["mp_mask"] = args.mp_mask
    hyper_dict["nei_mask"] = args.nei_mask
    hyper_dict["mp_prob"] = args.mp_prob
    hyper_dict["nei_rate"] = args.nei_rate
    h = Para(hyper_dict)
    
    model = HeCL(h)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
    
    t = 10 # warmup
    n_t = 0.5
    lambda1 = lambda epoch: (0.9*epoch / t+0.1) if epoch < t \
                else 1
                # else 0.1 if n_t * (1+math.cos(math.pi*(epoch - t)/(args.epochs-t)))<0.1 \
                # else n_t * (1+math.cos(math.pi*(epoch - t)/(args.epochs-t)))
                
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lambda1)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        feat_dic = {k: feat_dic[k].cuda() for k in feat_dic}
        mp_dict = {k: mp_dict[k].cuda() for k in mp_dict}
        label = label.cuda()
        idx_train = [i.cuda() for i in idx_train]
        idx_val = [i.cuda() for i in idx_val]
        idx_test = [i.cuda() for i in idx_test]

    data_dict["feat_dic"] = feat_dic
    data_dict["mp_dict"] = mp_dict

    for epoch in range(args.epochs+1):
        model.train()
        optimiser.zero_grad()

        d = Para(data_dict)
        loss = model(d)

        loss.backward()
        optimiser.step()
        scheduler.step()

        if epoch % 50 == 0:
            print("epoch {}, loss {}".format(epoch, loss.data.cpu()))

    model.eval()
    embeds = model.get_embeds(d)
    for i in range(len(idx_train)):
        evaluate(embeds, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label, classes_num, device, args.dataset,
                args.eva_lr, args.eva_wd)


if __name__ == '__main__':
    train()
    print("end")
