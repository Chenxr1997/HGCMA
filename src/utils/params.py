import argparse
import sys

argv = sys.argv
dataset = argv[1]


def acm_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--ratio', type=int, default=[1, 3, 5])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nei_mask', type=bool, default=True)
    parser.add_argument('--mp_mask', type=bool, default=True)

    # The parameters of train
    parser.add_argument('--epochs', type=int, default=200)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--lr', type=float, default=0.0010)
    parser.add_argument('--l2_coef', type=float, default=0)
    
    # model-specific parameters
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--feat_drop', type=float, default=0.3)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--interest_type', type=str, default='p')

    # CL parameters
    parser.add_argument('--mp_prob', type=float, default=0.2)
    parser.add_argument('--nei_rate', type=float, default=0.3)

    args, _ = parser.parse_known_args()
    args.type_num = [3025, 5912, 57]  # the number of every node type
    return args


def dblp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--ratio', type=int, default=[1, 3, 5])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nei_mask', type=bool, default=True)
    parser.add_argument('--mp_mask', type=bool, default=True)

    # The parameters of train
    parser.add_argument('--epochs', type=int, default=950)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--l2_coef', type=float, default=0)
    
    # model-specific parameters
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--feat_drop', type=float, default=0.4)
    parser.add_argument('--attn_drop', type=float, default=0.35)
    parser.add_argument('--lam', type=float, default=0.5) 
    parser.add_argument('--interest_type', type=str, default='a')

    # CL parameters
    parser.add_argument('--mp_prob', type=float, default=0.7)
    parser.add_argument('--nei_rate', type=float, default=0.3)

    args, _ = parser.parse_known_args()
    args.type_num = [4057, 14328, 20]  # the number of every node type
    return args


def imdb_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="imdb")
    parser.add_argument('--ratio', type=int, default=[1, 3, 5])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nei_mask', type=bool, default=True)
    parser.add_argument('--mp_mask', type=bool, default=True)

    # The parameters of train
    parser.add_argument('--epochs', type=int, default=1150)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.25)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_coef', type=float, default=0.0)
    
    # model-specific parameters
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--feat_drop', type=float, default=0.4)
    parser.add_argument('--attn_drop', type=float, default=0.3)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--interest_type', type=str, default='m')

    # CL parameters
    parser.add_argument('--mp_prob', type=float, default=1.0)
    parser.add_argument('--nei_rate', type=float, default=0.7)
    
    args, _ = parser.parse_known_args()
    args.type_num = [4661, 2270, 5841]  # the number of every node type
    return args


def set_params():
    if dataset == "acm":
        args = acm_params()
    elif dataset == "dblp":
        args = dblp_params()
    elif dataset == "imdb":
        args = imdb_params()
    return args
