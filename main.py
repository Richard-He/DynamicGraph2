from datasets import UCI, sbm, Reddit, Elliptic, Bitcoin, AS
from model import EGNNC, Classifier
from splitter import splitter
from trainer import Trainer
from tasker import Edge_Cls_Tasker, Node_Cls_Tasker, Link_Pred_Tasker
import torch.nn as nn
import utils as u
import torch
import logger
import Cross_Entropy as ce
from graph import NeighborFinder
from TGAT import TGAN

def build_dataset(args):
    if args.data == 'bitcoinotc' or args.data == 'bitcoinalpha':
        if args.data == 'bitcoinotc':
            args.bitcoin_args = args.bitcoinotc_args
        elif args.data == 'bitcoinalpha':
            args.bitcoin_args = args.bitcoinalpha_args
        return Bitcoin(args)
    elif args.data == 'elliptic':
        return Elliptic(args)
    elif args.data == 'UCI':
        return UCI(args)
    elif args.data == 'AS':
        return AS(args)
    elif args.data == 'reddit':
        return Reddit(args)
    elif args.data == 'sbm20':
        args.sbm_args = args.sbm20_args
        return sbm(args)
    elif args.data == 'sbm50':
        args.sbm_args = args.sbm50_args
        return sbm(args)

def build_tasker(args, dataset):
    if args.task == 'link_pred':
        return Link_Pred_Tasker(args, dataset)
    elif args.task == 'edge_cls':
        return Edge_Cls_Tasker(args, dataset)
    elif args.task == 'node_cls':
        return Node_Cls_Tasker(args, dataset)

def build_classifier(args):
    if 'node_cls' == args.task:
        mult = 1
    else:
        mult = 2
    return Classifier(args, in_features=args.gcn_out_feats * mult, out_features=2).cuda()


if __name__ == '__main__':
    parser = u.create_parser()
    args = u.parse_args(parser)

    args.device = 'cuda'

    dataset = build_dataset(args)
    tasker = build_tasker(args, dataset)
    splitter = splitter(args, tasker)
    adj_list = u.build_adj_list(dataset)
    ngh_finder = NeighborFinder(adj_list)
    gcn = TGAN(ngh_finder, args.gcn_out_feats, dataset.num_nodes, args.num_hist_steps)
    classifier = build_classifier(args)
    loss = ce.Cross_Entropy(args, dataset).to('cuda')
    trainer = Trainer(args,
                      splitter=splitter,
                      gcn=gcn,
                      classifier=classifier,
                      comp_loss=loss,
                      dataset=dataset,
                      num_classes=dataset.num_classes)
    trainer.train()
