from __future__ import print_function, absolute_import

import argparse
import os
import os.path as osp
import sys
import yaml
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

from torch import nn

from reid import models
from reid.datasets.data_builder_attr import DataBuilder_attr
from reid.datasets.data_builder_cc import DataBuilder_cc
from reid.datasets.data_builder_ctcc import DataBuilder_ctcc
from reid.datasets.data_builder_sc import DataBuilder_sc
from reid.evaluation.evaluators import Evaluator
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, copy_state_dict


def main_worker(args):
    log_dir = osp.dirname(args.resume)
    sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    data_builder = DataBuilder_sc(args)
    test_loader, query_dataset, gallery_dataset = data_builder.build_data(is_train=False)

    # Create model
    model = models.create(args.arch, num_classes=0, net_config=args)

    # Load from checkpoint
    checkpoint = load_checkpoint(args.resume)
    copy_state_dict(checkpoint['state_dict'], model, strip='module.')

    model.cuda()
    model = nn.DataParallel(model)

    # Evaluator
    evaluator = Evaluator(model, args.validate_feat)
    print("retrieval visiual:")
    evaluator.vis_retrieval(test_loader, query_dataset.data, gallery_dataset.data, args.root, args.vis_root)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing the model")
    # data
    parser.add_argument('--config', default='scripts/config.yaml')
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--width_clo', type=int, default=128, help="input width")
    parser.add_argument('--query-list', type=str, required=True)
    parser.add_argument('--gallery-list', type=str, required=True)
    parser.add_argument('--gallery-list-add', type=str, default=None)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--vis_root', type=str, required=True)
    parser.add_argument('--root_additional', type=str, default=None)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--validate_feat', type=str, default='fusion', choices = ['person', 'clothes','fusion'])
    # model
    parser.add_argument('--dropout_clo', type=float, default=0)
    parser.add_argument('--patch_size_clo', type=int, default=16)
    parser.add_argument('--stride_size_clo', type=int, default=16)
    parser.add_argument('--patch_size_bio', type=int, default=16)
    parser.add_argument('--stride_size_bio', type=int, default=16)
    parser.add_argument('--attn_type', type=str, )
    parser.add_argument('--fusion_loss',type=str)
    parser.add_argument('--fusion_branch', type=str)
    parser.add_argument('--vit_type',type=str)
    parser.add_argument('--vit_fusion_layer',type=int)
    parser.add_argument('--test_feat_type', type=str, choices=['f','f_c','f_b','b','c'])
    parser.add_argument('--pool_clo', action='store_true')

    parser.add_argument('-a', '--arch', type=str, required=True, choices=models.names())
    parser.add_argument('--num_features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--resume', type=str, required=True, metavar='PATH')
    # testing configs
    parser.add_argument('--rerank', action='store_true', help="evaluation only")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--k1', type=int, default=30)
    parser.add_argument('--k2', type=int, default=6)
    parser.add_argument('--lambda-value', type=float, default=0.3)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    if 'common' in config:
        for k, v in config['common'].items():
            print(k, v)
            setattr(args, k, v)
    args.config = config

    main_worker(args)
