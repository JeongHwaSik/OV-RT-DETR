"""by lyuwenyu
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import wandb

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS

"""
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/ov_rtdetr_r50vd_6x_o365.yml
"""

def main(args, ) -> None:
    '''
    main
    '''

    # issue: https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning,
        use_wandb=args.use_wandb,
    )

    # 🐝 use wandb.ai
    if args.use_wandb and dist.is_main_process():
        exp_name = args.config.split("/")[-1].rstrip(".yml")
        wandb.init(
            project="RT-DETR",
            name=exp_name,
            id="xd0r8u9o",
            config=cfg.yaml_cfg,
            resume="allow"
        )

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--use-wandb', action='store_true', default=False,)
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--seed', type=int, help='seed',)
    args = parser.parse_args()

    main(args)
