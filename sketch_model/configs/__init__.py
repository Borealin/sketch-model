import argparse
from .config import *


def default_config() -> SketchModelConfig:
    return SketchModelConfig()


def config_with_arg() -> SketchModelConfig:
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', type=str, default=None)
    argparser.add_argument('--test', type=str, default=None)
    argparser.add_argument('--device', type=str, default=None)
    argparser.add_argument('--output', type=str, default=None)
    argparser.add_argument('--task', type=str, default=None)
    argparser.add_argument('--workers', type=str, default=None)
    args = argparser.parse_args()
    config = default_config()
    if args.train is not None:
        config.train_index_json = args.train
    if args.test is not None:
        config.test_index_json = args.test
    if args.device is not None:
        config.device = args.device
    if args.output is not None:
        config.output_dir = args.output
    if args.task is not None:
        config.task_name = args.task
    if args.workers is not None:
        config.num_workers = int(args.workers)
    return config
