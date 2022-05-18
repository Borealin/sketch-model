import argparse
from .config import *


def get_config() -> SketchModelConfig:
    return SketchModelConfig()

def config_from_arg() -> SketchModelConfig:
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', type=str)
    argparser.add_argument('--test', type=str)
    argparser.add_argument('--device', type=str, default='cuda')
    argparser.add_argument('--output_dir', type=str, default='work_dir')
    args = argparser.parse_args()
    return SketchModelConfig(
        train_index_json=args.train,
        test_index_json=args.test,
        device=args.device,
        output_dir=args.output_dir,
    )