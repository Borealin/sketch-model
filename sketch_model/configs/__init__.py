import argparse

from .class_def import *
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
    argparser.add_argument('--aggregation', type=str, default=None)
    argparser.add_argument('--pos_pattern', type=str, default=None)
    argparser.add_argument('--name_sum', type=str, default=None)
    argparser.add_argument('--noimage', dest="use_image", action="store_false", default=True)
    argparser.add_argument('--noname', dest="use_name",
                           action="store_false", default=True)
    argparser.add_argument('--nocolor', dest="use_color",
                           action="store_false", default=True)
    argparser.add_argument('--noclass', dest="use_class",
                           action="store_false", default=True)
    argparser.add_argument('--nomask', dest="use_mask",
                           action="store_false", default=True)
    argparser.add_argument('--lazy', dest="lazy",
                           action="store_true", default=False)
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
    if args.pos_pattern is not None:
        if args.pos_pattern == '1/4':
            config.pos_pattern = PosPattern.ONE
        elif args.pos_pattern == '2/2':
            config.pos_pattern = PosPattern.TWO
        elif args.pos_pattern == '4/1':
            config.pos_pattern = PosPattern.FOUR
    if args.name_sum is not None:
        if args.name_sum == 'sum':
            config.sentence_method = SentenceMethod.SUM
        elif args.name_sum == 'mean':
            config.sentence_method = SentenceMethod.MEAN
        elif args.name_sum == 'max':
            config.sentence_method = SentenceMethod.MAX
    if args.aggregation is not None:
        if args.aggregation == "sum":
            config.aggregation = Aggregation.SUM
        elif args.aggregation == "concat":
            config.aggregation = Aggregation.CONCAT
    config.use_image = args.use_image
    config.use_name = args.use_name
    config.use_color = args.use_color
    config.use_class = args.use_class
    config.use_mask = args.use_mask
    config.lazy_load = args.lazy
    return config
