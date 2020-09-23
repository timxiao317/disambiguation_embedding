import data_parser
import pickle
from os.path import dirname, join, abspath

import embedding
import train_helper
import sampler
import eval_metric
import argparse


def parse_args():
    """
    parse the embedding model arguments
    """
    parser_arg = argparse.ArgumentParser(description =
                                         "run embedding for name disambiguation")
    parser_arg.add_argument("file_path", type = str, default="", help = 'input file name')
    parser_arg.add_argument("latent_dimen", type = int, default = 20,
                            help = 'number of dimension in embedding')
    parser_arg.add_argument("alpha", type = float, default = 0.02,
                            help = 'learning rate')
    parser_arg.add_argument("matrix_reg", type = float, default = 0.01,
                            help = 'matrix regularization parameter')
    parser_arg.add_argument("num_epoch", type = int, default = 100,
                            help = "number of epochs for SGD inference")
    parser_arg.add_argument("sampler_method", type = str,default='uniform', help = "sampling approach")
    return parser_arg.parse_args()


def main(args):
    """
    pipeline for representation learning for all papers for a given name reference
    """
    dataset = data_parser.DataSet(args.file_path)
    bpr_optimizer = embedding.BprOptimizer(args.latent_dimen, args.alpha,
                                           args.matrix_reg)
    pp_sampler = sampler.CoauthorGraphSampler()
    pd_sampler = sampler.BipartiteGraphSampler()
    dd_sampler = sampler.LinkedDocGraphSampler()
    eval_f1 = eval_metric.Evaluator()
    run_helper = train_helper.TrainHelper()
    run_helper.helper(args.num_epoch, dataset, bpr_optimizer,
                      pp_sampler, pd_sampler, dd_sampler,
                      eval_f1, args.sampler_method)


if __name__ == "__main__":
    DATA_SET_NAME = 'whoiswho_new'
    PROJ_DIR = dirname(dirname(abspath(__file__)))
    PARENT_PROJ_DIR = dirname(PROJ_DIR)
    print PARENT_PROJ_DIR
    RAW_DATA_DIR = join(PARENT_PROJ_DIR, 'sota_data', 'cikm_data', DATA_SET_NAME)
    SPLIT_PATH = join(PARENT_PROJ_DIR, 'split')
    with open(join(SPLIT_PATH, '{}_python2'.format(DATA_SET_NAME)), 'rb') as load:
        _, TRAIN_NAME_LIST, VAL_NAME_LIST, TEST_NAME_LIST = pickle.load(load)

    args = parse_args()
    for test_name in TEST_NAME_LIST:
        print test_name
        args.file_path = join(RAW_DATA_DIR, '{}.xml'.format(test_name))
        main(args)
