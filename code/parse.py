import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="EGHG")
    parser.add_argument('--bpr_batch', type=int, default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int, default=64,
                        help="the embedding size")
    parser.add_argument('--layer', type=int, default=3,
                        help="the layer num")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int, default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int, default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str, default='AMusic',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon, AToy, AMusic, ml-1m, lastfm]")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?', default="[20]",
                        help="@k test list")
    parser.add_argument('--comment', type=str, default="lgn")
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--multicore', type=int, default=0,
                        help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020,
                        help='random seed')
    parser.add_argument('--model', type=str, default='eghg',
                        help='rec-model eghg')
    parser.add_argument('--GPU', type=int, default=7,
                        help='choose free gpu')
    parser.add_argument('--load_adj', type=str, default='H_adj',
                        help='load user-item adj(simple_adj)')
    parser.add_argument('--cache', type=int, default=0,
                        help='use cache or not')
    parser.add_argument('--dropadj', type=float, default=1,
                        help='keep adj ratio')
    parser.add_argument('--Hadj', type=float, default=1,
                        help='the method of generate Hadj')
    parser.add_argument('--k_G', type=float, default=1,
                        help='laplace in simple Graph')
    parser.add_argument('--k_HG', type=float, default=1,
                        help='laplace in Hypergraph')
    parser.add_argument('--useW', type=int, default=0,
                        help='feature trans')
    parser.add_argument('--useA', type=int, default=0,
                        help='active function')
    parser.add_argument('--useT', type=int, default=1,
                        help='T-order Laplacian convolution')
    parser.add_argument('--Enhanced', type=int, default=0,
                        help='Enhanced  convolution kerenl')

    return parser.parse_args()
