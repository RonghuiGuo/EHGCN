import world
import utils
from world import cprint
import torch
import numpy as np
import time
import Procedure
from os.path import join
import csv
from parse import parse_args

args = parse_args()

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
print("world.dataset:{}".format(world.dataset))
# ==============================
import register
from register import dataset

GPU_ID = world.GPU_id
torch.cuda.set_device(GPU_ID)
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

# log_file_name = 'logs/ablation_experiment/' + str(args.dataset) + '/' + str(args.k_G) + '_' + str(args.k_HG) + '_' \
#                 + str(args.lr) + '_' + \
#                 str(args.layer) + '_' + str(args.dropadj) + '_' + str(args.Hadj) + str(args.cache) + '_' + \
#                 str(args.load_adj) + '_' + str(args.Enhanced) + '_' + \
#                 str(args.decay) + '_' + str(args.useW) + '_' + str(args.useA) + '_' + str(args.useW) + '.txt'
log_file_name = 'logs/' + str(args.dataset) + '/' + str(args.k_G) + '_' + str(args.k_HG) + '_' \
                + str(args.lr) + '_' + \
                str(args.layer) + '_' + str(args.dropadj) + '_' + str(args.Hadj) + str(args.cache) + '_' + \
                str(args.load_adj) + '_' + str(args.Enhanced) + '.txt'
log_file = open(log_file_name, 'w')
log_file.write(str(args) + '\n\n')
hr_list = []
ndcg_list = []
mybest_hr = 0
mybest_ndcg = 0
mybest_epoch_hr = -1
mybest_epoch_ndcg = -1

Neg_k = 1

best_recall = -1000
best_ndcg = -1000
for epoch in range(world.TRAIN_epochs):
    print('======================')
    print(f'EPOCH[{epoch}/{world.TRAIN_epochs}]')
    start = time.time()

    if epoch % 10 == 0 and epoch != 0:
        cprint("[TEST]")
        results = Procedure.Test(dataset, Recmodel, epoch, world.config['multicore'])

        if results['recall'].item() > mybest_hr:
            mybest_hr = results['recall'].item()
            mybest_epoch_hr = epoch
        if results['ndcg'].item() > mybest_ndcg:
            mybest_ndcg = results['ndcg'].item()
            mybest_epoch_ndcg = epoch

        if results['recall'].item() > best_recall:
            best_recall = results['recall'].item()
        if results['ndcg'].item() > best_ndcg:
            best_ndcg = results['ndcg'].item()
        perf_str = 'epoch:' + str(epoch) + ' hr:' + str(results['recall']) + ' ndcg:' + str(results['ndcg'])
        log_file.write(perf_str + '\n')
        str_print = 'best epoch:' + str(mybest_epoch_hr) + " hr :" + str(mybest_hr) + 'best epoch:' \
                    + str(mybest_epoch_ndcg) + ' ndcg:' + str(mybest_ndcg)
        log_file.write(str_print + '\n')
        log_file.flush()
        print("best_recall:{}".format(best_recall))
        print("best_ndcg:{}".format(best_ndcg))
    output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k)
    print(f'[saved][{output_information}]')
    print(f"[TOTAL TIME] {time.time() - start}")

str_print = 'best epoch:' + str(mybest_epoch_hr) + " hr :" + str(mybest_hr) + 'best epoch:' \
            + str(mybest_epoch_ndcg) + ' ndcg:' + str(mybest_ndcg)

log_file.write(str_print + '\n')
log_file.flush()
