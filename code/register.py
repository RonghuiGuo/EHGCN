import world
import dataloader
import model
import utils
from pprint import pprint

if world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

elif world.dataset in ['AMusic', 'ml-1m', 'AToy']:
    dataset = dataloader.newLoader(path="../data/"+world.dataset)

print('===========config================')
print("Decay: ", world.config["decay"])
print("LR: ", world.config["lr"])
print("Layers: ", world.config["EGHG_n_layers"])
print("Use cache or not: ", world.config["cache"])
print("Load Adj Way: ", world.config["load_adj"])
print("Hadj Way: ", world.config["Hadj"])
print("Epoch: ", world.TRAIN_epochs)
print("Hyperparameter k_G: ", world.config["k_G"])
print("Hyperparameter k_HG: ", world.config["k_HG"])
print("Keep edge prob: ", world.config['dropadj'])
print("Use W or not: ", world.config['useW'])
print("Use A or not: ", world.config['useA'])
print("t-order kernel: ", world.config['useT'])
print("Convolution Kernel Enhanced: ", world.config['Enhanced'])
print('===========end===================')

MODELS = {
    'eghg': model.EGHG
}