from datetime import datetime
from statistics import mean
from utils.logger import logger
import torch.nn.parallel
import torch.optim
import torch
from utils.loaders import ActionEMGDataset
from utils.args import args
from utils.utils import pformat_dict
import utils
import numpy as np
import os
import models as model_list
import tasks
import wandb
from torchvision import transforms

# global variables among training functions
training_iterations = 0
modalities = None
np.random.seed(13696641)
torch.manual_seed(13696641)


def init_operations():
    """
    parse all the arguments, generate the logger, check gpus to be used and wandb
    """
    logger.info("Running with parameters: " + pformat_dict(args, indent=1))

    # this is needed for multi-GPUs systems where you just want to use a predefined set of GPUs
    if args.gpus is not None:
        logger.debug('Using only these GPUs: {}'.format(args.gpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)

    # wanbd logging configuration
    if args.wandb_name is not None:
        wandb.init(group=args.wandb_name, dir=args.wandb_dir)
        wandb.run.name = args.name + "_" + args.shift.split("-")[0] + "_" + args.shift.split("-")[-1]


def main():

    train_loader = torch.utils.data.DataLoader(
            ActionEMGDataset(args.dataset.shift.split("-")[0], 'train', args.dataset),
            batch_size=1, shuffle=False, num_workers=args.dataset.workers,
            pin_memory=True, drop_last=True
        )
    
    data_loader_source = iter(train_loader)
    for i in range(0, args.train.num_iter):
        try:
            source_data, source_label = next(data_loader_source)    #source_label serve per la validation
            # print(next(data_loader_source))
        except StopIteration:
            return
    
    num_classes, valid_labels = utils.utils.get_domains_and_labels_action_net(args)
    model = getattr(model_list, args.model)(num_classes, 1) #ToDO: must be edited

    #? serve il wrapper? Direi di no al momento
    # action_classifier = tasks.ActionRecognition("action-classifier", model, args.batch_size,      #* Passa alcuni parametri del default.yaml
    #                                             None, args.models_dir, num_classes,
    #                                             args.train.num_clips, args.models, args=args)
    
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.load_on_gpu(device)

    model.forward(source_data)
    # logits, _ = model.forward(source_data)

    # for i_val, (label, left_reading, right_reading, id) in enumerate(train_loader):
    #     print(id, left_reading, right_reading)
    #     exit()



def train(action_classifier, train_loader, val_loader, device, num_classes):
    return


def validate(model, val_loader, device, it, num_classes):
   return 


if __name__ == '__main__':
    main()
