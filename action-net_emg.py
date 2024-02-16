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

    global training_iterations

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes, valid_labels = utils.utils.get_domains_and_labels_action_net(args)
    input_size = 16
    model = getattr(model_list, args.model)(num_classes, input_size, args.batch_size) #ToDO: must be edited

    # the models are wrapped into the ActionRecognition task which manages all the training steps
    action_classifier = tasks.ActionRecognition("action-classifier", model, args.batch_size,      #* Passa alcuni parametri del default.yaml
                                                args.total_batch, args.models_dir, num_classes,
                                                args.train.num_clips, args.models, args=args)
    action_classifier.load_on_gpu(device)

    #if (args.action == "train"):
    training_iterations = args.train.num_iter * (args.total_batch // args.batch_size)
    train_loader = torch.utils.data.DataLoader(
            ActionEMGDataset(args.dataset.shift.split("-")[0], 'train', args.dataset),
            batch_size=args.batch_size, shuffle=False, num_workers=args.dataset.workers,
            pin_memory=True, drop_last=True
        )

    val_loader = torch.utils.data.DataLoader(torch.utils.data.DataLoader(
            ActionEMGDataset(args.dataset.shift.split("-")[0], 'val', args.dataset),
            batch_size=args.batch_size, shuffle=False, num_workers=args.dataset.workers,
            pin_memory=True, drop_last=True
        )
    
    train(action_classifier, train_loader, val_loader, device, num_classes, input_size)


def train(action_classifier, train_loader, val_loader, device, num_classes, input_size):

    global training_iterations

    data_loader_source = iter(train_loader)
    action_classifier.zero_grad()
    iteration = action_classifier.current_iter * (args.total_batch // args.batch_size)

    for i in range(iteration, training_iterations):
        # iteration w.r.t. the paper (w.r.t the bs to simulate).... i is the iteration with the actual bs( < tot_bs)
        real_iter = (i + 1) / (args.total_batch // args.batch_size)
        if real_iter == args.train.lr_steps:
            # learning rate decay at iteration = lr_steps
            action_classifier.reduce_learning_rate()
        # gradient_accumulation_step is a bool used to understand if we accumulated at least total_batch
        # samples' gradient
        gradient_accumulation_step = real_iter.is_integer()

        """
        Retrieve the data from the loaders
        """
        start_t = datetime.now()
        
        try:
            source_data, source_label = next(data_loader_source)    #source_label serve per la validation
        except StopIteration:
            return
        end_t = datetime.now()
        # print(source_data.shape)

        logits, _ = action_classifier.forward(source_data)
        action_classifier.compute_loss(logits, source_label, loss_weight=1)
        action_classifier.backward(retain_graph=False)
        action_classifier.compute_accuracy(logits, source_label)
    
            # update weights and zero gradients if total_batch samples are passed
        if gradient_accumulation_step:
            logger.info("[%d/%d]\tlast Verb loss: %.4f\tMean verb loss: %.4f\tAcc@1: %.2f%%\tAccMean@1: %.2f%%" %
                        (real_iter, args.train.num_iter, action_classifier.loss.val, action_classifier.loss.avg,
                         action_classifier.accuracy.val[1], action_classifier.accuracy.avg[1]))

            action_classifier.check_grad()
            action_classifier.step()
            action_classifier.zero_grad()

        # every eval_freq "real iteration" (iterations on total_batch) the validation is done, notice we validate and
        # save the last 9 models
        if gradient_accumulation_step and real_iter % args.train.eval_freq == 0:
            val_metrics = validate(action_classifier, val_loader, device, int(real_iter), num_classes)

            if val_metrics['top1'] <= action_classifier.best_iter_score:
                logger.info("New best accuracy {:.2f}%"
                            .format(action_classifier.best_iter_score))
            else:
                logger.info("New best accuracy {:.2f}%".format(val_metrics['top1']))
                action_classifier.best_iter = real_iter
                action_classifier.best_iter_score = val_metrics['top1']

            action_classifier.save_model(real_iter, val_metrics['top1'], prefix=None)
            action_classifier.train(True)


def validate(model, val_loader, device, it, num_classes):
   return 


if __name__ == '__main__':
    main()
