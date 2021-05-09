import os
import h5py
import math
import copy
import shutil
from tqdm import tqdm
import torch
import torch.nn as nn
import itertools
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from model.model_util import WarmUpLR, fix_random_seeds
from model.word_embeddings import ViCoWordEmbeddings
import model.resnet as resnet_models
from data.dataset import get_custom_dataset

import util.io as io
from logging import getLogger
from util.util import *
logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of SKA (Supervised Kernel Alignment)")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

###################################
# DATASET
###################################
parser.add_argument(
    '--dataset-type',
    type=str,
    default='STL10',
    choices=['Cifar100', 'STL10'],
    help='which dataset to train on'
)
parser.add_argument(
    '--data-path',
    type=str,
    default='/HDD/DATASETS/',
    help='where dataset is located'
)
parser.add_argument(
    '--embed-path',
    type=str,
    default='/HDD/DATASETS/pretrained-embeddings',
    help='where vico word embeddings is located'
)
parser.add_argument(
    '--download-dataset',
    action='store_true',
    help='download the dataset from pytorch.datasets'   
)

###################################
# TRAINING PARAMS
###################################
parser.add_argument(
    '--num-epochs',
    default=500,
    type=int,
    help='Number of epochs to train'
)
parser.add_argument(
    '--batch-size',
    default=128,
    type=int,
    help='batch size.'
)
parser.add_argument(
    '--optimizer',
    type=str,
    default='SGD',
    choices=['SGD', 'Adam'],
    help='Optimizer. Choose from: [SGD, Adam]'
)
parser.add_argument("--final_lr", 
    default=0.001, 
    type=float, 
    help="base learning rate")
parser.add_argument(
    '--lr',
    type=float,
    default=0.1,
    help='learning rate'
)
parser.add_argument("--warmup_epochs", 
    default=10, 
    type=int, 
    help="number of warmup epochs"
)
parser.add_argument("--start_warmup", 
    default=0.01, 
    type=float,                
    help="initial warmup learning rate"
)
parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='momentum for the optimizer.'
)
parser.add_argument(
    '--num-workers',
    default=4,
    type=int,
    help='number of workers for the dataloader'
)


###################################
# MODEL
###################################
parser.add_argument(
    '--num-layers',
    default=50,
    type=int,
    choices=[18, 34, 50, 101],
    help='Number of layer in the feature extractor(resnet): [18,34,50,101].'
)
parser.add_argument(
    '--sim-loss',
    action='store_true',
    help='include semantic similarity in the loss'
)

parser.add_argument(
    '--one-hot',
    action='store_true',
    help='use one-hot labels instead of embeddings'
)

parser.add_argument(
    '--ce-warmup-epochs',
    default=100,
    type=int,
    help='linearly increase the cross-entropy loss weight'
)
parser.add_argument(
    '--vico-mode',
    type=str,
    default='vico_linear',
    choices=['vico_linear', 'vico_select'],
    help='embedding types to be used in semantic similarity loss')
parser.add_argument(
    '--linear-dim',
    type=int, 
    default=200,
    help='dimension of embeddings if linear is selected')

parser.add_argument(
    '--no-hypernym',
    action='store_true',
    help='exclude hypernym co-oc. from vico embeddings')

parser.add_argument(
    '--no-glove',
    action='store_true',
    help='exclude glove embeddings from vico embeddings')


##################################
#  LOGGING & SAVING
##################################
parser.add_argument("--dump-path", 
    type=str, 
    default="./experiments/SKA/default",
    help="experiment dump path for checkpoints and log"
)
parser.add_argument(
    '--checkpoint-freq',
    type=int,
    default=10,
    help='save the model at every model-save-step iterations.'
)
parser.add_argument(
    '--val-freq',
    type=int,
    default=1,
    help='evaluate the model at every val-freq epochs.'
)
parser.add_argument("--seed", type=int, default=31, help="seed")

def main():
    global args
    args = parser.parse_args()
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss", "acc", "acc_val")
    writer = SummaryWriter(args.dump_path)

    dataloaders = {}
    num_classes = 10 if args.dataset_type == 'STL10' else 100
    for split in ['train', 'test']:
        dataset = get_custom_dataset(args.dataset_type,
                                    root=args.data_path, 
                                    split=split, 
                                    download=args.download_dataset, 
                                    return_target_word=True
        )
        #collate_fn = dataset.get_collate_fn()
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split=='train',
            num_workers=args.num_workers
        )

    word_embeddings = None
    if args.sim_loss:
        word_embeddings = ViCoWordEmbeddings(
            root = args.embed_path, 
            num_classes=num_classes,
            vico_mode=args.vico_mode,
            one_hot=args.one_hot,
            linear_dim =args.linear_dim,
            no_hypernym=args.no_hypernym,
            no_glove=args.no_glove,
            pool_size=None
        )
        word_embeddings = word_embeddings.to(device)
    
    model = resnet_models.__dict__['resnet{}'.format(args.num_layers) ](
        small_image=True,
        hidden_mlp=0,
        output_dim=num_classes,
        returned_featmaps = [3,4,5],
        multi_cropped_input=False
    )
    model = model.to(device)
    
    lr = args.lr
    params = model.parameters()
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(
            params,
            lr=lr,
            momentum=args.momentum,
            weight_decay=1e-4)
    elif args.optimizer == 'Adam':
        opt = optim.Adam(
            params,
            lr=lr,
            weight_decay=1e-4)
    else:
        assert(False), 'optimizer not implemented'

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    warmup_lr_schedule = np.linspace(args.start_warmup, args.lr, len(dataloaders['train']) * args.warmup_epochs)
    iters = np.arange(len(dataloaders['train']) * (args.num_epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(dataloaders['train']) * (args.num_epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    logger.info("Building optimizer done.")

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0, "val_acc":0, "best_val_acc":0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
    )

    eval_score = to_restore["val_acc"]
    start_epoch = to_restore["epoch"]
    best_val_acc = to_restore["best_val_acc"]
    for epoch in range(start_epoch, args.num_epochs):
        
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # train for one epoch
        scores = train_model(model, word_embeddings, dataloaders['train'], optimizer, criterion, epoch, lr_schedule, writer)

        # evaluate if needed
        if epoch % args.val_freq == 0:
            eval_score = eval_model(model, word_embeddings, dataloaders['test'], epoch, writer)
            if eval_score > best_val_acc:
                best_val_acc = eval_score
        
        training_stats.update(scores + (eval_score,))

        # after epoch: save checkpoints
        save_dict = {
            "epoch": epoch + 1,
            "val_acc": eval_score,
            "best_val_acc": best_val_acc,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(
            save_dict,
            os.path.join(args.dump_path, "checkpoint.pth.tar"),
        )
        if epoch % args.checkpoint_freq == 0 or epoch == args.num_epochs - 1:
            shutil.copyfile(
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
                os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
            )

    writer.close()

def train_model(model, word_embeddings, dataloader, optimizer, criterion, epoch, lr_schedule, writer):
    # train the network for one epoch
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    model.train()
    if word_embeddings != None:
        word_embeddings.train()

    ce_warmup_iters = len(dataloader) * args.ce_warmup_epochs
    end = time.time() 
    for it,data in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)
        # update learning rate
        iteration = epoch * len(dataloader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]
        
        # ============ forward passes ... ============
        imgs = data['img'].to(device)
        label_idxs = data['label_idx'].to(device)
        logits, feats = model(imgs)

        # ============ backward and optim step ... ============
        log_items = {}
            
        # CKA loss between feature maps and word embeddings
        sim_loss_sum = 0.0
        if word_embeddings != None:
            targets = torch.linspace(0.4, 0.8, steps=len(feats)).requires_grad_(False).to(device)
            for _idx, feat in enumerate(feats):
                cka, sim_loss = word_embeddings(feat, label_idxs, targets[_idx])
                sim_loss_sum += sim_loss
                log_items["train/sim_loss_C{}".format(_idx+2)] = sim_loss.item()
                log_items["train/CKA_C{}".format(_idx+2)] = cka.item()

        # cross entropy with linearly increasing weight
        # convert warmup from epoch to steps for cross entropy loss
      
        ce_loss_weight = max(1, iteration/ce_warmup_iters) if ce_warmup_iters else 1
        loss = ce_loss_weight*criterion(logits,label_idxs) + sim_loss_sum
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ============ misc ... ============
        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(loss.item(), data['img'].size(0))
        if it % 50 == 0:
            # training accuracy 
            _,argmax = torch.max(logits,1)
            argmax = argmax.data.cpu().numpy()
            label_idxs_ = label_idxs.data.cpu().numpy()
            acc = np.mean(argmax==label_idxs_)*100
            accuracy.update(acc, data['img'].size(0))

            # update the logs and the tensorboard
            log_items['train/loss'] = loss.item()
            log_items['train/acc'] = acc

            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iteration)
            for name,value in log_items.items():
                writer.add_scalar(name, value, iteration)

            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Train Acc. {accuracy.val:.4f} ({accuracy.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    accuracy=accuracy,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
            writer.flush()
        
    return (epoch, losses.avg, accuracy.avg)


def eval_model(model, word_embeddings, dataloader, epoch, writer):
    # Set mode
    #model.eval()
    #if word_embeddings != None:
    #    word_embeddings.eval()

    with torch.no_grad():
        softmax = nn.Softmax(dim=1)
        correct = 0
        seen_correct_per_class = {l: 0 for l in dataloader.dataset.labels}
        sample_per_class = {l: 0 for l in dataloader.dataset.labels}
        
        end = time.time()
        for it,data in enumerate(dataloader): #enumerate(tqdm(dataloader)):
            # Forward pass
            imgs = data['img'].to(device)
            logits,_ = model(imgs)
            gt_labels = data['label']
            prob = softmax(logits)
            prob = prob.data.cpu().numpy()
            prob_zero_seen = np.copy(prob)

            argmax_zero_seen = np.argmax(prob_zero_seen,1)
            for i in range(prob.shape[0]):
                pred_label = dataloader.dataset.labels[argmax_zero_seen[i]]
                gt_label = gt_labels[i]
                sample_per_class[gt_label] += 1
                if gt_label==pred_label:
                    seen_correct_per_class[gt_label] += 1
            

        seen_acc = 0
        num_seen_classes = 0
        for l in dataloader.dataset.labels:
            seen_acc += (seen_correct_per_class[l] / sample_per_class[l])
            num_seen_classes += 1

        val_acc = round(seen_acc*100 / num_seen_classes,4)
        iteration = epoch * len(dataloader)
        writer.add_scalar('val/acc',val_acc,iteration)

        # update the logger
        logger.info(
            "Epoch: [{0}]\t"
            "Validation Acc.: {val_acc:.4f}".format(
                epoch,
                val_acc=val_acc
            )
        )
        writer.flush()

    return val_acc

if __name__ == "__main__":
    main()