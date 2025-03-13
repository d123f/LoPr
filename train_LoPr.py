import argparse
import logging
import os
import random
import shutil
import sys
import copy
import time
from itertools import cycle
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from utils_Lopr.ap_semi_strategy import loss_predictor, semi_predictor_wcs
from dataloaders.dataset import BaseDataSets, RandomGenerator,BaseDataSets_easy,BaseDataSets_gen,BaseDataSets_easy_melt,BaseDataSets_ACDC_melt,BaseDataSets_ACDC
from networks.discriminator import FCDiscriminator
from networks.net_factory import net_factory, get_predictor
from utils import losses, metrics, ramps
from utils_Lopr.data_loader import get_data_loader
from utils_Lopr.utils import random_sampling
from torch.cuda.amp import autocast as autocast
from val_2D import test_single_volume,RunningDice,binary_dice,compute_dice,compute_dice_loss
from data_utils_transplant.data_loader import DataGenerator, CropResize, To_Tensor, Trunc_and_Normalize
from data_utils_transplant.transformer_2d import Get_ROI, RandomFlip2D, RandomRotate2D, RandomErase2D, RandomAdjust2D, RandomDistort2D, RandomZoom2D, RandomNoise2D
#################################
from setproctitle import setproctitle

setproctitle("acdclopr 20%") 

torch.cuda.set_device(0)
#################################
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/LoPr', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_LoPr', help='model_name')
parser.add_argument('--fold', type=int,
                    default=1, help='cross validation')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--cross_val', type=bool,
                    default=True, help='5-fold cross validation or random split 7/1/2 for training/validation/testing')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=1e-3,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=str, default="512 512", help='patch size of network input')
parser.add_argument('--seed', type=int,  default=100, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_ratio_init', type=float, default=0.1,
                    help='1/labeled_ratio data is provided mask')
parser.add_argument('--labeled_ratio_max', type=float, default=0.2,
                    help='1/labeled_ratio max data is provided mask')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

parser.add_argument('--scale', nargs=2, type=int,  default=[0,255], help='scale') #[-100, 250]
parser.add_argument('--channels', type=int,  default=1, help='channels') #1
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    
def train(args):
    print(f"--------------------fold is {args.fold}---------------------")
    log_dir="model/{}/{}_of_{}_labeled/log".format(
                args.exp, args.labeled_ratio_init,args.labeled_ratio_max)
    output_dir="model/{}/{}_of_{}_labeled/ckpt".format(
                args.exp, args.labeled_ratio_init,args.labeled_ratio_max)
    semi_save_dir="model/{}/{}_of_{}_labeled/dataset".format(
                args.exp, args.labeled_ratio_init,args.labeled_ratio_max)
    log_dir = os.path.join(log_dir, "fold" + str(args.fold))
    output_dir = os.path.join(output_dir, "fold" + str(args.fold))
    semi_save_dir = os.path.join(semi_save_dir, "fold" + str(args.fold))
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        os.makedirs(log_dir)
    else:
        os.makedirs(log_dir)


    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    else:
        os.makedirs(output_dir)
    if os.path.exists(semi_save_dir):
        shutil.rmtree(semi_save_dir)
        os.makedirs(semi_save_dir)
    else:
        os.makedirs(semi_save_dir)
    segnet_output_dir = os.path.join(output_dir, 'segnet')
    predictor_output_dir = os.path.join(output_dir, 'predictor')
    if not os.path.exists(segnet_output_dir):
        os.makedirs(segnet_output_dir)
    if not os.path.exists(predictor_output_dir):
        os.makedirs(predictor_output_dir)
    writer = SummaryWriter(log_dir)
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    scale = args.scale
    channels = args.channels
    metrics_threshold =0


    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=channels,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    model = create_model()
    predictor = get_predictor(in_chns=channels, class_num=num_classes)
    predictor = predictor.cuda()

    if isinstance(args.patch_size, str):
        args.patch_size = [int(item) for item in args.patch_size.split()]
    split_json = 'ACDC/split28_norm.json'   
    with open(split_json,'r') as fp:
        data_list = json.load(fp)
    train_path = data_list[f'fold{args.fold}']['train_path']
    val_path = data_list[f'fold{args.fold}']['val_path']


    model.train()
    from utils_Lopr.combine_loss import CELabelSmoothingPlusDice
    loss = CELabelSmoothingPlusDice(smoothing=0.1,weight=None,ignore_index=0).cuda()
    predictor_loss = torch.nn.MSELoss().cuda()
    from utils_Lopr.optimizer import get_optimizer,get_lr_scheduler
    seg_optimizer = get_optimizer(model, base_lr)
    predictor_optimizer = get_optimizer(predictor, base_lr)
    scaler = GradScaler()
    predictor_scaler = GradScaler()
    seg_lr_scheduler = get_lr_scheduler(seg_optimizer)
    predictor_lr_scheduler = get_lr_scheduler(predictor_optimizer)
    early_stopping = EarlyStopping(patience=50,verbose=True,delta=1e-3,monitor='val_run_dice',op_type='max')
    unlabeled_data_pool = copy.deepcopy(train_path)
    labeled_data_pool = []
    from utils_Lopr.utils import get_samples_per_epoch
    samples_per_epoch = get_samples_per_epoch(len(train_path),15,args.labeled_ratio_init,args.labeled_ratio_max)
    import queue
    sample_queue = queue.Queue()
    semi_sample_queue = queue.Queue()
    for item in samples_per_epoch[1:]:
            sample_queue.put(item)
            semi_sample_queue.put(int(item * 4))
    val_loader = get_data_loader(val_path,'val',1.0,scale,num_classes,channels,args.patch_size,args.batch_size)


    iter_num = 0
    max_epoch = 500
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch in iterator:
        sample_flag = False
        labeled_data = []
        semi_data = []
        if epoch == 0:
            labeled_data = random_sampling(sample_pool=train_path,init_percent=args.labeled_ratio_init)
        else:
            if early_stopping.counter >= 15 and not sample_queue.empty():
                sample_flag = True 
                early_stopping.counter = 0 
        if sample_flag:
            sample_loader = get_data_loader(unlabeled_data_pool,'val',1.0,scale,num_classes,channels,args.patch_size,args.batch_size)
            sample_nums = sample_queue.get()
            if sample_nums != 0:
                labeled_data = loss_predictor(
                    seg_net=model, 
                    predictor=predictor, 
                    unlabeled_data_pool=unlabeled_data_pool,
                    sample_loader=sample_loader,
                    sample_nums=sample_nums,
                    sample_weight=None,
                    al_mode='lp+wcs',
                    score_type='log_mean')
                print(f'************* finish sampling : {sample_nums} *************')
        if len(labeled_data) != 0:
            labeled_data_pool.extend(labeled_data)
            for sample in labeled_data:
                unlabeled_data_pool.remove(sample)
            random.shuffle(labeled_data_pool)
            train_loader = get_data_loader(labeled_data_pool,'train',1.0,scale,num_classes,channels,args.patch_size,args.batch_size)
     
        if sample_queue.empty() and early_stopping.counter >= 2 * 15 and not semi_sample_queue.empty():
            early_stopping.counter = 0
            sample_loader = get_data_loader(unlabeled_data_pool,'val',1.0,scale,num_classes,channels,args.patch_size,args.batch_size)
            semi_sample_nums = semi_sample_queue.get()
            semi_data = semi_predictor_wcs( 
                    seg_net=model, 
                    predictor=predictor, 
                    unlabeled_data_pool=unlabeled_data_pool,
                    sample_loader=sample_loader,
                    semi_sample_nums=semi_sample_nums,
                    sample_weight=None,
                    al_mode='lp+wcs',  
                    semi_save_dir=os.path.join(semi_save_dir,f'epoch_{epoch}'),
                    score_type='log_mean')
        if len(semi_data) != 0:
            random.shuffle(semi_data)
            train_loader = get_data_loader(labeled_data_pool + semi_data,'train',1.0,scale,num_classes,channels,args.patch_size,args.batch_size)

        train_loss, train_dice, train_run_dice, train_predictor_loss = _train_on_epoch(
                epoch=epoch,
                net=model,
                predictor=predictor,
                criterion=loss,
                predictor_criterion=predictor_loss,
                optimizer=seg_optimizer,
                predictor_optimizer=predictor_optimizer,
                scaler=scaler,
                predictor_scaler=predictor_scaler,
                train_loader=train_loader,
                num_classes=num_classes)

        val_loss, val_dice, val_run_dice, val_predictor_loss = _val_on_epoch(
            epoch=epoch,
            net=model,
            predictor=predictor,
            criterion=loss,
            predictor_criterion=predictor_loss,
            val_loader=val_loader,
            num_classes=num_classes)
        
        seg_lr_scheduler.step()
        if epoch >= 10:
            predictor_lr_scheduler.step()

        print('epoch:{},train_loss:{:.5f},train_predictor_loss:{:.5f},val_loss:{:.5f},val_predictor_loss:{:.5f}'.format(
                epoch, train_loss, train_predictor_loss, val_loss, val_predictor_loss))

        print('epoch:{},train_dice:{:.5f},train_run_dice:{:.5f},val_dice:{:.5f},val_run_dice:{:.5f}'
            .format(epoch, train_dice, train_run_dice[0], val_dice, val_run_dice[0]))
        
        writer.add_scalars('data/seg_loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
        writer.add_scalars('data/predictor_loss', {
            'train': train_predictor_loss,
            'val': val_predictor_loss
        }, epoch)
        writer.add_scalars('data/dice', {
            'train': train_dice,
            'val': val_dice
        }, epoch)
        writer.add_scalars('data/run_dice', {
            'train': train_run_dice[0],
            'val': val_run_dice[0]
        }, epoch)

        writer.add_scalars('data/dice_dataratio', {
            'train': train_run_dice[0],
            'val': val_run_dice[0],
            'data_ratio':len(labeled_data_pool)/len(train_path),
        }, epoch)

        writer.add_scalar('data/lr', seg_optimizer.param_groups[0]['lr'], epoch)

        early_stopping(val_run_dice[0])

        if val_run_dice[0] > metrics_threshold or epoch % 50 == 0:
            if val_run_dice[0] > metrics_threshold:
                metrics_threshold = val_run_dice[0]

            state_dict = model.state_dict()
            predictor_state_dict = predictor.state_dict()

            saver = {
                'epoch': epoch,
                'save_dir': segnet_output_dir,
                'state_dict': state_dict,
                #'optimizer':seg_optimizer.state_dict()
            }
            predictor_saver = {
                'epoch': epoch,
                'save_dir': predictor_output_dir,
                'state_dict': predictor_state_dict,
                #'optimizer':predictor_optimizer.state_dict()
            }

            file_name = 'epoch={}-train_loss={:.5f}-train_dice={:.5f}-train_run_dice={:.5f}-val_loss={:.5f}-val_dice={:.5f}-val_run_dice={:.5f}.pth'.format(
                epoch, train_loss, train_dice, train_run_dice[0], val_loss, val_dice, val_run_dice[0])
            predictor_file_name = 'epoch={}-train_predictor_loss={:.5f}-val_predictor_loss={:.5f}.pth'.format(
                epoch, train_predictor_loss, val_predictor_loss)
            save_path = os.path.join(segnet_output_dir, file_name)
            predictor_save_path = os.path.join(predictor_output_dir, predictor_file_name)
            print("SegNet save as: %s" % file_name)
            print("Predictor save as: %s" % predictor_file_name)

            torch.save(saver, save_path)
            torch.save(predictor_saver, predictor_save_path)

        #early stopping
        if early_stopping.early_stop:
            print("Early stopping")
            break

        writer.close()
class EarlyStopping(object):
    """Early stops the training if performance doesn't improve after a given patience."""
    def __init__(self,
                 patience=10,
                 verbose=True,
                 delta=0,
                 monitor='val_loss',
                 op_type='min'):
        """
        Args:
            patience (int): How long to wait after last time performance improved.
                            Default: 10
            verbose (bool): If True, prints a message for each performance improvement. 
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            monitor (str): Monitored variable.
                            Default: 'val_loss'
            op_type (str): 'min' or 'max'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.monitor = monitor
        self.op_type = op_type

        if self.op_type == 'min':
            self.val_score_min = np.Inf
        else:
            self.val_score_min = 0

    def __call__(self, val_score):

        score = -val_score if self.op_type == 'min' else val_score

        if self.best_score is None:
            self.best_score = score
            self.print_and_update(val_score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.print_and_update(val_score)
            self.counter = 0

    def print_and_update(self, val_score):
        '''print_message when validation score decrease.'''
        if self.verbose:
            print(
                self.monitor,
                f'optimized ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...'
            )
        self.val_score_min = val_score
class AverageMeter(object):
    '''
  Computes and stores the average and current value
  '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def _train_on_epoch(epoch,
                    net,
                    predictor,
                    criterion,
                    predictor_criterion,
                    optimizer,
                    predictor_optimizer,
                    scaler,
                    predictor_scaler,
                    train_loader=None,
                    num_classes=None,):
         
        net.train()
        predictor.train()
        train_loss = AverageMeter()
        train_dice = AverageMeter()
        train_predictor_loss = AverageMeter()

        from utils_Lopr.metrics import RunningDice
        run_dice = RunningDice(labels=range(num_classes), ignore_label=-1)

        for step, sample in enumerate(train_loader):

            # train seg net
            data = sample['image']  #N1HW
            target = sample['label']  #NCHW

            data = data.cuda()
            target = target.cuda()

            with autocast(True):
                output = net(data)
                loss = criterion(output, target)

                if isinstance(output, tuple):
                    output = output[0]  #NCHW

            optimizer.zero_grad()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ## train predictor
            if epoch >= 10:  

                predictor_target = torch.from_numpy(compute_dice_loss(output.detach(),target,ignore_index=-1,reduction=None)).cuda()  #NC
                predictor_data = torch.stack([torch.cat((i, j), dim=0) for i, j in zip(data.clone().detach(),torch.softmax(output.clone().detach(), dim=1))],dim=0)  #N,C+1,H,W
                with autocast(True):
                    predictor_output = predictor(predictor_data)
                    predictor_loss = predictor_criterion(predictor_output, predictor_target)
                
                predictor_output = predictor_output.mean(dim=0).detach().cpu().numpy().tolist()

                predictor_optimizer.zero_grad()
                
                predictor_scaler.scale(predictor_loss).backward()
                predictor_scaler.step(predictor_optimizer)
                predictor_scaler.update()


            else:
                predictor_target = torch.from_numpy(np.ones((data.size(0),num_classes),dtype=np.float32)*-1.0) #NC
                predictor_output = [-1.0]*num_classes
                predictor_loss = torch.tensor(-1.0).cuda()

            output = output.float()
            loss = loss.float()
            predictor_loss = predictor_loss.float()

            # measure dice and record loss
            dice = compute_dice(output.detach(), target)
            train_loss.update(loss.item(), data.size(0))
            train_dice.update(dice, data.size(0))
            train_predictor_loss.update(predictor_loss.item(), data.size(0))

            # measure run dice
            output = torch.argmax(torch.softmax(output, dim=1),1).detach().cpu().numpy()  #N*H*W
            target = torch.argmax(target, 1).detach().cpu().numpy()
            run_dice.update_matrix(target, output)

            if step % 2 == 0:
                rundice, dice_list = run_dice.compute_dice()
                print("Category Dice: ", np.round(predictor_target.cpu().numpy().mean(axis=0),4))
                print("Predicted Dice: ", np.round(np.array(predictor_output),4))
                print(
                    'epoch:{},step:{},train_loss:{:.5f},train_dice:{:.5f},train_run_dice:{:.5f},train_predictor_loss:{:.5f},lr:{:.5f}'
                    .format(epoch, step, loss.item(), dice, rundice,
                            predictor_loss.item(),
                            optimizer.param_groups[0]['lr']))
        return train_loss.avg, train_dice.avg, run_dice.compute_dice(), train_predictor_loss.avg


def _val_on_epoch(epoch,
                net,
                predictor,
                criterion,
                predictor_criterion,
                val_loader=None,
                num_classes=2):

        net.eval()
        predictor.eval()

        val_loss = AverageMeter()
        val_dice = AverageMeter()
        val_predictor_loss = AverageMeter()

        from utils_Lopr.metrics import RunningDice
        run_dice = RunningDice(labels=range(num_classes), ignore_label=-1)

        with torch.no_grad():
            for step, sample in enumerate(val_loader):
                data = sample['image']
                target = sample['label']

                data = data.cuda()
                target = target.cuda()
                with autocast(True):
                    output = net(data)
                    loss = criterion(output, target)

                    if isinstance(output, tuple):
                        output = output[0]
                   

                if epoch >= 10:
                    predictor_target = torch.from_numpy(compute_dice_loss(output.detach(),target,ignore_index=-1,reduction=None)).cuda()  #NC
                    predictor_data = torch.stack([torch.cat((i, j), dim=0) for i, j in zip(data.clone().detach(),torch.softmax(output.clone().detach(), dim=1))],dim=0)  #N,C+1,H,W

                    with autocast(True):
                        predictor_output = predictor(predictor_data)
                        predictor_loss = predictor_criterion(predictor_output, predictor_target)
                    
                    predictor_output = predictor_output.mean(dim=0).detach().cpu().numpy().tolist()
                else:
                    # predictor_target = torch.from_numpy(np.ones(data.size(0),self.num_classes)*-1.0)
                    predictor_target = torch.from_numpy(np.ones((data.size(0),num_classes),dtype=np.float32)*-1.0) #NC
                    predictor_output = [-1.0]*num_classes
                    predictor_loss = torch.tensor(-1.0).cuda()

                output = output.float()
                loss = loss.float()
                predictor_loss = predictor_loss.float()

                # measure dice and record loss
                dice = compute_dice(output.detach(), target)
                val_loss.update(loss.item(), data.size(0))
                val_dice.update(dice, data.size(0))
                val_predictor_loss.update(predictor_loss.item(), data.size(0))

                # measure run dice
                output = torch.argmax(torch.softmax(output, dim=1),1).detach().cpu().numpy()  #N*H*W
                target = torch.argmax(target, 1).detach().cpu().numpy()
                run_dice.update_matrix(target, output)

                # torch.cuda.empty_cache()

                if step % 2 == 0:
                    rundice, dice_list = run_dice.compute_dice()
                    print("Category Dice: ", np.round(predictor_target.cpu().numpy().mean(axis=0),4))
                    print("Predicted Dice: ", np.round(np.array(predictor_output),4))
                    print(
                        'epoch:{},step:{},val_loss:{:.5f},val_dice:{:.5f},val_run_dice:{:.5f},val_predictor_loss:{:.5f}'
                        .format(epoch, step, loss.item(), dice, rundice, predictor_loss.item()))
                    # run_dice.init_op()

        # return val_loss.avg,run_dice.compute_dice()[0]
        return val_loss.avg, val_dice.avg, run_dice.compute_dice(), val_predictor_loss.avg

if __name__ == "__main__":
    for i in range(5):
        args.fold = i + 1
        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        train(args)
