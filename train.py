import argparse

import time
import datetime

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root, dino_root
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups

from dino import vision_transformer as vits
from dino_peft import Dino_Model


def train(student, train_loader, unlabelled_train_loader, vanilla_loader, args):

    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3,
    )

    cluster_criterion = DistillLoss(
        args.warmup_teacher_temp_epochs,
        args.epochs,
        args.n_views,
        args.warmup_teacher_temp,
        args.teacher_temp,
    )

    best_all_acc, best_seen_acc, best_novel_acc, best_epoch = 0, 0, 0, 0

    all_class_num = args.mlp_out_dim
    seen_class_num = args.num_labeled_classes

    use_da = args.use_da
    da_scale = args.da_scale
    cur_dist = torch.ones(all_class_num).cuda()
    cur_dist = cur_dist / all_class_num
    prior_dist = torch.ones(all_class_num).cuda()
    prior_dist = prior_dist / all_class_num

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        data_time = AverageMeter()
        img2feat_time = AverageMeter()
        cluster_time = AverageMeter()
        repl_time = AverageMeter()
        backward_time = AverageMeter()

        seen_count = 0
        novel_count = 0

        da = torch.log(cur_dist / prior_dist)
        da = da * da_scale

        student.train()
        one_epoch_start = time.time()
        end = time.time()

        for batch_idx, batch in enumerate(train_loader):
            data_time.update(time.time() - end)
            data_end_time = time.time()

            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]
            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):

                student_proj, student_out = student(images)
                teacher_out = student_out.detach()

                img2feat_time.update(time.time() - data_end_time)
                img2feat_end_time = time.time()

                # clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)
                # clustering, unsup
                cluster_loss, seen_counti, novel_counti = cluster_criterion(student_out, teacher_out, epoch, seen_class_num, use_da, da)
                seen_count += seen_counti
                novel_count += novel_counti
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss

                cluster_time.update(time.time() - img2feat_end_time)
                cluster_end_time = time.time()

                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)
                # representation learning, sup
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                repl_time.update(time.time() - cluster_end_time)
                repl_end_time = time.time()

                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
                loss = 0
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss

            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            backward_time.update(time.time() - repl_end_time)

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                                 .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))
                data_avg = str(datetime.timedelta(seconds=data_time.avg))
                img2feat_avg = str(datetime.timedelta(seconds=img2feat_time.avg))
                cluster_avg = str(datetime.timedelta(seconds=cluster_time.avg))
                repl_avg = str(datetime.timedelta(seconds=cluster_time.avg))
                backward_avg = str(datetime.timedelta(seconds=backward_time.avg))
                args.logger.info('Time: Data Process: {} | Img2Feat: {} | Cluster: {}'.format(data_avg, img2feat_avg, cluster_avg))
                args.logger.info('Time: Representation Learning: {} | Backward: {}'.format(repl_avg, backward_avg))
            end = time.time()

        one_epoch_end = time.time()
        elapsed_time = round(one_epoch_end - one_epoch_start)
        elapsed_time = str(datetime.timedelta(seconds=elapsed_time))
        args.logger.info('Train Epoch: {} | Avg Loss: {:.4f} | Total Train Time: {}'.format(epoch, loss_record.avg, elapsed_time))
        test_start = time.time()
        all_acc, seen_acc, novel_acc, test_seen_count, test_novel_count = test(student, unlabelled_train_loader, epoch=epoch,
                                                               save_name='Train ACC Unlabelled', args=args)
        one_test_time = round(time.time() - test_start)
        one_test_time = str(datetime.timedelta(seconds=one_test_time))

        args.logger.info('Total Test Time: {}'.format(one_test_time))
        args.logger.info('Test Accuracy: All {:.4f} | Seen {:.4f} | Novel {:.4f}'.format(all_acc, seen_acc, novel_acc))
        args.logger.info('Train Sample Count: Seen {} | Novel {} '.format(seen_count, novel_count))
        args.logger.info('Test Sample Count: Seen {} | Novel {}'.format(test_seen_count, test_novel_count))
        if all_acc >= best_all_acc:
            best_all_acc, best_seen_acc, best_novel_acc, best_epoch = all_acc, seen_acc, novel_acc, epoch
        args.logger.info('Best Accuracy: Epoch {} | All {:.4f} | Seen {:.4f} | Novel {:.4f}'.format(best_epoch, best_all_acc, best_seen_acc, best_novel_acc))

        # Updating the distribution alignment vector Only Once (Default)
        if (epoch + 1) >= args.warmup_teacher_temp_epochs:
            epoch_condition = (epoch + 1 - args.warmup_teacher_temp_epochs) % args.da_freq
            if epoch_condition == 0 or (epoch + 1) == args.epochs:
                cur_dist = cap_dist(student, vanilla_loader, args)

        exp_lr_scheduler.step()

def cap_dist(model, dist_loader, args):
    model.eval()
    all_class_num = args.mlp_out_dim
    initial_dist = torch.zeros(all_class_num).cuda()
    initial_dist = initial_dist.reshape(1, all_class_num)
    for batch_idx, (images, label, _, _) in enumerate(tqdm(dist_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            prob = F.softmax(logits / 0.1, dim=-1)
            initial_dist = torch.cat((initial_dist, prob), dim=0)
    cur_dist = torch.mean(initial_dist, dim=0)
    return cur_dist

def test(model, test_loader, epoch, save_name, args):
    model.eval()

    preds, targets = [], []
    mask = np.array([])

    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    seen_num = args.num_labeled_classes
    test_seen_count = np.sum(preds < seen_num)
    test_novel_count = np.sum(preds >= seen_num)

    all_acc, seen_acc, novel_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, seen_acc, novel_acc, test_seen_count, test_novel_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=40, type=int)
    parser.add_argument('--exp_name', default=None, type=str)

    # Modify New Arguments in Our PEFT Framework
    parser.add_argument('--use_vpt', action='store_true', default=False)
    parser.add_argument('--use_relu', action='store_true', default=False)
    parser.add_argument('--use_linear', action='store_true', default=False)
    parser.add_argument('--use_gcdtune', action='store_true', default=False)

    parser.add_argument('--mid_dim', type=int, default=64)
    parser.add_argument('--adapter_scale', type=float, default=0.1)
    parser.add_argument('--partial', type=int, default=12)
    parser.add_argument('--dropout_prob', type=float, default=0.1)
    parser.add_argument('--prompt_len', type=int, default=30)

    parser.add_argument('--use_da', action='store_true', default=False)
    parser.add_argument('--da_scale', type=float, default=0.2)
    parser.add_argument('--da_freq', type=int, default=200)

    parser.add_argument('--seed', type=int, default=0)


    args = parser.parse_args()

    # Utilizing Fixed random seeds 0, 1, and 2 to reproduce the results in the paper
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda:0')

    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    init_experiment(args, runner_name=['LAGCD'], exp_id=0)

    dino_model = vits.__dict__['vit_base']()
    state_dict = torch.load(dino_root, map_location='cpu')
    dino_model.load_state_dict(state_dict)
    backbone = Dino_Model(dino_model, args)

    for name, param in backbone.named_parameters():
        param.requires_grad_(False)
    for name, param in backbone.tuner.named_parameters():
        param.requires_grad_(True)

    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    train_dataset, test_dataset, unlabelled_train_examples_test, vanilla_dataset = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    total_len = label_len + unlabelled_len
    sample_weights = [1 if i < label_len else label_len/unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=total_len)

    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    vanilla_loader = DataLoader(vanilla_dataset, num_workers=args.num_workers, batch_size=256, shuffle=False,
                             pin_memory=False)

    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    model = nn.Sequential(backbone, projector).to(device)

    tuned_params = sum(p.numel() for p in backbone.tuner.parameters())
    print(f'Tunable Model params: {tuned_params}')
    proj_params = sum(p.numel() for p in projector.parameters())
    print(f'Projector params: {proj_params}')

    train(model, train_loader, test_loader_unlabelled, vanilla_loader, args)
