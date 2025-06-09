import wandb
import torch
import numpy as np
import random
import argparse
import time
import json
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
#GPU process인식을 위한 작업
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import sys
import datasets

from typing import Iterable
from copy import deepcopy
from pathlib import Path
from torch.utils.data import DataLoader
from resource import getrusage, RUSAGE_CHILDREN, RUSAGE_SELF

import util.misc as utils

from torch.cuda import memory_summary

from models import build_model
from engine import evaluate
from datasets.coco import make_coco_transforms
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.data_prefetcher import data_prefetcher
from datasets.torchvision_datasets.open_world import OWDetection
from main_open_world import get_args_parser

# from gpustat import GPUStatCollection
# def get_my_gpu_usage():
#     stats = GPUStatCollection.new_query()
#     my_pid = os.getpid()
#     usages = []
#     for gpu in stats.gpus:
#         print(gpu.processes)
#         for proc in gpu.processes:
#             # proc: {'pid': int, 'gpu_memory_usage': int, ...}
#             if proc['pid'] == my_pid:
#                 usages.append((gpu.index, proc['gpu_memory_usage']))
#     return usages
import pynvml , pandas as pd
import psutil

def get_my_gpu_memory_usage():
    pynvml.nvmlInit()
    result = []
    print('os.getpid : ', os.getpid())
    print('pustil.process : ', psutil.Process(os.getpid()).ppid())
    for dev_id in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            print(proc.pid, proc.usedGpuMemory, dev_id)
            result.append([proc.pid, proc.usedGpuMemory, dev_id])
    gpu_usage = pd.DataFrame(result,columns=["pid","bytes of memory", "device"])
    gpu_usage["MB of memory"] = gpu_usage["bytes of memory"] / (1024*1024)
    gpu_usage["GB of memory"] = gpu_usage["bytes of memory"] / (1024*1024*1024)

    gpu_usage_by_id_r = gpu_usage.groupby("pid").apply(lambda x : ", ".join([str(i) for i in x["device"].tolist()])).reset_index(drop=False)
    gpu_usage_by_id_r.columns = ["pid","device_list"]
    gpu_usage_by_id_l = gpu_usage.groupby("pid").agg({"MB of memory" : "sum","GB of memory" : "sum",}).reset_index(drop=False)
    gpu_usage_by_id = gpu_usage_by_id_l.merge(gpu_usage_by_id_r,on="pid",how="left")

    print("▶ GPU Memory Usage (PID : MB)")
    for pid, mem in gpu_usage.groupby("pid")["MB of memory"].sum().items():
        mark = "<- current" if pid == os.getpid() else ""
        print(f"{pid} : {mem:.2f} MB {mark}")

    print('os.getpid : ', os.getpid())
    print('pustil.process : ', psutil.Process(os.getpid()).ppid())
    return gpu_usage["MB of memory"], gpu_usage_by_id

def config_init():
    parser = argparse.ArgumentParser('PROB_LOGGING script', parents=[get_args_parser()])
    args = parser.parse_args(args = []) # args = [] : fix the error https://chaeso-coding.tistory.com/113
    args.epochs = 1
    args.batch_size = 1
    args.distributed = False

    #########################
    #########################
    print('default : ', args.num_feature_levels )
    args.num_feature_levels  = 4
    print('changed : ', args.num_feature_levels )
    #########################
    #########################

    # W&B 초기화
    wandb.init(
        project="PROB_CHU",  # 프로젝트 이름 설정
        config = args
    )

    wandb.run.name = 'python_file_test'
    wandb.run.save()

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    return args

def build_prob_model(args):

    device = torch.device(args.device)
    if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            
    model, criterion, postprocessors, exemplar_selection = build_model(args, mode = args.model_type)
    model.to(device) # create_model

    model_without_ddp = model
    print(model_without_ddp)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    args.n_parameters = n_parameters
    wandb.config.update(vars(args))

    return model, criterion, postprocessors, exemplar_selection, device, n_parameters

def get_datasets(args):

    # Ready for first task 0 (Initial training)

    args.train_set = 'owod_t1_train' # data/OWOD/ImageSets/TOWOD/owod_t1_train.txt
    args.test_set = 'owod_all_task_test' # data/OWOD/ImageSets/TOWOD/owod_all_task_test.txt
    args.dataset = 'TOWOD' # data/OWOD/ImageSets/TOWOD

    print(args.dataset)

    train_set = args.train_set
    test_set = args.test_set
    dataset_train = OWDetection(args, args.data_root, image_set=args.train_set, transforms=make_coco_transforms(args.train_set), dataset = args.dataset)
    dataset_val = OWDetection(args, args.data_root, image_set=args.test_set, dataset = args.dataset, transforms=make_coco_transforms(args.test_set))

    print(args.train_set)
    print(args.test_set)
    print(dataset_train)
    print(dataset_val)
    return dataset_train, dataset_val


# CPU 메모리 측정
def get_memory_mb():
    """
    Get the memory usage of the current process and its children.

    Returns:
        dict: A dictionary containing the memory usage of the current process and its children.

        The dictionary has the following keys:
            - self: The memory usage of the current process.
            - children: The memory usage of the children of the current process.
            - total: The total memory usage of the current process and its children.
    """
    res = {
        "self": getrusage(RUSAGE_SELF).ru_maxrss / 1024,
        "children": getrusage(RUSAGE_CHILDREN).ru_maxrss / 1024,
        "total": getrusage(RUSAGE_SELF).ru_maxrss / 1024 + getrusage(RUSAGE_CHILDREN).ru_maxrss / 1024
    }
    return res

def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, nc_epoch: int, max_norm: float = 0, wandb: object = None):
    
    gpu_memory_history = []
    
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    for _idx, _ in enumerate(metric_logger.log_every(range(len(data_loader)), print_freq, header)):
        get_my_gpu_memory_usage()
        outputs = model(samples)
        loss_dict = criterion(outputs, targets) 
        weight_dict = deepcopy(criterion.weight_dict)
        
        ## condition for starting nc loss computation after certain epoch so that the F_cls branch has the time
        ## to learn the within classes seperation.
        if epoch < nc_epoch: 
            for k,v in weight_dict.items():
                if 'NC' in k:
                    weight_dict[k] = 0
         
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        ## Just printing NOt affectin gin loss function
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
 
        loss_value = losses_reduced_scaled.item()
 
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
 
        optimizer.zero_grad()
        losses.backward()
        
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        '''

        # 다 MB 단위
        cpu_res = get_memory_mb()['total']
        gpu_memory_history.append(print_gpu_utilization())
        # 현재 디바이스 설정
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 현재 할당된 메모리 (바이트 단위)
        allocated = torch.cuda.memory_allocated(device)

        # 현재 예약된 메모리 (바이트 단위)
        reserved = torch.cuda.memory_reserved(device)

        # 최대 할당된 메모리 (바이트 단위)
        max_allocated = torch.cuda.max_memory_allocated(device)

        # 최대 예약된 메모리 (바이트 단위)
        max_reserved = torch.cuda.max_memory_reserved(device)
        print("Used CPU memory : ", cpu_res, ' MB')

        wandb.log({
            "Used CPU memory" : cpu_res,
            'Allocated GPU memory' : float(f"{allocated / 1024 ** 2:.2f}"),
            "Reserved GPU memory" : float(f"{reserved / 1024 ** 2:.2f}"),
            "MAX Allocated GPU memory" : float(f"{max_allocated / 1024 ** 2:.2f}"),
            "MAX Reserved GPU memory" : float(f"{max_reserved / 1024 ** 2:.2f}"),
            "gpustat GPU memory" : float(gpu_memory_history[-1]) 
        })
        
        # if wandb is not None:
        #     wandb.log({"total_loss":loss_value})
        #     wandb.log(loss_dict_reduced_scaled)
        #     wandb.log(loss_dict_reduced_unscaled)
        '''
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        
        
        samples, targets = prefetcher.next()


        if _idx == 100 :
            gpu_memory_history = np.array(gpu_memory_history)
            print('평균 사용량', np.mean(gpu_memory_history))
            print('최대 사용량', np.max(gpu_memory_history))
            print('최소 사용량', np.min(gpu_memory_history))
            break
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    args = config_init()
    model, criterion, postprocessors, exemplar_selection, device, n_parameters = build_prob_model(args)
    dataset_train, dataset_val = get_datasets(args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                    collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                    pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                    pin_memory=True)
    
    model_without_ddp = model

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                    if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                        weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module

    if args.dataset == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    elif args.dataset == "coco":
        base_ds = get_coco_api_from_dataset(dataset_val)
    else:
        base_ds = dataset_val

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)

    if args.pretrain:
        print('Initialized from the pre-training model')
        checkpoint = torch.load(args.pretrain, map_location='cpu')
        state_dict = checkpoint['model']
        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        print(msg)
        args.start_epoch = checkpoint['epoch'] + 1
        if args.eval:
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args)
            # return
        
        
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        # check the resumed model
        if (not args.eval and not args.viz and args.dataset in ['coco', 'voc']):
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args
            )
        if args.eval:
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args)
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            # return
        
    if args.freeze_prob_model:           
        if isinstance(model_without_ddp.prob_obj_head, torch.nn.ModuleList):
            for obj_head in model_without_ddp.prob_obj_head:
                obj_head.freeze_prob_model()
        else:
            model_without_ddp.prob_obj_head.freeze_prob_model()
            
        obj_bn_mean_before=model_without_ddp.prob_obj_head[0].objectness_bn.running_mean

    args.distributed = False
    print(f'Start training from epoch {args.start_epoch} to {args.epochs}')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
            
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.nc_epoch, args.clip_max_norm, wandb)
            
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch % args.eval_every == 0 or epoch == 0 or epoch == 1 or (args.epochs-epoch)<1):
                test_stats, coco_evaluator = evaluate(
                    model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args)
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                if wandb is not None:
                    test_stats["metrics"]['epoch']=epoch
                    # wandb.log({str(key): val for key, val in test_stats["metrics"].items()})
            elif epoch > args.epochs-6:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            else:
                    test_stats = {}
                    
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        try:        
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}
        except:
            log_stats = {'None' : 'None'}
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if args.dataset in ['owod', 'owdetr'] and epoch % args.eval_every == 0 and epoch > 0:
                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    output_dir / "eval" / name)