# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# -----------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
 
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
 
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.open_world_eval import OWEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from util.plot_utils import plot_prediction
import matplotlib.pyplot as plt
from copy import deepcopy


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, nc_epoch: int, max_norm: float = 0, wandb: object = None):
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

    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
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
        
        if wandb is not None:
            wandb.log({"total_loss":loss_value})
            wandb.log(loss_dict_reduced_scaled)
            wandb.log(loss_dict_reduced_unscaled)
 
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        
        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

## ORIGINAL FUNCTION
@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args):
    # import ipdb; ipdb.set_trace()
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = OWEvaluator(base_ds, iou_types, args=args)
    
    # === EVAL 메모리 측정 초기화 ===
    import wandb
    from main_open_world import get_memory_mb, get_my_gpu_memory_usage
    import torch
    
    print("=== EVAL 메모리 측정 시작 ===")
    eval_step = 0
    prev_cpu_memory = None  # 이전 스텝의 CPU 메모리 저장
    prev_gpu_memory = None  # 이전 스텝의 GPU 메모리 저장
 
    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
 
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # === EVAL Forward 전 메모리 측정 ===
        before_forward_memory = get_memory_mb()
        before_forward_gpu_usage = get_my_gpu_memory_usage()[0][1] if get_my_gpu_memory_usage() else 0
        before_forward_gpu_allocated = torch.cuda.memory_allocated(device)
        before_forward_gpu_reserved = torch.cuda.memory_reserved(device)
        
        outputs = model(samples)
        
        # === EVAL Forward 후 메모리 측정 ===
        after_forward_memory = get_memory_mb()
        after_forward_gpu_usage = get_my_gpu_memory_usage()[0][1] if get_my_gpu_memory_usage() else 0
        after_forward_gpu_allocated = torch.cuda.memory_allocated(device)
        after_forward_gpu_reserved = torch.cuda.memory_reserved(device)
        after_forward_gpu_max_allocated = torch.cuda.max_memory_allocated(device)
        
        # === 메모리 변화량 계산 ===
        cpu_memory_delta = after_forward_memory['current_total'] - before_forward_memory['current_total']
        gpu_allocated_delta = (after_forward_gpu_allocated - before_forward_gpu_allocated) / 1024 ** 2
        gpu_pynvml_delta = after_forward_gpu_usage - before_forward_gpu_usage
        
        # === 매 100 스텝마다 WandB 로깅 (효율성 향상) ===
        if eval_step % 100 == 0 and wandb is not None and wandb.run is not None:
            wandb.log({
                # 기본 메모리 상태
                "EVAL_CPU_Current": after_forward_memory['current_total'],
                "EVAL_GPU_Allocated_MB": float(after_forward_gpu_allocated / 1024 ** 2),
                "EVAL_GPU_Reserved_MB": float(after_forward_gpu_reserved / 1024 ** 2),
                "EVAL_GPU_Max_Allocated_MB": float(after_forward_gpu_max_allocated / 1024 ** 2),
                "EVAL_GPU_pynvml_MB": float(after_forward_gpu_usage),
                
                # 메모리 변화량 (최적화 효과 확인용)
                "EVAL_CPU_Delta": cpu_memory_delta,
                "EVAL_GPU_Allocated_Delta_MB": gpu_allocated_delta,
                "EVAL_GPU_pynvml_Delta_MB": gpu_pynvml_delta,
                
                # 진행 상황
                "EVAL_Step": eval_step,
                "EVAL_Progress": float(eval_step / len(data_loader) * 100),
            })
            
            # 터미널에도 간단히 출력 (100 스텝마다)
            progress = eval_step / len(data_loader) * 100
            print(f"[Step {eval_step:4d}] 진행률: {progress:5.1f}% | "
                  f"CPU: {after_forward_memory['current_total']:.0f}MB | "
                  f"GPU: {after_forward_gpu_allocated/1024**2:.0f}MB")
        
        eval_step += 1

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)
            
        # === Panoptic 평가 (최적화 전에 처리) ===
        if panoptic_evaluator is not None:
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name
            panoptic_evaluator.update(res_pano)
            
        # === 메모리 최적화 전 측정 ===
        before_optimization_memory = get_memory_mb()
        before_optimization_gpu_allocated = torch.cuda.memory_allocated(device)
        
        # === 메모리 최적화 기법 적용 ===
        # 1. 불필요한 변수 명시적 삭제
        del outputs
        del results
        del res
        del orig_target_sizes
        if 'segm' in postprocessors.keys() or panoptic_evaluator is not None:
            if 'target_sizes' in locals():
                del target_sizes
        if panoptic_evaluator is not None:
            del res_pano
            
        # 2. 파이썬 가비지 컬렉션 강제 실행
        import gc
        gc.collect()
        
        # 3. CUDA 캐시 정리
        torch.cuda.empty_cache()
        
        # === 메모리 최적화 후 측정 ===
        after_optimization_memory = get_memory_mb()
        after_optimization_gpu_allocated = torch.cuda.memory_allocated(device)
        
        # === 최적화 효과 계산 ===
        optimization_cpu_saved = before_optimization_memory['current_total'] - after_optimization_memory['current_total']
        optimization_gpu_saved = (before_optimization_gpu_allocated - after_optimization_gpu_allocated) / 1024 ** 2
        
        # === 최적화 효과 로깅 (매 100 스텝마다) ===
        if eval_step % 100 == 0 and wandb is not None and wandb.run is not None:
            # === 최적화 전후 메모리 값들 ===
            before_opt_cpu = before_optimization_memory['current_total']
            after_opt_cpu = after_optimization_memory['current_total']
            before_opt_gpu = float(before_optimization_gpu_allocated / 1024 ** 2)
            after_opt_gpu = float(after_optimization_gpu_allocated / 1024 ** 2)
            
            # === 스텝 간 변화량 계산 ===
            step_cpu_delta = after_opt_cpu - prev_cpu_memory if prev_cpu_memory is not None else 0.0
            step_gpu_delta = after_opt_gpu - prev_gpu_memory if prev_gpu_memory is not None else 0.0
            
            # === 최적화 효율성 계산 ===
            cpu_efficiency = (optimization_cpu_saved / before_opt_cpu * 100) if before_opt_cpu > 0 else 0
            gpu_efficiency = (optimization_gpu_saved / before_opt_gpu * 100) if before_opt_gpu > 0 else 0
            
            # === WandB 상세 로깅 ===
            wandb.log({
                # === 기본 메모리 상태 ===
                "EVAL_Current_CPU_MB": after_opt_cpu,
                "EVAL_Current_GPU_MB": after_opt_gpu,
                
                # === 최적화 전후 비교 (핵심 메트릭) ===
                "EVAL_Before_Opt_CPU_MB": before_opt_cpu,
                "EVAL_After_Opt_CPU_MB": after_opt_cpu,
                "EVAL_Before_Opt_GPU_MB": before_opt_gpu,
                "EVAL_After_Opt_GPU_MB": after_opt_gpu,
                
                # === 최적화 절약량 (핵심 메트릭) ===
                "EVAL_CPU_Saved_MB": optimization_cpu_saved,
                "EVAL_GPU_Saved_MB": optimization_gpu_saved,
                "EVAL_Total_Saved_MB": optimization_cpu_saved + optimization_gpu_saved,
                
                # === 최적화 효율성 (%) ===
                "EVAL_CPU_Efficiency_Percent": cpu_efficiency,
                "EVAL_GPU_Efficiency_Percent": gpu_efficiency,
                "EVAL_Total_Efficiency_Percent": (cpu_efficiency + gpu_efficiency) / 2,
                
                # === 스텝 간 변화량 (추세 분석용) ===
                "EVAL_Step_CPU_Delta_MB": step_cpu_delta,
                "EVAL_Step_GPU_Delta_MB": step_gpu_delta,
                "EVAL_Step_Total_Delta_MB": step_cpu_delta + step_gpu_delta,
                
                # === 진행 상황 ===
                "EVAL_Step": eval_step,
                "EVAL_Progress_Percent": float(eval_step / len(data_loader) * 100),
                
                # === 누적 통계 (양수인 경우만) ===
                "EVAL_Cumulative_CPU_Saved_MB": max(0, float(optimization_cpu_saved)),
                "EVAL_Cumulative_GPU_Saved_MB": max(0, float(optimization_gpu_saved)),
                
                # === 메모리 상태 분류 ===
                "EVAL_Memory_Status": "Optimized" if (optimization_cpu_saved > 0 or optimization_gpu_saved > 0) else "Stable",
                "EVAL_CPU_Trend": "Decreasing" if optimization_cpu_saved > 0 else ("Increasing" if optimization_cpu_saved < -1 else "Stable"),
                "EVAL_GPU_Trend": "Decreasing" if optimization_gpu_saved > 0 else ("Increasing" if optimization_gpu_saved < -1 else "Stable"),
            })
            
            # === 터미널 출력 (명확하고 구조화된 형태) ===
            progress = eval_step / len(data_loader) * 100
            print(f"\n{'='*70}")
            print(f"[EVAL Step {eval_step:4d}] 진행률: {progress:5.1f}%")
            print(f"{'='*70}")
            
            # 최적화 전후 비교 (핵심 정보)
            print(f"🔧 메모리 최적화 전후 비교:")
            print(f"   ├─ CPU: {before_opt_cpu:6.1f}MB → {after_opt_cpu:6.1f}MB "
                  f"({optimization_cpu_saved:+6.1f}MB, {cpu_efficiency:+5.1f}%)")
            print(f"   └─ GPU: {before_opt_gpu:6.1f}MB → {after_opt_gpu:6.1f}MB "
                  f"({optimization_gpu_saved:+6.1f}MB, {gpu_efficiency:+5.1f}%)")
            
            # 스텝 간 변화량 (추세 분석)
            if prev_cpu_memory is not None and prev_gpu_memory is not None:
                print(f"📈 이전 스텝 대비 변화량:")
                print(f"   ├─ CPU: {step_cpu_delta:+6.1f}MB")
                print(f"   └─ GPU: {step_gpu_delta:+6.1f}MB")
            
            # 전체 절약량 요약
            total_saved = optimization_cpu_saved + optimization_gpu_saved
            if total_saved > 0:
                print(f"💾 총 절약량: {total_saved:+6.1f}MB")
            elif total_saved < -1:
                print(f"⚠️  메모리 증가: {total_saved:+6.1f}MB")
            else:
                print(f"✅ 메모리 안정: {total_saved:+6.1f}MB")
                
            print(f"{'='*70}\n")
            
            # === 다음 스텝을 위해 현재 메모리 저장 ===
            prev_cpu_memory = after_opt_cpu
            prev_gpu_memory = after_opt_gpu

 
    # === 평가 완료 후 메모리 요약 ===
    if wandb is not None and wandb.run is not None:
        final_memory = get_memory_mb()
        final_gpu = torch.cuda.memory_allocated(device) / 1024 ** 2
        
        print("\n" + "="*60)
        print("평가 완료 - 메모리 사용량 요약")
        print("="*60)
        print(f"최종 CPU 메모리: {final_memory['current_total']:.0f}MB")
        print(f"최종 GPU 메모리: {final_gpu:.0f}MB")
        print(f"총 처리 스텝: {eval_step}")
        print("="*60)
        
        # WandB 최종 요약 로깅
        wandb.log({
            "EVAL_Final_CPU_MB": final_memory['current_total'],
            "EVAL_Final_GPU_MB": float(final_gpu),
            "EVAL_Total_Steps": eval_step,
            "EVAL_Status": "Completed"
        })

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        res = coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['metrics']=res
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
 
    
@torch.no_grad()
def get_exemplar_replay(model, exemplar_selection, device, data_loader):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = '[ExempReplay]'
    print_freq = 10
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    image_sorted_scores_reduced={}
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        image_sorted_scores = exemplar_selection(samples, outputs, targets)
        for i in utils.combine_dict(image_sorted_scores):
            image_sorted_scores_reduced.update(i[0])
            
        metric_logger.update(loss=len(image_sorted_scores_reduced.keys()))
        samples, targets = prefetcher.next()
        
    print(f'found a total of {len(image_sorted_scores_reduced.keys())} images')
    return image_sorted_scores_reduced