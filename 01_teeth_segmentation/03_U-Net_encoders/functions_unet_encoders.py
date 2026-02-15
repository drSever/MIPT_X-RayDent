##########################################################################################
# Функции для решения задачи сегментации зубов с использованием U-Net с предобученными энкодерами
##########################################################################################

import os
from google.colab import drive
import zipfile
from pathlib import Path
import yaml
import json
from tqdm import tqdm
from datetime import datetime
import random
import traceback

import cv2
from PIL import Image
import numpy as np
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
from scipy import ndimage

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

import matplotlib.pyplot as plt
import seaborn as sns

### Фиксация SEED ###

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

### Создание модели с предобученным энкодером ###

def create_model(architecture, encoder_name, encoder_weights, in_channels, classes):
    """Создает модель с предобученным энкодером"""
    if architecture == 'unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
            encoder_depth=5,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_use_batchnorm=True,
            decoder_attention_type=None
        )
    elif architecture == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
            encoder_depth=5,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_use_batchnorm=True,
            decoder_attention_type='scse'
        )
    elif architecture == 'fpn':
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
            encoder_depth=5,
            decoder_pyramid_channels=256,
            decoder_segmentation_channels=128,
            decoder_dropout=0.0
        )
    elif architecture == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
            encoder_depth=5,
            encoder_output_stride=16,
            decoder_channels=256,
            decoder_atrous_rates=(12, 24, 36)
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model

### Датасет ###

class TeethSegmentationDataset(Dataset):
    def __init__(self, data_dir, img_size, split='train', transform=None):
        """
        Датасет для instance-сегментации зубов, разметка в формате YOLO
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.img_size = img_size

        with open(self.data_dir / 'data.yaml', 'r') as f:
            self.config = yaml.safe_load(f)

        self.num_classes = self.config['nc'] + 1
        self.class_names = ['Background'] + self.config['names']

        self.images_dir = self.data_dir / split / 'images'
        self.labels_dir = self.data_dir / split / 'labels'

        self.image_files = list(self.images_dir.glob('*.jpg')) + \
                          list(self.images_dir.glob('*.png')) + \
                          list(self.images_dir.glob('*.jpeg'))

        print(f"Загружено {len(self.image_files)} изображений для {split}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {img_path}")

        original_h, original_w = image.shape
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = self.load_yolo_mask(img_path, original_w, original_h)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).float()
                if len(image.shape) == 3 and image.shape[2] == 1:
                    image = image.permute(2, 0, 1)
                elif len(image.shape) == 2:
                    image = image.unsqueeze(0)

            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).long()
            else:
                mask = mask.long()
        else:
            if len(image.shape) == 3 and image.shape[2] == 1:
                image = image.squeeze(2)
            image = torch.from_numpy(image).float().unsqueeze(0)
            mask = torch.from_numpy(mask).long()

        if isinstance(image, torch.Tensor):
            if len(image.shape) == 2:
                image = image.unsqueeze(0)

        if isinstance(mask, torch.Tensor):
            mask = mask.long()

        return image, mask

    def load_yolo_mask(self, img_path, original_w, original_h):
        """Загружает маску сегментации из YOLO формата"""
        label_path = self.labels_dir / (img_path.stem + '.txt')
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        if not label_path.exists():
            return mask

        scale_x = self.img_size / original_w
        scale_y = self.img_size / original_h

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            class_id = int(parts[0]) + 1
            coords = list(map(float, parts[1:]))

            polygon_points = []
            for i in range(0, len(coords), 2):
                x_orig = coords[i] * original_w
                y_orig = coords[i+1] * original_h
                x = int(x_orig * scale_x)
                y = int(y_orig * scale_y)
                x = max(0, min(x, self.img_size - 1))
                y = max(0, min(y, self.img_size - 1))
                polygon_points.append([x, y])

            if len(polygon_points) >= 3:
                polygon_points = np.array(polygon_points, dtype=np.int32)
                cv2.fillPoly(mask, [polygon_points], class_id)

        return mask

    def get_class_weights(self):
        """Вычисляет веса классов для борьбы с дисбалансом"""
        class_counts = np.zeros(self.num_classes)
        print(f"Вычисляем веса для {self.num_classes} классов...")

        sample_indices = range(0, len(self), max(1, len(self) // 100))

        for idx in sample_indices:
            img_path = self.image_files[idx]
            temp_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if temp_img is None:
                continue
            original_h, original_w = temp_img.shape
            mask = self.load_yolo_mask(img_path, original_w, original_h)

            unique, counts = np.unique(mask, return_counts=True)
            for cls, count in zip(unique, counts):
                if cls < self.num_classes:
                    class_counts[cls] += count

        total_pixels = class_counts.sum()
        weights = total_pixels / (class_counts + 1e-8)
        weights = weights / weights.sum() * len(weights)

        return torch.FloatTensor(weights)

### Loss Functions ###

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        target = target.long()
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()

        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        valid_classes = target_one_hot.sum(dim=(0, 2, 3)) > 0
        if valid_classes.sum() > 0:
            dice = dice[:, valid_classes]

        return 1.0 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        target = target.long()
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0, focal_weight=1.0,
                 class_weights=None, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss(smooth=smooth)
        self.focal_loss = FocalLoss()

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)

        total_loss = (self.ce_weight * ce +
                     self.dice_weight * dice +
                     self.focal_weight * focal)

        return total_loss, {
            'ce_loss': ce.detach().item(),
            'dice_loss': dice.detach().item(),
            'focal_loss': focal.detach().item(),
            'total_loss': total_loss.detach().item()
        }

def get_loss_function(loss_type='combined', num_classes=33, class_weights=None,
                     ce_weight=1.0, dice_weight=1.0, focal_weight=1.0):
    """Фабрика функций потерь"""
    if loss_type == 'combined':
        return CombinedLoss(
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            focal_weight=focal_weight,
            class_weights=class_weights
        )
    elif loss_type == 'ce':
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'focal':
        return FocalLoss()
    else:
        raise ValueError(f"Неизвестный тип функции потерь: {loss_type}")

### Метрики  ###

def dice_coefficient(pred, target, num_classes, smooth=1e-6, exclude_background=False):
    target = target.long().to(pred.device)
    pred_softmax = F.softmax(pred, dim=1)
    pred_binary = torch.argmax(pred_softmax, dim=1)
    pred_binary_one_hot = F.one_hot(pred_binary, num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    intersection = (pred_binary_one_hot * target_one_hot).sum(dim=(0, 2, 3))
    pred_sum = pred_binary_one_hot.sum(dim=(0, 2, 3))
    target_sum = target_one_hot.sum(dim=(0, 2, 3))

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    dice_scores = dice.cpu().numpy().tolist()

    if exclude_background and num_classes > 1:
        return dice_scores[1:]
    return dice_scores

def iou_score(pred, target, num_classes, smooth=1e-6, exclude_background=False):
    target = target.long().to(pred.device)
    pred_softmax = F.softmax(pred, dim=1)
    pred_binary = torch.argmax(pred_softmax, dim=1)
    pred_binary_one_hot = F.one_hot(pred_binary, num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    intersection = (pred_binary_one_hot * target_one_hot).sum(dim=(0, 2, 3))
    pred_sum = pred_binary_one_hot.sum(dim=(0, 2, 3))
    target_sum = target_one_hot.sum(dim=(0, 2, 3))
    union = pred_sum + target_sum - intersection

    iou = (intersection + smooth) / (union + smooth)
    iou_scores = iou.cpu().numpy().tolist()

    if exclude_background and num_classes > 1:
        return iou_scores[1:]
    return iou_scores

def pixel_accuracy(pred, target):
    target = target.long().to(pred.device)
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy.item()

class SegmentationMetrics:
    """Класс для накопления и вычисления метрик сегментации"""
    def __init__(self, num_classes, class_names=None, exclude_background=True):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.exclude_background = exclude_background
        self.reset()

    def reset(self):
        self.intersection_sum = np.zeros(self.num_classes)
        self.pred_sum_sum = np.zeros(self.num_classes)
        self.target_sum_sum = np.zeros(self.num_classes)
        self.total_accuracy = 0.0
        self.total_samples = 0
        self.total_tp = np.zeros(self.num_classes)
        self.total_fp = np.zeros(self.num_classes)
        self.total_fn = np.zeros(self.num_classes)
        self.accumulated_predictions = []
        self.accumulated_targets = []
        self.map50 = 0.0
        self.map50_95 = 0.0

    def update(self, pred, target):
        batch_size = pred.size(0)
        target = target.long().to(pred.device)

        pred_softmax = F.softmax(pred, dim=1)
        pred_binary = torch.argmax(pred_softmax, dim=1)
        pred_binary_one_hot = F.one_hot(pred_binary, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        intersection = (pred_binary_one_hot * target_one_hot).sum(dim=(0, 2, 3))
        pred_sum = pred_binary_one_hot.sum(dim=(0, 2, 3))
        target_sum = target_one_hot.sum(dim=(0, 2, 3))

        self.intersection_sum += intersection.detach().cpu().numpy()
        self.pred_sum_sum += pred_sum.detach().cpu().numpy()
        self.target_sum_sum += target_sum.detach().cpu().numpy()

        acc = pixel_accuracy(pred, target)
        self.total_accuracy += acc * batch_size

        pred_classes = torch.argmax(pred_softmax, dim=1)
        for cls in range(self.num_classes):
            pred_cls = (pred_classes == cls)
            target_cls = (target == cls)
            tp = (pred_cls & target_cls).sum().item()
            fp = (pred_cls & ~target_cls).sum().item()
            fn = (~pred_cls & target_cls).sum().item()
            self.total_tp[cls] += tp
            self.total_fp[cls] += fp
            self.total_fn[cls] += fn

        if len(self.accumulated_predictions) < 20:
            if len(self.accumulated_predictions) < 5 or batch_size % 5 == 0:
                self.accumulated_predictions.append(pred.detach().cpu())
                self.accumulated_targets.append(target.detach().cpu())

        self.total_samples += batch_size

    def compute(self):
        if self.total_samples == 0:
            return {}

        smooth = 1e-6
        dice_denominator = self.pred_sum_sum + self.target_sum_sum
        mean_dice = (2.0 * self.intersection_sum + smooth) / (dice_denominator + smooth)
        union = self.pred_sum_sum + self.target_sum_sum - self.intersection_sum
        mean_iou = (self.intersection_sum + smooth) / (union + smooth)
        pixel_acc = self.total_accuracy / self.total_samples

        class_precision = np.zeros(self.num_classes)
        class_recall = np.zeros(self.num_classes)
        class_f1 = np.zeros(self.num_classes)

        for cls in range(self.num_classes):
            tp = self.total_tp[cls]
            fp = self.total_fp[cls]
            fn = self.total_fn[cls]
            class_precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            class_recall[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if class_precision[cls] + class_recall[cls] > 0:
                class_f1[cls] = 2 * class_precision[cls] * class_recall[cls] / (class_precision[cls] + class_recall[cls])

        total_tp = self.total_tp.sum()
        total_fp = self.total_fp.sum()
        total_fn = self.total_fn.sum()
        precision_micro = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall_micro = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

        classes_present = (self.total_tp + self.total_fn) > 0

        if self.exclude_background and self.num_classes > 1:
            valid_classes_mask = classes_present[1:]
            if valid_classes_mask.any():
                overall_mean_dice = np.mean(mean_dice[1:][valid_classes_mask])
                overall_mean_iou = np.mean(mean_iou[1:][valid_classes_mask])
                precision_macro = np.mean(class_precision[1:][valid_classes_mask])
                recall_macro = np.mean(class_recall[1:][valid_classes_mask])
                f1_macro = np.mean(class_f1[1:][valid_classes_mask])
            else:
                overall_mean_dice = overall_mean_iou = precision_macro = recall_macro = f1_macro = 0.0
        else:
            if classes_present.any():
                overall_mean_dice = np.mean(mean_dice[classes_present])
                overall_mean_iou = np.mean(mean_iou[classes_present])
                precision_macro = np.mean(class_precision[classes_present])
                recall_macro = np.mean(class_recall[classes_present])
                f1_macro = np.mean(class_f1[classes_present])
            else:
                overall_mean_dice = overall_mean_iou = precision_macro = recall_macro = f1_macro = 0.0

        return {
            'pixel_accuracy': pixel_acc,
            'mean_dice': overall_mean_dice,
            'mean_iou': overall_mean_iou,
            'mean_dice_with_bg': np.mean(mean_dice[classes_present]) if classes_present.any() else 0.0,
            'mean_iou_with_bg': np.mean(mean_iou[classes_present]) if classes_present.any() else 0.0,
            'map50': self.map50,
            'map50_95': self.map50_95,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'class_dice': {name: score for name, score in zip(self.class_names, mean_dice)},
            'class_iou': {name: score for name, score in zip(self.class_names, mean_iou)},
            'class_precision': {name: score for name, score in zip(self.class_names, class_precision)},
            'class_recall': {name: score for name, score in zip(self.class_names, class_recall)},
            'class_f1': {name: score for name, score in zip(self.class_names, class_f1)}
        }

    def compute_map_metrics(self):
        if not self.accumulated_predictions or not self.accumulated_targets:
            self.map50 = 0.0
            self.map50_95 = 0.0
            return

        try:
            all_predictions = torch.cat(self.accumulated_predictions, dim=0)
            all_targets = torch.cat(self.accumulated_targets, dim=0)
            map_results = calculate_map50_map95(all_predictions, all_targets, self.num_classes, self.exclude_background)
            self.map50 = map_results['mAP50']
            self.map50_95 = map_results['mAP50_95']
        except Exception as e:
            print(f"Ошибка при вычислении mAP: {e}")
            self.map50 = 0.0
            self.map50_95 = 0.0

        self.accumulated_predictions.clear()
        self.accumulated_targets.clear()

    def print_results(self):
        results = self.compute()
        print(f"Pixel Accuracy: {results['pixel_accuracy']:.4f}")
        if self.exclude_background:
            print(f"Mean Dice (без фона): {results['mean_dice']:.4f}")
            print(f"Mean IoU (без фона): {results['mean_iou']:.4f}")
        print(f"mAP@0.5: {results['map50']:.4f}")
        print(f"mAP@0.5:0.95: {results['map50_95']:.4f}")

def calculate_map_segmentation(pred, target, num_classes, iou_thresholds=[0.5], exclude_background=True):
    """
    Вычисляет mAP для сегментации на основе IoU thresholds
    Args:
        pred: предсказания модели [B, C, H, W]
        target: истинные маски [B, H, W]
        num_classes: количество классов
        iou_thresholds: список IoU порогов для вычисления AP
        exclude_background: исключить фон (класс 0) из расчета
    Returns:
        dict: словарь с mAP метриками для каждого порога
    """
    if pred.dim() != 4 or target.dim() != 3:
        raise ValueError(f"Ожидаемые размерности: pred [B,C,H,W], target [B,H,W]. Получено: pred {pred.shape}, target {target.shape}")

    target = target.long().to(pred.device)

    if target.min() < 0 or target.max() >= num_classes:
        raise ValueError(f"Значения target должны быть в диапазоне [0, {num_classes-1}]. Получено: [{target.min()}, {target.max()}]")

    pred_softmax = F.softmax(pred, dim=1)

    start_cls = 1 if exclude_background else 0
    classes_to_analyze = list(range(start_cls, num_classes))

    results = {}

    for threshold in iou_thresholds:
        class_aps = []

        for cls in classes_to_analyze:
            pred_probs = pred_softmax[:, cls]
            target_binary = (target == cls).float()

            if target_binary.sum() == 0:
                continue

            ap = calculate_ap_for_class(pred_probs, target_binary, threshold)
            class_aps.append(ap)

        if class_aps:
            mean_ap = np.mean(class_aps)
        else:
            mean_ap = 0.0

        results[f'mAP@{threshold}'] = mean_ap

    return results

def calculate_ap_for_class(pred_probs, target_binary, iou_threshold):
    """
    Вычисляет Average Precision для одного класса при заданном IoU threshold
    Args:
        pred_probs: вероятности предсказания [B, H, W]
        target_binary: бинарная маска истинных значений [B, H, W]
        iou_threshold: порог IoU для считания предсказания правильным
    Returns:
        float: Average Precision
    """
    batch_size = pred_probs.shape[0]

    all_scores = []
    all_ious = []
    all_targets = []

    for b in range(batch_size):
        pred_prob = pred_probs[b]
        target_mask = target_binary[b]

        if target_mask.sum() == 0:
            continue

        prob_thresholds = torch.linspace(0.3, 0.7, 5).to(pred_prob.device)

        for prob_thresh in prob_thresholds:
            pred_binary = (pred_prob > prob_thresh).float()

            intersection = (pred_binary * target_mask).sum()
            union = pred_binary.sum() + target_mask.sum() - intersection

            if union > 0:
                iou = intersection / union
            else:
                iou = 1.0 if (pred_binary.sum() == 0 and target_mask.sum() == 0) else 0.0

            all_scores.append(pred_prob.max().item())
            all_ious.append(iou.item())
            all_targets.append(1.0)

    if not all_scores:
        return 0.0

    sorted_indices = np.argsort(all_scores)[::-1]
    sorted_ious = np.array(all_ious)[sorted_indices]

    tp = (sorted_ious >= iou_threshold).astype(float)
    fp = (sorted_ious < iou_threshold).astype(float)

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / len(all_targets) if len(all_targets) > 0 else np.array([0])
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

    ap = compute_ap(recalls, precisions)

    return ap

def compute_ap(recalls, precisions):
    """Вычисляет Average Precision по кривой precision-recall"""
    recalls = np.concatenate(([0], recalls, [1]))
    precisions = np.concatenate(([0], precisions, [0]))

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1

    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])

    return ap

def calculate_map50_map95(pred, target, num_classes, exclude_background=True):
    """
    Вычисляет mAP@0.5 и mAP@0.5:0.95 для сегментации (оптимизированная версия)
    Args:
        pred: предсказания модели [B, C, H, W]
        target: истинные маски [B, H, W]
        num_classes: количество классов
        exclude_background: исключить фон из расчета
    Returns:
        dict: {'mAP50': float, 'mAP50_95': float}
    """
    # mAP@0.5
    map50_result = calculate_map_segmentation(pred, target, num_classes, [0.5], exclude_background)
    map50 = map50_result.get('mAP@0.5', 0.0)

    # mAP@0.5:0.95 (уменьшенное количество порогов для ускорения)
    iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    map50_95_result = calculate_map_segmentation(pred, target, num_classes, iou_thresholds, exclude_background)

    all_maps = [map50_95_result.get(f'mAP@{thresh}', 0.0) for thresh in iou_thresholds]
    map50_95 = np.mean(all_maps)

    return {
        'mAP50': map50,
        'mAP50_95': map50_95
    }

### Загрузка чекпоинта ###

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cuda'):
    """Загружает чекпоинт модели с историей обучения"""
    print(f"\n{'='*60}")
    print(f"ЗАГРУЗКА ЧЕКПОИНТА")
    print(f"{'='*60}")
    print(f"Файл: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"*** Загружены веса модели")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"*** Загружено состояние оптимизатора")

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"*** Загружено состояние планировщика")

    start_epoch = checkpoint.get('epoch', 0) + 1
    training_history = checkpoint.get('training_history', None)
    best_val_dice = checkpoint.get('best_val_dice', 0.0)

    print(f"*** Эпоха чекпоинта: {checkpoint.get('epoch', 0)}")
    print(f"*** Best Dice: {best_val_dice:.4f}")

    if training_history is not None:
        print(f"*** Загружена история: {len(training_history['epoch'])} эпох")

    print(f"\nОбучение будет продолжено с эпохи {start_epoch}")
    print(f"{'='*60}\n")

    return {
        'start_epoch': start_epoch,
        'training_history': training_history,
        'best_val_dice': best_val_dice,
        'train_results': checkpoint.get('train_results', {}),
        'val_results': checkpoint.get('val_results', {})
    }

### Визуализация истории обучения ###

def plot_training_history(history, save_dir):
    """Визуализирует полную историю обучения"""
    print("\n" + "="*60)
    print("СОЗДАНИЕ ГРАФИКОВ ИСТОРИИ ОБУЧЕНИЯ")
    print("="*60)

    plots_dir = Path(save_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs = history['epoch']
    total_epochs = len(epochs)

    print(f"Всего эпох в истории: {total_epochs}")
    print(f"Создание графиков...")

    # 1. Loss и Learning Rate
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3)
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['learning_rate'], label='Learning Rate', marker='o', markersize=3, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / 'loss_and_lr.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  * loss_and_lr.png")

    # 2. Dice и IoU
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_dice'], label='Train Dice', marker='o', markersize=3)
    plt.plot(epochs, history['val_dice'], label='Val Dice', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Dice Coefficient')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_iou'], label='Train IoU', marker='o', markersize=3)
    plt.plot(epochs, history['val_iou'], label='Val IoU', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score')
    plt.title('Intersection over Union')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / 'dice_and_iou.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  * dice_and_iou.png")

    # 3. Accuracy
    plt.figure(figsize=(6, 5))
    plt.plot(epochs, history['train_accuracy'], label='Train Accuracy', marker='o', markersize=3)
    plt.plot(epochs, history['val_accuracy'], label='Val Accuracy', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Accuracy')
    plt.title('Pixel Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  * accuracy.png")

    print(f"\n* Все графики сохранены в: {plots_dir}")
    print(f"{'='*60}\n")


### Загрузка и вывод истории обучения ###

def load_training_history(history_path):
    """Загружает историю обучения из JSON файла или чекпоинта"""
    history_path = Path(history_path)

    if not history_path.exists():
        raise FileNotFoundError(f"Файл не найден: {history_path}")

    if history_path.suffix == '.json':
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
    elif history_path.suffix == '.pth':
        checkpoint = torch.load(history_path, map_location='cpu', weights_only=False)
        history = checkpoint.get('training_history', None)
        if history is None:
            raise ValueError("Чекпоинт не содержит истории обучения")
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {history_path.suffix}")

    return history

def print_training_summary(history):
    """Выводит сводку обучения"""
    print("="*60)
    print("СВОДКА ОБУЧЕНИЯ")
    print("="*60)
    print(f"Общее количество эпох: {len(history['epoch'])}")
    
    if len(history['val_dice']) > 0:
        best_dice_idx = np.argmax(history['val_dice'])
        best_iou_idx = np.argmax(history['val_iou'])
        best_loss_idx = np.argmin(history['val_loss'])
        
        print(f"\nЛучшие результаты:")
        print(f"Лучший Val Dice: {history['val_dice'][best_dice_idx]:.4f} (эпоха {history['epoch'][best_dice_idx]})")
        print(f"Лучший Val IoU: {history['val_iou'][best_iou_idx]:.4f} (эпоха {history['epoch'][best_iou_idx]})")
        print(f"Минимальный Val Loss: {history['val_loss'][best_loss_idx]:.4f} (эпоха {history['epoch'][best_loss_idx]})")
        
        print(f"\nФинальные результаты (эпоха {history['epoch'][-1]}):")
        print(f"Train Loss: {history['train_loss'][-1]:.4f}, Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"Train Dice: {history['train_dice'][-1]:.4f}, Val Dice: {history['val_dice'][-1]:.4f}")
        print(f"Train IoU: {history['train_iou'][-1]:.4f}, Val IoU: {history['val_iou'][-1]:.4f}")

def plot_metrics_comparison(history, output_dir, show_plots=False):
    """Создает сравнительные графики метрик"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = history['epoch']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train', marker='o', markersize=2)
    axes[0, 0].plot(epochs, history['val_loss'], label='Val', marker='s', markersize=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dice
    axes[0, 1].plot(epochs, history['train_dice'], label='Train', marker='o', markersize=2)
    axes[0, 1].plot(epochs, history['val_dice'], label='Val', marker='s', markersize=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice')
    axes[0, 1].set_title('Dice Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # IoU
    axes[0, 2].plot(epochs, history['train_iou'], label='Train', marker='o', markersize=2)
    axes[0, 2].plot(epochs, history['val_iou'], label='Val', marker='s', markersize=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('IoU')
    axes[0, 2].set_title('IoU Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 0].plot(epochs, history['train_accuracy'], label='Train', marker='o', markersize=2)
    axes[1, 0].plot(epochs, history['val_accuracy'], label='Val', marker='s', markersize=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Pixel Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 1].plot(epochs, history['train_f1_macro'], label='Train', marker='o', markersize=2)
    axes[1, 1].plot(epochs, history['val_f1_macro'], label='Val', marker='s', markersize=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 (Macro)')
    axes[1, 1].set_title('F1 Score (Macro)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 2].plot(epochs, history['learning_rate'], label='LR', marker='o', markersize=2, color='green')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_title('Learning Rate')
    axes[1, 2].set_yscale('log')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'key_metrics.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Ключевые метрики сохранены: {save_path}")
    
    if show_plots:
        plt.show()
    
    return fig

### Инференс ###

def run_inference(model, test_loader, device, num_classes, class_names, save_dir=None):
    """Запускает инференс на test подвыборке"""
    model.eval()
    test_metrics = SegmentationMetrics(num_classes, class_names, exclude_background=True)

    sample_images = []
    sample_masks = []
    sample_predictions = []

    print("Запуск инференса на test подвыборке...")

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc='Inference')):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            test_metrics.update(outputs, masks)

            if batch_idx < 2:
                pred_classes = torch.argmax(F.softmax(outputs, dim=1), dim=1)
                sample_images.extend(images.cpu())
                sample_masks.extend(masks.cpu())
                sample_predictions.extend(pred_classes.cpu())

    print("\nВычисление mAP метрик...")
    test_metrics.compute_map_metrics()

    results = test_metrics.compute()

    # Вывод результатов
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ИНФЕРЕНСА НА TEST ПОДВЫБОРКЕ")
    print("="*60)
    print(f"Pixel Accuracy: {results['pixel_accuracy']:.4f}")
    print(f"Mean Dice (без фона): {results['mean_dice']:.4f}")
    print(f"Mean IoU (без фона): {results['mean_iou']:.4f}")
    print(f"Mean Dice (с фоном): {results['mean_dice_with_bg']:.4f}")
    print(f"Mean IoU (с фоном): {results['mean_iou_with_bg']:.4f}")
    print(f"mAP@0.5: {results['map50']:.4f}")
    print(f"mAP@0.5:0.95: {results['map50_95']:.4f}")
    print(f"F1 (Macro): {results['f1_macro']:.4f}")
    print(f"F1 (Micro): {results['f1_micro']:.4f}")
    print(f"Precision (Macro): {results['precision_macro']:.4f}")
    print(f"Recall (Macro): {results['recall_macro']:.4f}")
    print(f"Precision (Micro): {results['precision_micro']:.4f}")
    print(f"Recall (Micro): {results['recall_micro']:.4f}")
    
    # Вывод метрик по классам
    print("\nПо классам (только классы с данными):")
    for class_name in class_names:
        dice = results['class_dice'].get(class_name, 0.0)
        iou = results['class_iou'].get(class_name, 0.0)
        f1 = results['class_f1'].get(class_name, 0.0)
        precision = results['class_precision'].get(class_name, 0.0)
        recall = results['class_recall'].get(class_name, 0.0)
        
        # Выводим только классы с данными
        if dice > 0 or iou > 0:
            print(f"{class_name}: Dice={dice:.4f}, IoU={iou:.4f}, F1={f1:.4f}, "
                  f"Precision={precision:.4f}, Recall={recall:.4f}")
    
    print("="*60)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Сохраняем метрики в JSON
        metrics_path = os.path.join(save_dir, 'test_metrics.json')
        metrics_to_save = {}
        for k, v in results.items():
            if isinstance(v, dict):
                metrics_to_save[k] = {str(key): float(val) if isinstance(val, (np.floating, np.integer)) else val 
                                     for key, val in v.items()}
            elif isinstance(v, (np.floating, np.integer)):
                metrics_to_save[k] = float(v)
            else:
                metrics_to_save[k] = v
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        print(f"Метрики сохранены в: {metrics_path}")
        
        # Визуализация примеров предсказаний
        if len(sample_images) > 0:
            viz_path = os.path.join(save_dir, 'test_predictions_visualization.png')
            visualize_predictions(sample_images[:8], sample_masks[:8], sample_predictions[:8], 
                                class_names, viz_path)
            print(f"Визуализация сохранена в: {viz_path}")
        
        # Сохраняем детальный отчет
        report_path = os.path.join(save_dir, 'test_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("РЕЗУЛЬТАТЫ ИНФЕРЕНСА НА TEST ПОДВЫБОРКЕ\n")
            f.write("="*60 + "\n\n")
            f.write(f"Pixel Accuracy: {results['pixel_accuracy']:.4f}\n")
            f.write(f"Mean Dice (без фона): {results['mean_dice']:.4f}\n")
            f.write(f"Mean IoU (без фона): {results['mean_iou']:.4f}\n")
            f.write(f"Mean Dice (с фоном): {results['mean_dice_with_bg']:.4f}\n")
            f.write(f"Mean IoU (с фоном): {results['mean_iou_with_bg']:.4f}\n")
            f.write(f"mAP@0.5: {results['map50']:.4f}\n")
            f.write(f"mAP@0.5:0.95: {results['map50_95']:.4f}\n")
            f.write(f"F1 (Macro): {results['f1_macro']:.4f}\n")
            f.write(f"F1 (Micro): {results['f1_micro']:.4f}\n")
            f.write(f"Precision (Macro): {results['precision_macro']:.4f}\n")
            f.write(f"Recall (Macro): {results['recall_macro']:.4f}\n")
            f.write(f"Precision (Micro): {results['precision_micro']:.4f}\n")
            f.write(f"Recall (Micro): {results['recall_micro']:.4f}\n\n")
            
            f.write("По классам (только классы с данными):\n")
            for class_name in class_names:
                dice = results['class_dice'].get(class_name, 0.0)
                iou = results['class_iou'].get(class_name, 0.0)
                f1 = results['class_f1'].get(class_name, 0.0)
                precision = results['class_precision'].get(class_name, 0.0)
                recall = results['class_recall'].get(class_name, 0.0)
                
                if dice > 0 or iou > 0:
                    f.write(f"{class_name}: Dice={dice:.4f}, IoU={iou:.4f}, F1={f1:.4f}, "
                           f"Precision={precision:.4f}, Recall={recall:.4f}\n")
            
            f.write("="*60 + "\n")
        
        print(f"Детальный отчет сохранен в: {report_path}\n")
        print(f"Инференс завершен! Результаты сохранены в: {save_dir}")

    return results

def visualize_predictions(images, masks, predictions, class_names, save_path):
    """Визуализирует примеры предсказаний"""
    n_samples = min(len(images), 8)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(n_samples):
        # Оригинальное изображение
        img = images[idx].squeeze().numpy()
        axes[idx, 0].imshow(img, cmap='gray')
        axes[idx, 0].set_title('Изображение')
        axes[idx, 0].axis('off')
        
        # Ground truth маска
        mask = masks[idx].numpy()
        axes[idx, 1].imshow(mask, cmap='tab20', vmin=0, vmax=len(class_names)-1)
        axes[idx, 1].set_title('Ground Truth')
        axes[idx, 1].axis('off')
        
        # Предсказанная маска
        pred = predictions[idx].numpy()
        axes[idx, 2].imshow(pred, cmap='tab20', vmin=0, vmax=len(class_names)-1)
        axes[idx, 2].set_title('Предсказание')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def inference_single_image(model_path, image_path, data_yaml_path, transform, output_dir='output', device='cuda'):
    """Инференс на одном изображении"""
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)

    # Загружаем конфигурацию датасета
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    num_classes = config['nc'] + 1
    class_names = ['Background'] + config['names']

    print(f"Количество классов: {num_classes}")
    print("Имена классов:", class_names)

    # Загружаем модель
    print("Загрузка модели...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    architecture = checkpoint.get('architecture', 'unetplusplus')
    encoder_name = checkpoint.get('encoder_name', 'efficientnet-b3')
    
    model = create_model(
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=1,
        classes=num_classes
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Предобработка изображения
    print("Предобработка изображения...")
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    original_h, original_w = image.shape
    original_image = image.copy()

    image_resized = cv2.resize(image, (512, 512))
    if len(image_resized.shape) == 2:
        image_resized = np.expand_dims(image_resized, axis=2)

    transformed = transform(image=image_resized)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    # Инференс
    print("Выполнение инференса...")
    with torch.no_grad():
        output = model(image_tensor)

    # Постобработка маски
    print("Постобработка маски...")
    mask_pred = torch.softmax(output, dim=1)
    mask_pred = torch.argmax(mask_pred, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    mask = cv2.resize(mask_pred, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    # Визуализация результатов
    print("Визуализация результатов...")
    image_name = Path(image_path).stem
    save_path = output_dir / f'{image_name}_result.png'
    
    result_image = visualize_inference_results(original_image, mask, class_names, save_path)

    # Сохраняем маску отдельно
    mask_save_path = output_dir / f'{image_name}_mask.png'
    plt.imsave(mask_save_path, mask, cmap='tab20')

    # Сохраняем результат с наложением
    result_save_path = output_dir / f'{image_name}_overlay.png'
    cv2.imwrite(str(result_save_path), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

    print(f"Маска сохранена в: {mask_save_path}")
    print(f"Результат с наложением сохранен в: {result_save_path}")

    # Выводим статистику по классам
    unique_classes = np.unique(mask)
    print("\nОбнаруженные классы:")
    for class_id in unique_classes:
        if class_id < len(class_names):
            class_name = class_names[class_id]
            pixel_count = np.sum(mask == class_id)
            print(f"  {class_name} (ID: {class_id}): {pixel_count} пикселей")

    return mask, result_image

def visualize_inference_results(original_image, mask, class_names, save_path=None):
    """Визуализация результатов сегментации"""
    # Создаем цветную маску
    colored_mask = create_colored_mask(mask, class_names)

    # Накладываем маску на оригинальное изображение
    if len(original_image.shape) == 2:
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        original_rgb = original_image

    # Смешиваем изображение с маской
    alpha = 0.6
    blended = cv2.addWeighted(original_rgb, 1 - alpha, colored_mask, alpha, 0)

    # Добавляем подписи классов
    result_with_labels = add_class_labels(blended, mask, class_names)

    # Создаем фигуру для отображения
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Оригинальное изображение
    if len(original_image.shape) == 2:
        axes[0, 0].imshow(original_image, cmap='gray')
    else:
        axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Оригинальное изображение')
    axes[0, 0].axis('off')

    # Предсказанная маска
    axes[0, 1].imshow(mask, cmap='tab20')
    axes[0, 1].set_title('Предсказанная маска')
    axes[0, 1].axis('off')

    # Цветная маска
    axes[1, 0].imshow(colored_mask)
    axes[1, 0].set_title('Цветная маска')
    axes[1, 0].axis('off')

    # Результат с наложением и подписями
    axes[1, 1].imshow(cv2.cvtColor(result_with_labels, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Результат сегментации с подписями')
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Результат сохранен в: {save_path}")

    plt.show()

    return result_with_labels

def create_colored_mask(mask, class_names, alpha=0.7):
    """Создание цветной маски с разными цветами для каждого класса"""
    num_classes = len(class_names)

    # Используем фиксированную палитру
    colors = [
        [0, 0, 0],        # 0: фон - черный
        [255, 0, 0],      # 1: красный
        [0, 255, 0],      # 2: зеленый
        [0, 0, 255],      # 3: синий
        [255, 255, 0],    # 4: желтый
        [255, 0, 255],    # 5: пурпурный
        [0, 255, 255],    # 6: голубой
        [255, 165, 0],    # 7: оранжевый
        [128, 0, 128],    # 8: фиолетовый
        [165, 42, 42],    # 9: коричневый
        [128, 128, 128],  # 10: серый
        [255, 192, 203],  # 11: розовый
        [0, 128, 0],      # 12: темно-зеленый
        [0, 0, 128],      # 13: темно-синий
        [128, 0, 0],      # 14: темно-красный
        [128, 128, 0],    # 15: оливковый
        [0, 128, 128],    # 16: бирюзовый
        [128, 0, 128],    # 17: фиолетовый
        [192, 192, 192],  # 18: серебряный
        [64, 64, 64],     # 19: темно-серый
        [255, 165, 0],    # 20: оранжевый
        [210, 180, 140],  # 21: танг
        [32, 178, 170],   # 22: светло-морской
        [135, 206, 235],  # 23: небесно-голубой
        [221, 160, 221],  # 24: сливовый
        [240, 230, 140],  # 25: хаки
        [245, 222, 179],  # 26: пшеничный
        [255, 228, 196],  # 27: бисквитный
        [240, 255, 240],  # 28: медовый
        [245, 245, 220],  # 29: бежевый
        [255, 250, 250],  # 30: снежный
        [47, 79, 79],     # 31: темный грифельно-серый
        [105, 105, 105]   # 32: димгрей
    ]

    # Дублируем цвета если классов больше
    while len(colors) < num_classes:
        colors.extend(colors)

    # Создаем цветное изображение
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for class_id in range(num_classes):
        class_mask = mask == class_id
        color = colors[class_id]
        colored_mask[class_mask] = color

    return colored_mask

def add_class_labels(image, mask, class_names, min_area=100):
    """Добавление подписей классов на изображение"""
    result_image = image.copy()

    for class_id, class_name in enumerate(class_names):
        if class_id == 0:  # Пропускаем фон
            continue

        # Создаем маску для текущего класса
        class_mask = (mask == class_id).astype(np.uint8)

        # Находим контуры
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Фильтруем по площади
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Находим ограничивающий прямоугольник
            x, y, w, h = cv2.boundingRect(contour)

            # Добавляем текст в центре прямоугольника
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            # Получаем размеры текста
            text_size = cv2.getTextSize(class_name, font, font_scale, thickness)[0]

            # Вычисляем позицию текста
            text_x = x + w // 2 - text_size[0] // 2
            text_y = y + h // 2 + text_size[1] // 2

            # Рисуем черный фон для текста
            cv2.rectangle(result_image,
                        (text_x - 2, text_y - text_size[1] - 2),
                        (text_x + text_size[0] + 2, text_y + 2),
                        (0, 0, 0), -1)

            # Рисуем белый текст
            cv2.putText(result_image, class_name,
                      (text_x, text_y),
                      font, font_scale, (255, 255, 255), thickness)

    return result_image

def inference_multiple_images(model_path, image_dir, data_yaml_path, transform, output_dir='output', device='cuda'):
    """Инференс на нескольких изображениях в директории"""
    image_dir = Path(image_dir)
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    # Проверяем существование директории
    if not image_dir.exists():
        raise FileNotFoundError(f"Директория {image_dir} не существует")

    # Получаем список всех файлов изображений
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.bmp', '*.BMP']
    image_files = []

    for extension in image_extensions:
        image_files.extend(image_dir.glob(f"**/{extension}"))

    # Если в корневой директории нет файлов, ищем без поддиректорий
    if len(image_files) == 0:
        for extension in image_extensions:
            image_files.extend(image_dir.glob(extension))

    # Удаляем дубликаты и сортируем
    image_files = sorted(list(set(image_files)))

    print(f"Найдено {len(image_files)} изображений в директории {image_dir}")

    # Выводим список найденных файлов
    if len(image_files) == 0:
        print("Содержимое директории:")
        for item in image_dir.iterdir():
            print(f"  {item.name} (директория: {item.is_dir()})")
    else:
        print("Первые 10 найденных изображений:")
        for img_path in image_files[:10]:
            print(f"  {img_path}")
        if len(image_files) > 10:
            print(f"  ... и еще {len(image_files) - 10} изображений")

    masks = []
    results = []

    for image_path in image_files:
        print(f"\n{'='*50}")
        print(f"Обработка изображения: {image_path.name}")
        print(f"{'='*50}")

        try:
            # Создаем поддиректорию для каждого изображения
            img_output_dir = output_dir / image_path.stem
            img_output_dir.mkdir(parents=True, exist_ok=True)

            mask, result = inference_single_image(
                model_path, image_path, data_yaml_path, transform, img_output_dir, device
            )
            masks.append(mask)
            results.append(result)
            print(f"✓ Успешно обработано: {image_path.name}")

        except Exception as e:
            print(f"✗ Ошибка при обработке {image_path}: {e}")
            traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"Обработка завершена. Успешно обработано: {len(masks)} из {len(image_files)} изображений")
    print(f"{'='*50}")

    return masks, results
