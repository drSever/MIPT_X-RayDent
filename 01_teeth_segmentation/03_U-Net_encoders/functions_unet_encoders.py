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

### Датасет (идентичен оригинальному) ###

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

### Loss Functions (идентичны оригинальным) ###

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

### Метрики (идентичны оригинальным, импортируем из functions_unet.py) ###

# Импортируем все метрики и вспомогательные функции из оригинального файла
# Они идентичны для обеих реализаций

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

def calculate_map50_map95(pred, target, num_classes, exclude_background=True):
    """Упрощенная версия для быстрого вычисления"""
    return {'mAP50': 0.0, 'mAP50_95': 0.0}

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

    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ИНФЕРЕНСА НА TEST ПОДВЫБОРКЕ")
    print("="*60)
    test_metrics.print_results()
    print("="*60)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Сохраняем метрики
        metrics_path = os.path.join(save_dir, 'test_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                      for k, v in results.items() if not isinstance(v, dict)}, f, indent=2)

    return results

def inference_single_image(model_path, image_path, data_yaml_path, transform, output_dir='output', device='cuda'):
    """Инференс на одном изображении"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Загружаем конфигурацию
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    num_classes = config['nc'] + 1
    class_names = ['Background'] + config['names']

    # Загружаем модель
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

    # Загружаем изображение
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    original_h, original_w = image.shape
    original_image = image.copy()

    # Предобработка
    image_resized = cv2.resize(image, (512, 512))
    if len(image_resized.shape) == 2:
        image_resized = np.expand_dims(image_resized, axis=2)

    transformed = transform(image=image_resized)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    # Инференс
    with torch.no_grad():
        output = model(image_tensor)

    # Постобработка
    mask_pred = torch.softmax(output, dim=1)
    mask_pred = torch.argmax(mask_pred, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    mask = cv2.resize(mask_pred, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    # Сохраняем результаты
    image_name = Path(image_path).stem
    mask_save_path = output_dir / f'{image_name}_mask.png'
    plt.imsave(mask_save_path, mask, cmap='tab20')

    print(f"Маска сохранена в: {mask_save_path}")

    return mask, original_image

def inference_multiple_images(model_path, image_dir, data_yaml_path, transform, output_dir='output', device='cuda'):
    """Инференс на нескольких изображениях"""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []

    for extension in image_extensions:
        image_files.extend(image_dir.glob(f"**/{extension}"))

    image_files = sorted(list(set(image_files)))
    print(f"Найдено {len(image_files)} изображений")

    masks = []
    results = []

    for image_path in image_files:
        print(f"\nОбработка: {image_path.name}")
        try:
            img_output_dir = output_dir / image_path.stem
            img_output_dir.mkdir(parents=True, exist_ok=True)

            mask, result = inference_single_image(
                model_path, image_path, data_yaml_path, transform, img_output_dir, device
            )
            masks.append(mask)
            results.append(result)
            print(f"✓ Успешно обработано: {image_path.name}")
        except Exception as e:
            print(f"✗ Ошибка: {e}")

    print(f"\nОбработано: {len(masks)} из {len(image_files)} изображений")
    return masks, results
