##########################################################################################
# Функции для решения задачи сегментации зубов на ортопантомогаммах с использованием U-Net
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

import matplotlib.pyplot as plt
import seaborn as sns

### Фиксация SEED ###

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # для multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Для современных версий PyTorch
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    
### Датасет ###

class TeethSegmentationDataset(Dataset):
    def __init__(self, data_dir, img_size, split='train', transform=None):
        """
        Датасет для instance-сегментации зубов, разметка в формате YOLO
        Args:
            data_dir: путь к папке с датасетом
            img_size: размер изображения для ресайза
            split: 'train', 'valid' или 'test'
            transform: трансформации для изображений
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.img_size = img_size

        # Загружаем конфигурацию датасета
        with open(self.data_dir / 'data.yaml', 'r') as f:
            self.config = yaml.safe_load(f)

        self.num_classes = self.config['nc'] + 1  # 32 класса зубов + 1 фон = 33
        self.class_names = ['Background'] + self.config['names']  # Добавляем фон как класс 0

        # Пути к изображениям и меткам
        self.images_dir = self.data_dir / split / 'images'
        self.labels_dir = self.data_dir / split / 'labels'

        # Получаем список файлов изображений
        self.image_files = list(self.images_dir.glob('*.jpg')) + \
                          list(self.images_dir.glob('*.png')) + \
                          list(self.images_dir.glob('*.jpeg'))

        print(f"Загружено {len(self.image_files)} изображений для {split}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Загружаем изображение
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {img_path}")

        # оригинальные высота и ширина изображения
        original_h, original_w = image.shape

        # Ресайзим изображение
        image = cv2.resize(image, (self.img_size, self.img_size))

        # Создаем маску сегментации
        mask = self.load_yolo_mask(img_path, original_w, original_h)

        # Добавляем канал для совместимости с Albumentations (HWC формат)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)  # [H, W] -> [H, W, 1]

        # Применяем трансформации если есть
        if self.transform:

            # Применяем трансформации
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

            # Если трансформации не включают ToTensorV2, конвертируем вручную
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).float()
                if len(image.shape) == 3 and image.shape[2] == 1:
                    image = image.permute(2, 0, 1)  # [H, W, 1] -> [1, H, W]
                elif len(image.shape) == 2:
                    image = image.unsqueeze(0)  # [H, W] -> [1, H, W]

            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).long()
            else:
                # Если маска уже тензор, убеждаемся что она Long
                mask = mask.long()
        else:
            # Без трансформаций - простая конвертация
            # Убираем лишний канал и конвертируем в тензор
            if len(image.shape) == 3 and image.shape[2] == 1:
                image = image.squeeze(2)  # [H, W, 1] -> [H, W]
            image = torch.from_numpy(image).float().unsqueeze(0)  # [1, H, W]
            mask = torch.from_numpy(mask).long()  # [H, W]

        # Финальная проверка размерности и типов
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 2:
                image = image.unsqueeze(0)  # [1, H, W]
            assert len(image.shape) == 3 and image.shape[0] == 1, f"Неправильная размерность изображения: {image.shape}"

        # Убеждаемся что маска имеет правильный тип
        if isinstance(mask, torch.Tensor):
            mask = mask.long()

        return image, mask

    def load_yolo_mask(self, img_path, original_w, original_h):
        """
        Загружает маску сегментации из YOLO формата с учетом пропорций изображения
        """
        # Путь к файлу с метками
        label_path = self.labels_dir / (img_path.stem + '.txt')

        # Создаем пустую маску (фон = 0)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        if not label_path.exists():
            return mask

        # Читаем аннотации YOLO
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # Вычисляем коэффициенты масштабирования для сохранения пропорций
        scale_x = self.img_size / original_w
        scale_y = self.img_size / original_h

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 6:  # class_id + минимум 4 координаты полигона
                continue

            class_id = int(parts[0]) + 1  # YOLO классы 0-31 становятся классами 1-32, фон остается 0

            # Координаты полигона (нормализованные относительно исходного изображения)
            coords = list(map(float, parts[1:]))

            # Преобразуем в абсолютные координаты с учетом масштабирования
            polygon_points = []
            for i in range(0, len(coords), 2):
                # Сначала денормализуем относительно исходного размера
                x_orig = coords[i] * original_w
                y_orig = coords[i+1] * original_h

                # Затем масштабируем к целевому размеру
                x = int(x_orig * scale_x)
                y = int(y_orig * scale_y)

                # Ограничиваем координаты размерами маски
                x = max(0, min(x, self.img_size - 1))
                y = max(0, min(y, self.img_size - 1))

                polygon_points.append([x, y])

            # Заполняем полигон на маске
            if len(polygon_points) >= 3:
                polygon_points = np.array(polygon_points, dtype=np.int32)
                cv2.fillPoly(mask, [polygon_points], class_id)

        return mask

    def get_class_weights(self):
        """
        Вычисляет веса классов для борьбы с дисбалансом
        Загружает только маски без трансформаций
        """
        class_counts = np.zeros(self.num_classes)  # num_classes уже включает фон (33 класса)

        print(f"Вычисляем веса для {self.num_classes} классов (фон + {self.num_classes-1} зубов)...")

        # Берем подвыборку для ускорения (каждое 10-е изображение)
        sample_indices = range(0, len(self), max(1, len(self) // 100))  # Максимум 100 изображений

        for idx in sample_indices:
            # Загружаем только маску без полной обработки изображения
            img_path = self.image_files[idx]

            # Получаем размеры исходного изображения для корректного масштабирования маски
            temp_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if temp_img is None:
                continue
            original_h, original_w = temp_img.shape

            # Загружаем маску напрямую
            mask = self.load_yolo_mask(img_path, original_w, original_h)

            unique, counts = np.unique(mask, return_counts=True)
            for cls, count in zip(unique, counts):
                if cls < self.num_classes:  # Проверяем границы
                    class_counts[cls] += count

        # Показываем статистику классов
        print("Распределение классов:")
        for i in range(self.num_classes):
            if class_counts[i] > 0:
                print(f"  Класс {i}: {class_counts[i]:,} пикселей")

        # Вычисляем веса (обратно пропорциональные частоте)
        total_pixels = class_counts.sum()
        weights = total_pixels / (class_counts + 1e-8)  # избегаем деления на 0
        weights = weights / weights.sum() * len(weights)  # нормализуем

        print(f"Размер весов: {len(weights)}")

        return torch.FloatTensor(weights)
    
### Loss ###

class DiceLoss(nn.Module):
    """
    Dice Loss для сегментации
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Проверка размерностей
        if pred.dim() != 4 or target.dim() != 3:
            raise ValueError(f"Ожидаемые размерности: pred [B,C,H,W], target [B,H,W]. Получено: pred {pred.shape}, target {target.shape}")

        if pred.shape[0] != target.shape[0] or pred.shape[2:] != target.shape[1:]:
            raise ValueError(f"Несовместимые размеры: pred {pred.shape}, target {target.shape}")

        # Убеждаемся что target имеет правильный тип
        target = target.long()

        # Проверяем валидность значений в target
        if target.min() < 0 or target.max() >= pred.size(1):
            raise ValueError(f"Значения target должны быть в диапазоне [0, {pred.size(1)-1}]. Получено: [{target.min()}, {target.max()}]")

        pred = F.softmax(pred, dim=1)

        # Конвертируем target в one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()

        # Вычисляем Dice для каждого класса
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Исключаем классы, которые отсутствуют в батче
        valid_classes = target_one_hot.sum(dim=(0, 2, 3)) > 0
        if valid_classes.sum() > 0:
            dice = dice[:, valid_classes]

        # Возвращаем среднюю потерю по всем классам и батчу
        return 1.0 - dice.mean()

class FocalLoss(nn.Module):
    """
    Focal Loss для борьбы с дисбалансом классов
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        # Проверка размерностей
        if pred.dim() != 4 or target.dim() != 3:
            raise ValueError(f"Ожидаемые размерности: pred [B,C,H,W], target [B,H,W]. Получено: pred {pred.shape}, target {target.shape}")

        if pred.shape[0] != target.shape[0] or pred.shape[2:] != target.shape[1:]:
            raise ValueError(f"Несовместимые размеры: pred {pred.shape}, target {target.shape}")

        # Убеждаемся что target имеет правильный тип
        target = target.long()

        # Проверяем валидность значений в target
        if target.min() < 0 or target.max() >= pred.size(1):
            raise ValueError(f"Значения target должны быть в диапазоне [0, {pred.size(1)-1}]. Получено: [{target.min()}, {target.max()}]")

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
    """
    Комбинированная функция потерь: CrossEntropy + Dice + Focal
    """
    def __init__(self, ce_weight=1.0, dice_weight=1.0, focal_weight=0.5,
                 class_weights=None, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss(smooth=smooth)
        self.focal_loss = FocalLoss()

    def forward(self, pred, target):
        # Проверка размерностей выполняется в каждой отдельной функции потерь
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)

        total_loss = (self.ce_weight * ce +
                     self.dice_weight * dice +
                     self.focal_weight * focal)

        # Создаем копии тензоров для логирования, чтобы не нарушить граф вычислений
        return total_loss, {
            'ce_loss': ce.detach().item(),
            'dice_loss': dice.detach().item(),
            'focal_loss': focal.detach().item(),
            'total_loss': total_loss.detach().item()
        }

class IoULoss(nn.Module):
    """
    IoU Loss для сегментации
    """
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Проверка размерностей
        if pred.dim() != 4 or target.dim() != 3:
            raise ValueError(f"Ожидаемые размерности: pred [B,C,H,W], target [B,H,W]. Получено: pred {pred.shape}, target {target.shape}")

        if pred.shape[0] != target.shape[0] or pred.shape[2:] != target.shape[1:]:
            raise ValueError(f"Несовместимые размеры: pred {pred.shape}, target {target.shape}")

        # Убеждаемся что target имеет правильный тип
        target = target.long()

        # Проверяем валидность значений в target
        if target.min() < 0 or target.max() >= pred.size(1):
            raise ValueError(f"Значения target должны быть в диапазоне [0, {pred.size(1)-1}]. Получено: [{target.min()}, {target.max()}]")

        pred = F.softmax(pred, dim=1)

        # Конвертируем target в one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()

        # Вычисляем IoU для каждого класса
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3)) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        # Исключаем классы, которые отсутствуют в батче
        valid_classes = target_one_hot.sum(dim=(0, 2, 3)) > 0
        if valid_classes.sum() > 0:
            iou = iou[:, valid_classes]

        # Возвращаем среднюю потерю по всем классам и батчу
        return 1.0 - iou.mean()

class TverskyLoss(nn.Module):
    """
    Tversky Loss - обобщение Dice Loss с контролем false positives и false negatives
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # вес false positives
        self.beta = beta    # вес false negatives
        self.smooth = smooth

    def forward(self, pred, target):
        # Проверка размерностей
        if pred.dim() != 4 or target.dim() != 3:
            raise ValueError(f"Ожидаемые размерности: pred [B,C,H,W], target [B,H,W]. Получено: pred {pred.shape}, target {target.shape}")

        if pred.shape[0] != target.shape[0] or pred.shape[2:] != target.shape[1:]:
            raise ValueError(f"Несовместимые размеры: pred {pred.shape}, target {target.shape}")

        # Убеждаемся что target имеет правильный тип
        target = target.long()

        # Проверяем валидность значений в target
        if target.min() < 0 or target.max() >= pred.size(1):
            raise ValueError(f"Значения target должны быть в диапазоне [0, {pred.size(1)-1}]. Получено: [{target.min()}, {target.max()}]")

        pred = F.softmax(pred, dim=1)

        # Конвертируем target в one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()

        # True Positives, False Positives, False Negatives
        tp = (pred * target_one_hot).sum(dim=(2, 3))
        fp = (pred * (1 - target_one_hot)).sum(dim=(2, 3))
        # Более стабильное вычисление false negatives
        fn = (target_one_hot * (1 - pred)).sum(dim=(2, 3))

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Исключаем классы, которые отсутствуют в батче
        valid_classes = target_one_hot.sum(dim=(0, 2, 3)) > 0
        if valid_classes.sum() > 0:
            tversky = tversky[:, valid_classes]

        return 1.0 - tversky.mean()

class WeightedCrossEntropyLoss(nn.Module):
    """
    Взвешенная Cross Entropy Loss с автоматическим вычислением весов
    """
    def __init__(self, num_classes):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.register_buffer('class_weights', torch.ones(num_classes))

    def update_weights(self, dataset):
        """
        Обновляет веса классов на основе датасета
        """
        if hasattr(dataset, 'get_class_weights'):
            weights = dataset.get_class_weights()

            # Проверяем размерность весов
            if len(weights) != self.num_classes:
                raise ValueError(f"Размер весов ({len(weights)}) не соответствует количеству классов ({self.num_classes})")

            # Убеждаемся что веса на том же устройстве
            if weights.device != self.class_weights.device:
                weights = weights.to(self.class_weights.device)

            self.class_weights = weights

    def forward(self, pred, target):
        # Проверка размерностей
        if pred.dim() != 4 or target.dim() != 3:
            raise ValueError(f"Ожидаемые размерности: pred [B,C,H,W], target [B,H,W]. Получено: pred {pred.shape}, target {target.shape}")

        if pred.shape[0] != target.shape[0] or pred.shape[2:] != target.shape[1:]:
            raise ValueError(f"Несовместимые размеры: pred {pred.shape}, target {target.shape}")

        # Убеждаемся что target имеет правильный тип
        target = target.long()

        # Проверяем валидность значений в target
        if target.min() < 0 or target.max() >= pred.size(1):
            raise ValueError(f"Значения target должны быть в диапазоне [0, {pred.size(1)-1}]. Получено: [{target.min()}, {target.max()}]")

        return F.cross_entropy(pred, target, weight=self.class_weights)

def get_loss_function(loss_type='combined', num_classes=33, class_weights=None):
    """
    Фабрика функций потерь
    Args:
        loss_type: тип функции потерь ('ce', 'dice', 'focal', 'combined', 'iou', 'tversky')
        num_classes: количество классов
        class_weights: веса классов для борьбы с дисбалансом
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'focal':
        return FocalLoss()
    elif loss_type == 'combined':
        return CombinedLoss(class_weights=class_weights)
    elif loss_type == 'iou':
        return IoULoss()
    elif loss_type == 'tversky':
        return TverskyLoss()
    elif loss_type == 'weighted_ce':
        return WeightedCrossEntropyLoss(num_classes)
    else:
        raise ValueError(f"Неизвестный тип функции потерь: {loss_type}")
    
### Метрики ###

def dice_coefficient(pred, target, num_classes, smooth=1e-6, exclude_background=False):
    """
    Вычисляет Dice коэффициент для каждого класса
    Args:
        pred: предсказания модели [B, C, H, W]
        target: истинные маски [B, H, W]
        num_classes: количество классов
        smooth: сглаживающий параметр
        exclude_background: исключить фон (класс 0) из расчета
    """
    # Проверка размерностей
    if pred.dim() != 4 or target.dim() != 3:
        raise ValueError(f"Ожидаемые размерности: pred [B,C,H,W], target [B,H,W]. Получено: pred {pred.shape}, target {target.shape}")

    if pred.shape[0] != target.shape[0] or pred.shape[2:] != target.shape[1:]:
        raise ValueError(f"Несовместимые размеры: pred {pred.shape}, target {target.shape}")

    # Убеждаемся что target имеет правильный тип и на том же устройстве
    target = target.long().to(pred.device)

    # Проверяем валидность значений в target
    if target.min() < 0 or target.max() >= num_classes:
        raise ValueError(f"Значения target должны быть в диапазоне [0, {num_classes-1}]. Получено: [{target.min()}, {target.max()}]")

    pred_softmax = F.softmax(pred, dim=1)

    # Получаем бинарные предсказания для вычисления Dice
    pred_binary = torch.argmax(pred_softmax, dim=1)  # [B, H, W]
    pred_binary_one_hot = F.one_hot(pred_binary, num_classes=num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]

    # Вычисляем intersection и union для всех классов одновременно (используем бинарные маски)
    intersection = (pred_binary_one_hot * target_one_hot).sum(dim=(0, 2, 3))  # [C]
    pred_sum = pred_binary_one_hot.sum(dim=(0, 2, 3))  # [C]
    target_sum = target_one_hot.sum(dim=(0, 2, 3))  # [C]

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)

    dice_scores = dice.cpu().numpy().tolist()

    # Если исключаем фон, возвращаем только классы без фона
    if exclude_background and num_classes > 1:
        return dice_scores[1:]

    return dice_scores

def iou_score(pred, target, num_classes, smooth=1e-6, exclude_background=False):
    """
    Вычисляет IoU (Intersection over Union) для каждого класса
    Args:
        pred: предсказания модели [B, C, H, W]
        target: истинные маски [B, H, W]
        num_classes: количество классов
        smooth: сглаживающий параметр
        exclude_background: исключить фон (класс 0) из расчета
    """
    # Проверка размерностей
    if pred.dim() != 4 or target.dim() != 3:
        raise ValueError(f"Ожидаемые размерности: pred [B,C,H,W], target [B,H,W]. Получено: pred {pred.shape}, target {target.shape}")

    if pred.shape[0] != target.shape[0] or pred.shape[2:] != target.shape[1:]:
        raise ValueError(f"Несовместимые размеры: pred {pred.shape}, target {target.shape}")

    # Убеждаемся что target имеет правильный тип и на том же устройстве
    target = target.long().to(pred.device)

    # Проверяем валидность значений в target
    if target.min() < 0 or target.max() >= num_classes:
        raise ValueError(f"Значения target должны быть в диапазоне [0, {num_classes-1}]. Получено: [{target.min()}, {target.max()}]")

    pred_softmax = F.softmax(pred, dim=1)

    # Получаем бинарные предсказания для корректного вычисления IoU
    pred_binary = torch.argmax(pred_softmax, dim=1)  # [B, H, W]
    pred_binary_one_hot = F.one_hot(pred_binary, num_classes=num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]

    # Вычисляем intersection и union для всех классов одновременно (используем бинарные маски)
    intersection = (pred_binary_one_hot * target_one_hot).sum(dim=(0, 2, 3))  # [C]
    pred_sum = pred_binary_one_hot.sum(dim=(0, 2, 3))  # [C]
    target_sum = target_one_hot.sum(dim=(0, 2, 3))  # [C]
    union = pred_sum + target_sum - intersection

    iou = (intersection + smooth) / (union + smooth)

    iou_scores = iou.cpu().numpy().tolist()

    # Если исключаем фон, возвращаем только классы без фона
    if exclude_background and num_classes > 1:
        return iou_scores[1:]

    return iou_scores

def pixel_accuracy(pred, target):
    """
    Вычисляет точность на уровне пикселей
    """
    # Проверка размерностей
    if pred.dim() != 4 or target.dim() != 3:
        raise ValueError(f"Ожидаемые размерности: pred [B,C,H,W], target [B,H,W]. Получено: pred {pred.shape}, target {target.shape}")

    if pred.shape[0] != target.shape[0] or pred.shape[2:] != target.shape[1:]:
        raise ValueError(f"Несовместимые размеры: pred {pred.shape}, target {target.shape}")

    # Убеждаемся что target имеет правильный тип и на том же устройстве
    target = target.long().to(pred.device)

    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)

    correct = (pred == target).float()
    accuracy = correct.sum() / correct.numel()

    return accuracy.item()

def mean_iou(pred, target, num_classes):
    """
    Вычисляет средний IoU по всем классам
    """
    iou_scores = iou_score(pred, target, num_classes)
    return np.mean(iou_scores)

def class_wise_accuracy(pred, target, num_classes):
    """
    Вычисляет точность для каждого класса отдельно
    """
    # Проверка размерностей
    if pred.dim() != 4 or target.dim() != 3:
        raise ValueError(f"Ожидаемые размерности: pred [B,C,H,W], target [B,H,W]. Получено: pred {pred.shape}, target {target.shape}")

    if pred.shape[0] != target.shape[0] or pred.shape[2:] != target.shape[1:]:
        raise ValueError(f"Несовместимые размеры: pred {pred.shape}, target {target.shape}")

    # Убеждаемся что target имеет правильный тип и на том же устройстве
    target = target.long().to(pred.device)

    # Проверяем валидность значений в target
    if target.min() < 0 or target.max() >= num_classes:
        raise ValueError(f"Значения target должны быть в диапазоне [0, {num_classes-1}]. Получено: [{target.min()}, {target.max()}]")

    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)

    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()

    accuracies = []

    for cls in range(num_classes):
        cls_mask = (target_np == cls)
        if cls_mask.sum() == 0:
            accuracies.append(float('nan'))  # NaN для отсутствующих классов
        else:
            cls_accuracy = (pred_np[cls_mask] == target_np[cls_mask]).mean()
            accuracies.append(float(cls_accuracy))

    return accuracies

def confusion_matrix_multiclass(pred, target, num_classes):
    """
    Вычисляет матрицу ошибок для многоклассовой сегментации
    """
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)

    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()

    cm = confusion_matrix(target_np, pred_np, labels=range(num_classes))
    return cm

class SegmentationMetrics:
    """
    Класс для накопления и вычисления метрик сегментации
    """
    def __init__(self, num_classes, class_names=None, exclude_background=True):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.exclude_background = exclude_background  # Исключать фон из средних метрик
        self.reset()

    def reset(self):
        # Накапливаем компоненты для Dice и IoU
        self.intersection_sum = np.zeros(self.num_classes)
        self.pred_sum_sum = np.zeros(self.num_classes)
        self.target_sum_sum = np.zeros(self.num_classes)

        # Для pixel accuracy
        self.total_accuracy = 0.0
        self.total_samples = 0

        # Накапливаем TP, FP, FN для каждого класса для точного вычисления precision/recall
        self.total_tp = np.zeros(self.num_classes)
        self.total_fp = np.zeros(self.num_classes)
        self.total_fn = np.zeros(self.num_classes)

        # Для mAP метрик - накапливаем предсказания и цели для вычисления в конце эпохи
        self.accumulated_predictions = []
        self.accumulated_targets = []
        self.map50 = 0.0
        self.map50_95 = 0.0

    def update(self, pred, target):
        """
        Обновляет метрики новым батчем
        """
        # Проверка размерностей
        if pred.dim() != 4 or target.dim() != 3:
            raise ValueError(f"Ожидаемые размерности: pred [B,C,H,W], target [B,H,W]. Получено: pred {pred.shape}, target {target.shape}")

        if pred.shape[0] != target.shape[0] or pred.shape[2:] != target.shape[1:]:
            raise ValueError(f"Несовместимые размеры: pred {pred.shape}, target {target.shape}")

        batch_size = pred.size(0)
        target = target.long().to(pred.device)

        # Проверяем валидность значений в target
        if target.min() < 0 or target.max() >= self.num_classes:
            raise ValueError(f"Значения target должны быть в диапазоне [0, {self.num_classes-1}]. Получено: [{target.min()}, {target.max()}]")

        # Накапливаем компоненты для Dice и IoU (НЕ готовые метрики!)
        pred_softmax = F.softmax(pred, dim=1)

        # Получаем бинарные предсказания (НЕ вероятности!)
        pred_binary = torch.argmax(pred_softmax, dim=1)  # [B, H, W]
        pred_binary_one_hot = F.one_hot(pred_binary, num_classes=self.num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Вычисляем компоненты для всех классов одновременно (используем бинарные маски)
        intersection = (pred_binary_one_hot * target_one_hot).sum(dim=(0, 2, 3))  # [C]
        pred_sum = pred_binary_one_hot.sum(dim=(0, 2, 3))  # [C] - количество пикселей каждого класса в предсказании
        target_sum = target_one_hot.sum(dim=(0, 2, 3))  # [C] - количество пикселей каждого класса в истине

        # Накапливаем компоненты (отсоединяем от графа градиентов)
        self.intersection_sum += intersection.detach().cpu().numpy()
        self.pred_sum_sum += pred_sum.detach().cpu().numpy()
        self.target_sum_sum += target_sum.detach().cpu().numpy()

        # Pixel accuracy
        acc = pixel_accuracy(pred, target)
        self.total_accuracy += acc * batch_size

        # Вычисление TP, FP, FN для каждого класса
        pred_classes = torch.argmax(pred_softmax, dim=1)

        # Векторизованное вычисление TP, FP, FN
        for cls in range(self.num_classes):
            pred_cls = (pred_classes == cls)
            target_cls = (target == cls)

            tp = (pred_cls & target_cls).sum().item()
            fp = (pred_cls & ~target_cls).sum().item()
            fn = (~pred_cls & target_cls).sum().item()

            self.total_tp[cls] += tp
            self.total_fp[cls] += fp
            self.total_fn[cls] += fn

        # Накапливаем предсказания и цели для вычисления mAP в конце эпохи
        # Сохраняем только каждый N-й батч для экономии памяти и ускорения
        if len(self.accumulated_predictions) < 20:  # Ограничиваем до 20 батчей
            # Сохраняем каждый 5-й батч или если это первые батчи
            if len(self.accumulated_predictions) < 5 or batch_size % 5 == 0:
                self.accumulated_predictions.append(pred.detach().cpu())
                self.accumulated_targets.append(target.detach().cpu())

        self.total_samples += batch_size

    def compute(self):
        """
        Вычисляет финальные метрики из накопленных компонентов
        """
        if self.total_samples == 0:
            return {}

        # Вычисляем Dice и IoU из накопленных компонентов
        smooth = 1e-6

        # Dice коэффициенты
        dice_denominator = self.pred_sum_sum + self.target_sum_sum
        mean_dice = (2.0 * self.intersection_sum + smooth) / (dice_denominator + smooth)

        # IoU коэффициенты
        union = self.pred_sum_sum + self.target_sum_sum - self.intersection_sum
        mean_iou = (self.intersection_sum + smooth) / (union + smooth)

        # Pixel accuracy
        pixel_acc = self.total_accuracy / self.total_samples

        # Precision и Recall по классам из накопленных TP, FP, FN
        class_precision = np.zeros(self.num_classes)
        class_recall = np.zeros(self.num_classes)
        class_f1 = np.zeros(self.num_classes)

        for cls in range(self.num_classes):
            tp = self.total_tp[cls]
            fp = self.total_fp[cls]
            fn = self.total_fn[cls]

            class_precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            class_recall[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # F1-score
            if class_precision[cls] + class_recall[cls] > 0:
                class_f1[cls] = 2 * class_precision[cls] * class_recall[cls] / (class_precision[cls] + class_recall[cls])
            else:
                class_f1[cls] = 0.0

        # Micro averages (правильное вычисление)
        total_tp = self.total_tp.sum()
        total_fp = self.total_fp.sum()
        total_fn = self.total_fn.sum()

        precision_micro = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall_micro = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

        # Определяем какие классы присутствуют в данных
        classes_present = (self.total_tp + self.total_fn) > 0

        # Macro averages (исключаем фон если нужно)
        if self.exclude_background and self.num_classes > 1:
            # Исключаем фон (класс 0) из средних метрик
            valid_classes_mask = classes_present[1:]
            if valid_classes_mask.any():
                overall_mean_dice = np.mean(mean_dice[1:][valid_classes_mask])
                overall_mean_iou = np.mean(mean_iou[1:][valid_classes_mask])
                precision_macro = np.mean(class_precision[1:][valid_classes_mask])
                recall_macro = np.mean(class_recall[1:][valid_classes_mask])
                f1_macro = np.mean(class_f1[1:][valid_classes_mask])
            else:
                overall_mean_dice = 0.0
                overall_mean_iou = 0.0
                precision_macro = 0.0
                recall_macro = 0.0
                f1_macro = 0.0
        else:
            # Включаем все классы
            if classes_present.any():
                overall_mean_dice = np.mean(mean_dice[classes_present])
                overall_mean_iou = np.mean(mean_iou[classes_present])
                precision_macro = np.mean(class_precision[classes_present])
                recall_macro = np.mean(class_recall[classes_present])
                f1_macro = np.mean(class_f1[classes_present])
            else:
                overall_mean_dice = 0.0
                overall_mean_iou = 0.0
                precision_macro = 0.0
                recall_macro = 0.0
                f1_macro = 0.0

        # mAP метрики (используем уже вычисленные значения)
        map50 = self.map50
        map50_95 = self.map50_95

        # Создаем словарь с результатами
        results = {
            'pixel_accuracy': pixel_acc,
            'mean_dice': overall_mean_dice,
            'mean_iou': overall_mean_iou,
            'mean_dice_with_bg': np.mean(mean_dice[classes_present]) if classes_present.any() else 0.0,
            'mean_iou_with_bg': np.mean(mean_iou[classes_present]) if classes_present.any() else 0.0,
            'map50': map50,
            'map50_95': map50_95,
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

        return results

    def compute_map_metrics(self):
        """
        Вычисляет mAP метрики в конце эпохи на накопленных данных
        """
        if not self.accumulated_predictions or not self.accumulated_targets:
            print("Предупреждение: нет накопленных данных для вычисления mAP")
            self.map50 = 0.0
            self.map50_95 = 0.0
            return

        try:
            print("Вычисление mAP метрик...")

            # Объединяем все накопленные батчи
            all_predictions = torch.cat(self.accumulated_predictions, dim=0)
            all_targets = torch.cat(self.accumulated_targets, dim=0)

            # Вычисляем mAP на объединенных данных
            map_results = calculate_map50_map95(
                all_predictions, all_targets,
                self.num_classes, self.exclude_background
            )

            self.map50 = map_results['mAP50']
            self.map50_95 = map_results['mAP50_95']

            print(f"mAP@0.5: {self.map50:.4f}, mAP@0.5:0.95: {self.map50_95:.4f}")

        except Exception as e:
            print(f"Ошибка при вычислении mAP: {e}")
            self.map50 = 0.0
            self.map50_95 = 0.0

        # Очищаем накопленные данные для экономии памяти
        self.accumulated_predictions.clear()
        self.accumulated_targets.clear()

    def print_results(self):
        """
        Выводит результаты в удобном формате
        """
        results = self.compute()

        print(f"Pixel Accuracy: {results['pixel_accuracy']:.4f}")
        if self.exclude_background:
            print(f"Mean Dice (без фона): {results['mean_dice']:.4f}")
            print(f"Mean IoU (без фона): {results['mean_iou']:.4f}")
            print(f"Mean Dice (с фоном): {results['mean_dice_with_bg']:.4f}")
            print(f"Mean IoU (с фоном): {results['mean_iou_with_bg']:.4f}")
        else:
            print(f"Mean Dice: {results['mean_dice']:.4f}")
            print(f"Mean IoU: {results['mean_iou']:.4f}")

        print(f"mAP@0.5: {results['map50']:.4f}")
        print(f"mAP@0.5:0.95: {results['map50_95']:.4f}")
        print(f"F1 (Macro): {results['f1_macro']:.4f}")
        print(f"F1 (Micro): {results['f1_micro']:.4f}")
        print(f"Precision (Macro): {results['precision_macro']:.4f}")
        print(f"Recall (Macro): {results['recall_macro']:.4f}")
        print(f"Precision (Micro): {results['precision_micro']:.4f}")
        print(f"Recall (Micro): {results['recall_micro']:.4f}")

        print("\nПо классам (только классы с данными):")
        for i, class_name in enumerate(self.class_names):
            dice = results['class_dice'][class_name]
            iou = results['class_iou'][class_name]
            precision = results['class_precision'][class_name]
            recall = results['class_recall'][class_name]
            f1 = results['class_f1'][class_name]

            # Показываем только классы, которые присутствуют в данных
            if self.total_tp[i] + self.total_fn[i] > 0:  # Есть истинные примеры этого класса
                print(f"{class_name}: Dice={dice:.4f}, IoU={iou:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

def calculate_boundary_iou(pred, target, num_classes, boundary_width=2):
    """
    Вычисляет IoU на границах объектов для более точной оценки качества сегментации
    """
    # Проверка размерностей
    if pred.dim() != 4 or target.dim() != 3:
        raise ValueError(f"Ожидаемые размерности: pred [B,C,H,W], target [B,H,W]. Получено: pred {pred.shape}, target {target.shape}")

    target = target.long().to(pred.device)
    pred_softmax = F.softmax(pred, dim=1)
    pred = torch.argmax(pred_softmax, dim=1)

    boundary_ious = []

    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        # Создаем маски границ с помощью морфологических операций
        pred_boundary = pred_cls - F.max_pool2d(pred_cls.unsqueeze(1), kernel_size=boundary_width*2+1, stride=1, padding=boundary_width).squeeze(1)
        target_boundary = target_cls - F.max_pool2d(target_cls.unsqueeze(1), kernel_size=boundary_width*2+1, stride=1, padding=boundary_width).squeeze(1)

        pred_boundary = (pred_boundary < 0).float()
        target_boundary = (target_boundary < 0).float()

        # Вычисляем IoU на границах
        intersection = (pred_boundary * target_boundary).sum()
        union = pred_boundary.sum() + target_boundary.sum() - intersection

        if union > 0:
            boundary_iou = intersection / union
        else:
            boundary_iou = 1.0 if (pred_boundary.sum() == 0 and target_boundary.sum() == 0) else 0.0

        boundary_ious.append(boundary_iou.item())

    return boundary_ious

def calculate_precision_recall_per_class(pred, target, num_classes):
    """
    Вычисляет Precision и Recall для каждого класса
    """
    # Проверка размерностей
    if pred.dim() != 4 or target.dim() != 3:
        raise ValueError(f"Ожидаемые размерности: pred [B,C,H,W], target [B,H,W]. Получено: pred {pred.shape}, target {target.shape}")

    target = target.long().to(pred.device)

    # Проверяем валидность значений в target
    if target.min() < 0 or target.max() >= num_classes:
        raise ValueError(f"Значения target должны быть в диапазоне [0, {num_classes-1}]. Получено: [{target.min()}, {target.max()}]")

    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)

    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()

    # Используем sklearn для вычисления precision и recall
    precision, recall, f1, support = precision_recall_fscore_support(
        target_np, pred_np, labels=range(num_classes), average=None, zero_division=0
    )

    return precision, recall, f1, support

def calculate_overall_precision_recall(pred, target, num_classes):
    """
    Вычисляет общие Precision и Recall (macro и micro averaging)
    """
    # Проверка размерностей
    if pred.dim() != 4 or target.dim() != 3:
        raise ValueError(f"Ожидаемые размерности: pred [B,C,H,W], target [B,H,W]. Получено: pred {pred.shape}, target {target.shape}")

    target = target.long().to(pred.device)

    # Проверяем валидность значений в target
    if target.min() < 0 or target.max() >= num_classes:
        raise ValueError(f"Значения target должны быть в диапазоне [0, {num_classes-1}]. Получено: [{target.min()}, {target.max()}]")

    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)

    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()

    # Macro averaging (среднее по классам)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        target_np, pred_np, labels=range(num_classes), average='macro', zero_division=0
    )

    # Micro averaging (общее по всем пикселям)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        target_np, pred_np, labels=range(num_classes), average='micro', zero_division=0
    )

    # Weighted averaging (взвешенное по поддержке классов)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        target_np, pred_np, labels=range(num_classes), average='weighted', zero_division=0
    )

    return {
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted
    }

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
    # Проверка размерностей
    if pred.dim() != 4 or target.dim() != 3:
        raise ValueError(f"Ожидаемые размерности: pred [B,C,H,W], target [B,H,W]. Получено: pred {pred.shape}, target {target.shape}")

    target = target.long().to(pred.device)

    # Проверяем валидность значений в target
    if target.min() < 0 or target.max() >= num_classes:
        raise ValueError(f"Значения target должны быть в диапазоне [0, {num_classes-1}]. Получено: [{target.min()}, {target.max()}]")

    pred_softmax = F.softmax(pred, dim=1)

    # Определяем классы для анализа
    start_cls = 1 if exclude_background else 0
    classes_to_analyze = list(range(start_cls, num_classes))

    results = {}

    for threshold in iou_thresholds:
        class_aps = []

        for cls in classes_to_analyze:
            # Получаем вероятности и бинарные маски для класса
            pred_probs = pred_softmax[:, cls]  # [B, H, W]
            target_binary = (target == cls).float()  # [B, H, W]

            # Если класс отсутствует в target, пропускаем
            if target_binary.sum() == 0:
                continue

            # Вычисляем AP для класса
            ap = calculate_ap_for_class(pred_probs, target_binary, threshold)
            class_aps.append(ap)

        # Средний AP по всем классам
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

    # Собираем все предсказания и истинные значения
    all_scores = []
    all_ious = []
    all_targets = []

    for b in range(batch_size):
        pred_prob = pred_probs[b]  # [H, W]
        target_mask = target_binary[b]  # [H, W]

        # Если нет истинных объектов в этом изображении, пропускаем
        if target_mask.sum() == 0:
            continue

        # Создаем различные пороги вероятности (уменьшено для ускорения)
        prob_thresholds = torch.linspace(0.3, 0.7, 5).to(pred_prob.device)  # 5 порогов вместо 9

        for prob_thresh in prob_thresholds:
            pred_binary = (pred_prob > prob_thresh).float()

            # Вычисляем IoU
            intersection = (pred_binary * target_mask).sum()
            union = pred_binary.sum() + target_mask.sum() - intersection

            if union > 0:
                iou = intersection / union
            else:
                iou = 1.0 if (pred_binary.sum() == 0 and target_mask.sum() == 0) else 0.0

            all_scores.append(pred_prob.max().item())  # Максимальная вероятность как score
            all_ious.append(iou.item())
            all_targets.append(1.0)  # Всегда есть истинный объект

    if not all_scores:
        return 0.0

    # Сортируем по убыванию score
    sorted_indices = np.argsort(all_scores)[::-1]
    sorted_ious = np.array(all_ious)[sorted_indices]

    # Вычисляем precision и recall
    tp = (sorted_ious >= iou_threshold).astype(float)
    fp = (sorted_ious < iou_threshold).astype(float)

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / len(all_targets) if len(all_targets) > 0 else np.array([0])
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

    # Вычисляем AP как площадь под кривой precision-recall
    ap = compute_ap(recalls, precisions)

    return ap

def compute_ap(recalls, precisions):
    """
    Вычисляет Average Precision по кривой precision-recall
    """
    # Добавляем точки (0,1) и (1,0) для корректного вычисления
    recalls = np.concatenate(([0], recalls, [1]))
    precisions = np.concatenate(([0], precisions, [0]))

    # Делаем precision монотонно убывающей
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Находим точки, где recall изменяется
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1

    # Вычисляем AP как сумму площадей прямоугольников
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
    # Используем только ключевые пороги вместо всех 10
    iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]  # 5 порогов вместо 10
    map50_95_result = calculate_map_segmentation(pred, target, num_classes, iou_thresholds, exclude_background)

    # Усредняем по всем порогам
    all_maps = [map50_95_result.get(f'mAP@{thresh}', 0.0) for thresh in iou_thresholds]
    map50_95 = np.mean(all_maps)

    return {
        'mAP50': map50,
        'mAP50_95': map50_95
    }
    
### Загрузка чекпоинта ###
    
def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cuda'):
    """
    Загружает чекпоинт модели с историей обучения

    Args:
        checkpoint_path: путь к файлу чекпоинта
        model: модель для загрузки весов
        optimizer: оптимизатор (опционально)
        scheduler: планировщик (опционально)
        device: устройство для загрузки

    Returns:
        dict: словарь с информацией о чекпоинте
    """
    print(f"\n{'='*60}")
    print(f"ЗАГРУЗКА ЧЕКПОИНТА")
    print(f"{'='*60}")
    print(f"Файл: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Загружаем веса модели
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"*** Загружены веса модели")

    # Загружаем состояние оптимизатора
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"*** Загружено состояние оптимизатора")

    # Загружаем состояние планировщика
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"*** Загружено состояние планировщика")


    # Извлекаем информацию о чекпоинте
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

# Визуализация полной истории обучения
def plot_training_history(history, save_dir):
    """
    Визуализирует полную историю обучения независимо от количества этапов

    Args:
        history: словарь с историей обучения
        save_dir: директория для сохранения графиков
    """
    print("\n" + "="*60)
    print("СОЗДАНИЕ ГРАФИКОВ ИСТОРИИ ОБУЧЕНИЯ")
    print("="*60)

    # Создаем директорию для графиков
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

    # 4. mAP метрики
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_map50'], label='Train mAP@0.5', marker='o', markersize=3)
    plt.plot(epochs, history['val_map50'], label='Val mAP@0.5', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('mAP@0.5')
    plt.title('Mean Average Precision @ IoU=0.5')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_map50_95'], label='Train mAP@0.5:0.95', marker='o', markersize=3)
    plt.plot(epochs, history['val_map50_95'], label='Val mAP@0.5:0.95', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('mAP@0.5:0.95')
    plt.title('Mean Average Precision @ IoU=0.5:0.95')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / 'map_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  * map_metrics.png")

    # 5. F1 Score
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_f1_macro'], label='Train F1 (Macro)', marker='o', markersize=3)
    plt.plot(epochs, history['val_f1_macro'], label='Val F1 (Macro)', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score (Macro)')
    plt.title('F1 Score - Macro Average')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_f1_micro'], label='Train F1 (Micro)', marker='o', markersize=3)
    plt.plot(epochs, history['val_f1_micro'], label='Val F1 (Micro)', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score (Micro)')
    plt.title('F1 Score - Micro Average')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / 'f1_scores.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  * f1_scores.png")


    # 6. Сводный график всех основных метрик
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train', marker='o', markersize=2)
    plt.plot(epochs, history['val_loss'], label='Val', marker='s', markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 2)
    plt.plot(epochs, history['train_dice'], label='Train', marker='o', markersize=2)
    plt.plot(epochs, history['val_dice'], label='Val', marker='s', markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.title('Dice Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 3)
    plt.plot(epochs, history['train_iou'], label='Train', marker='o', markersize=2)
    plt.plot(epochs, history['val_iou'], label='Val', marker='s', markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('IoU Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 4)
    plt.plot(epochs, history['train_map50'], label='Train', marker='o', markersize=2)
    plt.plot(epochs, history['val_map50'], label='Val', marker='s', markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('mAP@0.5')
    plt.title('mAP@0.5')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 5)
    plt.plot(epochs, history['train_f1_macro'], label='Train', marker='o', markersize=2)
    plt.plot(epochs, history['val_f1_macro'], label='Val', marker='s', markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('F1 (Macro)')
    plt.title('F1 Score (Macro)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 6)
    plt.plot(epochs, history['train_accuracy'], label='Train', marker='o', markersize=2)
    plt.plot(epochs, history['val_accuracy'], label='Val', marker='s', markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Pixel Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle(f'Training History - All Metrics (Total: {total_epochs} epochs)', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(plots_dir / 'all_metrics_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  * all_metrics_summary.png")

    print(f"\n* Все графики сохранены в: {plots_dir}")
    print(f"{'='*60}\n")
    
### Вывод истории обучения ###

def load_training_history(history_path):
    """
    Загружает историю обучения из JSON файла или чекпоинта
    """
    history_path = Path(history_path)

    # Проверяем существование файла
    if not history_path.exists():
        raise FileNotFoundError(f"Файл не найден: {history_path}")

    try:
        if history_path.suffix == '.json':
            # Загружаем из JSON файла
            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
        elif history_path.suffix == '.pth':
            # Загружаем из чекпоинта PyTorch (безопасно)
            import torch
            try:
                checkpoint = torch.load(history_path, map_location='cpu', weights_only=True)
            except:
                # Fallback для старых чекпоинтов
                checkpoint = torch.load(history_path, map_location='cpu', weights_only=False)
            history = checkpoint.get('training_history', {})
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {history_path.suffix}. Поддерживаются: .json, .pth")
    except Exception as e:
        raise RuntimeError(f"Ошибка при загрузке файла {history_path}: {e}")

    # Валидация структуры данных
    if not isinstance(history, dict):
        raise ValueError("История обучения должна быть словарем")

    if 'epoch' not in history or not history['epoch']:
        raise ValueError("История обучения не содержит данных об эпохах")

    return history


def plot_metrics_comparison(history, save_dir=None, show_plots=True):
    """
    Строит сравнительные графики ключевых метрик с обработкой ошибок
    """
    if not history or 'epoch' not in history or not history['epoch']:
        print("История обучения пуста или некорректна")
        return None

    epochs = history['epoch']

    # Функция для безопасного получения метрики
    def safe_get_metric(key, default_value=0.0):
        if key in history and len(history[key]) == len(epochs):
            return history[key]
        else:
            print(f"Предупреждение: метрика '{key}' отсутствует или имеет неправильную длину")
            return [default_value] * len(epochs)

    try:
        # Создаем фигуру для ключевых метрик (увеличиваем до 2x3)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Ключевые метрики обучения', fontsize=16)

        # 1. Loss
        ax = axes[0, 0]
        train_loss = safe_get_metric('train_loss')
        val_loss = safe_get_metric('val_loss')
        ax.plot(epochs, train_loss, 'b-', label='Train', linewidth=2, marker='o', markersize=3)
        ax.plot(epochs, val_loss, 'r-', label='Validation', linewidth=2, marker='s', markersize=3)
        ax.set_title('Loss', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Dice
        ax = axes[0, 1]
        train_dice = safe_get_metric('train_dice')
        val_dice = safe_get_metric('val_dice')
        ax.plot(epochs, train_dice, 'b-', label='Train', linewidth=2, marker='o', markersize=3)
        ax.plot(epochs, val_dice, 'r-', label='Validation', linewidth=2, marker='s', markersize=3)
        ax.set_title('Dice Coefficient', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dice')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. IoU
        ax = axes[0, 2]
        train_iou = safe_get_metric('train_iou')
        val_iou = safe_get_metric('val_iou')
        ax.plot(epochs, train_iou, 'b-', label='Train', linewidth=2, marker='o', markersize=3)
        ax.plot(epochs, val_iou, 'r-', label='Validation', linewidth=2, marker='s', markersize=3)
        ax.set_title('IoU Score', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('IoU')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. mAP метрики
        ax = axes[1, 0]
        train_map50 = safe_get_metric('train_map50')
        val_map50 = safe_get_metric('val_map50')
        train_map50_95 = safe_get_metric('train_map50_95')
        val_map50_95 = safe_get_metric('val_map50_95')

        ax.plot(epochs, train_map50, 'b-', label='Train mAP@0.5', linewidth=2, marker='o', markersize=3)
        ax.plot(epochs, val_map50, 'r-', label='Val mAP@0.5', linewidth=2, marker='s', markersize=3)
        ax.plot(epochs, train_map50_95, 'b--', label='Train mAP@0.5:0.95', linewidth=2, marker='^', markersize=3)
        ax.plot(epochs, val_map50_95, 'r--', label='Val mAP@0.5:0.95', linewidth=2, marker='v', markersize=3)
        ax.set_title('mAP Metrics', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 5. F1 Score
        ax = axes[1, 1]
        train_f1_macro = safe_get_metric('train_f1_macro')
        val_f1_macro = safe_get_metric('val_f1_macro')
        train_f1_micro = safe_get_metric('train_f1_micro')
        val_f1_micro = safe_get_metric('val_f1_micro')

        ax.plot(epochs, train_f1_macro, 'b-', label='Train F1 (Macro)', linewidth=2, marker='o', markersize=3)
        ax.plot(epochs, val_f1_macro, 'r-', label='Val F1 (Macro)', linewidth=2, marker='s', markersize=3)
        ax.plot(epochs, train_f1_micro, 'b--', label='Train F1 (Micro)', linewidth=2, marker='^', markersize=3)
        ax.plot(epochs, val_f1_micro, 'r--', label='Val F1 (Micro)', linewidth=2, marker='v', markersize=3)
        ax.set_title('F1 Score', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 6. Precision vs Recall
        ax = axes[1, 2]
        train_prec = safe_get_metric('train_precision_macro')
        val_prec = safe_get_metric('val_precision_macro')
        train_recall = safe_get_metric('train_recall_macro')
        val_recall = safe_get_metric('val_recall_macro')

        ax.plot(epochs, train_prec, 'b-', label='Train Precision', linewidth=2, marker='o', markersize=3)
        ax.plot(epochs, val_prec, 'r-', label='Val Precision', linewidth=2, marker='s', markersize=3)
        ax.plot(epochs, train_recall, 'b--', label='Train Recall', linewidth=2, marker='^', markersize=3)
        ax.plot(epochs, val_recall, 'r--', label='Val Recall', linewidth=2, marker='v', markersize=3)
        ax.set_title('Precision & Recall', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Сохраняем график
        if save_dir:
            try:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / 'key_metrics.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Ключевые метрики сохранены: {save_path}")
            except Exception as e:
                print(f"Ошибка при сохранении ключевых метрик: {e}")

        if show_plots:
            try:
                plt.show()
            except Exception as e:
                print(f"Не удалось показать графики: {e}")

        return fig

    except Exception as e:
        print(f"Ошибка при создании графиков ключевых метрик: {e}")
        return None

def print_training_summary(history):
    """
    Выводит сводку обучения с обработкой отсутствующих метрик
    """
    if not history or 'epoch' not in history or not history['epoch']:
        print("История обучения пуста")
        return

    print("=" * 60)
    print("СВОДКА ОБУЧЕНИЯ")
    print("=" * 60)

    total_epochs = len(history['epoch'])
    print(f"Общее количество эпох: {total_epochs}")

    if total_epochs > 0:
        # Функция для получения метрики
        def safe_get_metric(key):
            if key in history and len(history[key]) == total_epochs and history[key]:
                return history[key]
            return None

        # Получаем метрики
        val_dice = safe_get_metric('val_dice')
        val_iou = safe_get_metric('val_iou')
        val_loss = safe_get_metric('val_loss')
        train_dice = safe_get_metric('train_dice')
        train_iou = safe_get_metric('train_iou')
        train_loss = safe_get_metric('train_loss')

        # Лучшие результаты
        print(f"\nЛучшие результаты:")
        if val_dice:
            best_val_dice_idx = np.argmax(val_dice)
            print(f"Лучший Val Dice: {val_dice[best_val_dice_idx]:.4f} (эпоха {history['epoch'][best_val_dice_idx]})")

        if val_iou:
            best_val_iou_idx = np.argmax(val_iou)
            print(f"Лучший Val IoU: {val_iou[best_val_iou_idx]:.4f} (эпоха {history['epoch'][best_val_iou_idx]})")

        if val_loss:
            min_val_loss_idx = np.argmin(val_loss)
            print(f"Минимальный Val Loss: {val_loss[min_val_loss_idx]:.4f} (эпоха {history['epoch'][min_val_loss_idx]})")

        # Финальные результаты
        print(f"\nФинальные результаты (эпоха {total_epochs}):")

        if train_loss and val_loss:
            print(f"Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}")

        if train_dice and val_dice:
            print(f"Train Dice: {train_dice[-1]:.4f}, Val Dice: {val_dice[-1]:.4f}")

        if train_iou and val_iou:
            print(f"Train IoU: {train_iou[-1]:.4f}, Val IoU: {val_iou[-1]:.4f}")

        # mAP метрики
        train_map50 = safe_get_metric('train_map50')
        val_map50 = safe_get_metric('val_map50')
        if train_map50 and val_map50:
            print(f"Train mAP@0.5: {train_map50[-1]:.4f}, Val mAP@0.5: {val_map50[-1]:.4f}")

        train_map50_95 = safe_get_metric('train_map50_95')
        val_map50_95 = safe_get_metric('val_map50_95')
        if train_map50_95 and val_map50_95:
            print(f"Train mAP@0.5:0.95: {train_map50_95[-1]:.4f}, Val mAP@0.5:0.95: {val_map50_95[-1]:.4f}")

        # F1 Score
        train_f1_macro = safe_get_metric('train_f1_macro')
        val_f1_macro = safe_get_metric('val_f1_macro')
        if train_f1_macro and val_f1_macro:
            print(f"Train F1 (Macro): {train_f1_macro[-1]:.4f}, Val F1 (Macro): {val_f1_macro[-1]:.4f}")

        train_f1_micro = safe_get_metric('train_f1_micro')
        val_f1_micro = safe_get_metric('val_f1_micro')
        if train_f1_micro and val_f1_micro:
            print(f"Train F1 (Micro): {train_f1_micro[-1]:.4f}, Val F1 (Micro): {val_f1_micro[-1]:.4f}")

        train_precision = safe_get_metric('train_precision_macro')
        val_precision = safe_get_metric('val_precision_macro')
        if train_precision and val_precision:
            print(f"Train Precision: {train_precision[-1]:.4f}, Val Precision: {val_precision[-1]:.4f}")

        train_recall = safe_get_metric('train_recall_macro')
        val_recall = safe_get_metric('val_recall_macro')
        if train_recall and val_recall:
            print(f"Train Recall: {train_recall[-1]:.4f}, Val Recall: {val_recall[-1]:.4f}")

### Оценка на тестовой выборке ###

def visualize_predictions(images, masks, predictions, class_names, save_path=None, num_samples=4):
    """
    Визуализирует предсказания модели
    """
    num_samples = min(num_samples, len(images))

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx in range(num_samples):
        # Оригинальное изображение
        img = images[idx].cpu().numpy().squeeze()
        axes[idx, 0].imshow(img, cmap='gray')
        axes[idx, 0].set_title('Original Image')
        axes[idx, 0].axis('off')

        # Истинная маска
        mask = masks[idx].cpu().numpy()
        axes[idx, 1].imshow(mask, cmap='tab20')
        axes[idx, 1].set_title('Ground Truth')
        axes[idx, 1].axis('off')

        # Предсказание
        pred = predictions[idx].cpu().numpy()
        axes[idx, 2].imshow(pred, cmap='tab20')
        axes[idx, 2].set_title('Prediction')
        axes[idx, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Визуализация сохранена в: {save_path}")

    plt.close()


def save_metrics_to_file(metrics_dict, save_path):
    """
    Сохраняет метрики в JSON файл
    """
    # Конвертируем numpy типы в Python типы для JSON
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_metrics = convert_to_serializable(metrics_dict)

    with open(save_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)

    print(f"Метрики сохранены в: {save_path}")


def run_inference(model, test_loader, device, num_classes, class_names, save_dir=None):
    """
    Запускает инференс на test подвыборке и вычисляет все метрики
    """
    model.eval()

    # Инициализируем метрики (исключаем фон из средних метрик)
    test_metrics = SegmentationMetrics(num_classes, class_names, exclude_background=True)

    # Для визуализации
    sample_images = []
    sample_masks = []
    sample_predictions = []

    print("Запуск инференса на test подвыборке...")

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc='Inference')):
            images, masks = images.to(device), masks.to(device)

            # Предсказание
            outputs = model(images)

            # Обновляем метрики
            test_metrics.update(outputs, masks)

            # Сохраняем несколько примеров для визуализации
            if batch_idx < 2:  # Первые 2 батча
                pred_classes = torch.argmax(F.softmax(outputs, dim=1), dim=1)
                sample_images.extend(images.cpu())
                sample_masks.extend(masks.cpu())
                sample_predictions.extend(pred_classes.cpu())

    # Вычисляем mAP метрики
    print("\nВычисление mAP метрик...")
    test_metrics.compute_map_metrics()

    # Получаем финальные метрики
    results = test_metrics.compute()

    # Выводим результаты
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ИНФЕРЕНСА НА TEST ПОДВЫБОРКЕ")
    print("="*60)
    test_metrics.print_results()
    print("="*60)

    # Сохраняем результаты
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Сохраняем метрики в JSON
        metrics_path = os.path.join(save_dir, 'test_metrics.json')
        save_metrics_to_file(results, metrics_path)

        # Визуализируем предсказания
        if sample_images:
            vis_path = os.path.join(save_dir, 'test_predictions_visualization.png')
            visualize_predictions(
                sample_images[:8],
                sample_masks[:8],
                sample_predictions[:8],
                class_names,
                save_path=vis_path,
                num_samples=min(8, len(sample_images))
            )

        # Сохраняем детальный отчет в текстовый файл
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

            f.write("МЕТРИКИ ПО КЛАССАМ:\n")
            f.write("-"*60 + "\n")

            for i, class_name in enumerate(class_names):
                dice = results['class_dice'][class_name]
                iou = results['class_iou'][class_name]
                precision = results['class_precision'][class_name]
                recall = results['class_recall'][class_name]
                f1 = results['class_f1'][class_name]

                # Показываем только классы с данными
                if test_metrics.total_tp[i] + test_metrics.total_fn[i] > 0:
                    f.write(f"\n{class_name}:\n")
                    f.write(f"  Dice: {dice:.4f}\n")
                    f.write(f"  IoU: {iou:.4f}\n")
                    f.write(f"  F1: {f1:.4f}\n")
                    f.write(f"  Precision: {precision:.4f}\n")
                    f.write(f"  Recall: {recall:.4f}\n")

        print(f"\nДетальный отчет сохранен в: {report_path}")

    return results

### Инференс ###

def load_model(model_path, num_classes=33, device='cuda'):
    """
    Загрузка обученной модели 
    """
    # Определяем архитектуру модели (должна совпадать с обучением)
    class DoubleConv(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super(DoubleConv, self).__init__()
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.conv(x)

    class UNet(torch.nn.Module):
        def __init__(self, in_channels=1, out_channels=33, features=[64, 128, 256, 512]):
            super(UNet, self).__init__()
            self.ups = torch.nn.ModuleList()
            self.downs = torch.nn.ModuleList()
            self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

            # Down part of UNet
            for feature in features:
                self.downs.append(DoubleConv(in_channels, feature))
                in_channels = feature

            # Bottleneck
            self.bottleneck = DoubleConv(features[-1], features[-1]*2)

            # Up part of UNet
            for feature in reversed(features):
                self.ups.append(
                    torch.nn.ConvTranspose2d(
                        feature*2, feature, kernel_size=2, stride=2,
                    )
                )
                self.ups.append(DoubleConv(feature*2, feature))

            self.final_conv = torch.nn.Conv2d(features[0], out_channels, kernel_size=1)

        def forward(self, x):
            skip_connections = []

            for down in self.downs:
                x = down(x)
                skip_connections.append(x)
                x = self.pool(x)

            x = self.bottleneck(x)
            skip_connections = skip_connections[::-1]

            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)
                skip_connection = skip_connections[idx//2]

                if x.shape != skip_connection.shape:
                    x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

                concat_skip = torch.cat((skip_connection, x), dim=1)
                x = self.ups[idx+1](concat_skip)

            return self.final_conv(x)

    # Создаем модель и загружаем веса
    model = UNet(in_channels=1, out_channels=num_classes)

    
    try:
        # Пробуем загрузить с weights_only=False
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError as e:
        # Если версия PyTorch не поддерживает weights_only, загружаем по-старому
        if "weights_only" in str(e):
            checkpoint = torch.load(model_path, map_location=device)
        else:
            raise e

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Пробуем разные варианты ключей
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"Модель успешно загружена с устройства: {device}")
    return model

def preprocess_image(image_path, transform, img_size=512):
    """
    Предобработка изображения для инференса
    """
    # Загружаем изображение
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    original_h, original_w = image.shape
    original_image = image.copy()

    # Ресайзим изображение
    image_resized = cv2.resize(image, (img_size, img_size))

    # Добавляем канал для совместимости с Albumentations
    if len(image_resized.shape) == 2:
        image_resized = np.expand_dims(image_resized, axis=2)  # [H, W] -> [H, W, 1]

    # Применяем трансформации
    transformed = transform(image=image_resized)
    image_tensor = transformed['image']

    # Убеждаемся что тензор имеет правильную размерность [1, H, W]
    if len(image_tensor.shape) == 3 and image_tensor.shape[0] == 1:
        # Уже в формате [C, H, W]
        pass
    elif len(image_tensor.shape) == 3 and image_tensor.shape[2] == 1:
        image_tensor = image_tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
    else:
        image_tensor = image_tensor.unsqueeze(0)  # [H, W] -> [C, H, W]

    return image_tensor.unsqueeze(0), original_image, (original_h, original_w)

def postprocess_mask(mask_tensor, original_size, img_size=512):
    """
    Постобработка маски: преобразование к исходному размеру
    """
    # Применяем softmax и получаем предсказанные классы
    mask_pred = torch.softmax(mask_tensor, dim=1)
    mask_pred = torch.argmax(mask_pred, dim=1)  # [1, H, W]

    # Конвертируем в numpy
    mask_np = mask_pred.squeeze().cpu().numpy().astype(np.uint8)

    # Ресайзим к исходному размеру
    original_h, original_w = original_size
    mask_resized = cv2.resize(mask_np, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    return mask_resized

def create_colored_mask(mask, class_names, alpha=0.7):
    """
    Создание цветной маски с разными цветами для каждого класса
    """
    num_classes = len(class_names)

    # Используем фиксированную палитру для лучшей воспроизводимости
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
    """
    Добавление подписей классов на изображение
    """
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

            # Вычисляем позицию текста (центр ограничивающего прямоугольника)
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

def visualize_results(original_image, mask, class_names, save_path=None):
    """
    Визуализация результатов сегментации
    """
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

def inference_single_image(model_path, image_path, data_yaml_path, transform, output_dir='output', device='cuda'):
    """
    Инференс на одном изображении
    """
    # Преобразуем output_dir в Path если это строка
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    # Создаем выходную директорию
    output_dir.mkdir(exist_ok=True)

    # Загружаем конфигурацию датасета для получения имен классов
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    num_classes = config['nc'] + 1  # 32 зуба + фон
    class_names = ['Background'] + config['names']  # Добавляем фон как класс 0

    print(f"Количество классов: {num_classes}")
    print("Имена классов:", class_names)

    # Загружаем модель
    print("Загрузка модели...")
    model = load_model(model_path, num_classes, device)

    # Предобработка изображения
    print("Предобработка изображения...")
    image_tensor, original_image, original_size = preprocess_image(image_path, transform)
    image_tensor = image_tensor.to(device)

    # Инференс
    print("Выполнение инференса...")
    with torch.no_grad():
        output = model(image_tensor)

    # Постобработка маски
    print("Постобработка маски...")
    mask = postprocess_mask(output, original_size)

    # Визуализация результатов
    print("Визуализация результатов...")
    image_name = Path(image_path).stem
    save_path = output_dir / f'{image_name}_result.png'

    result_image = visualize_results(original_image, mask, class_names, save_path)

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

def inference_multiple_images(model_path, image_dir, data_yaml_path, output_dir='output', device='cuda'):
    """
    Инференс на нескольких изображениях в директории
    """
    # Преобразуем пути в Path объекты
    image_dir = Path(image_dir)
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    # Проверяем существование директории
    if not image_dir.exists():
        raise FileNotFoundError(f"Директория {image_dir} не существует")

    # Получаем список всех файлов изображений (включая поддиректории)
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

    # Выводим список найденных файлов для отладки
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
                model_path, image_path, data_yaml_path, img_output_dir, device
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

