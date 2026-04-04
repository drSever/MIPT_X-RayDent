#############################################################
# Аугментации для Mask R-CNN с использованием Albumentations
#############################################################

import numpy as np
import cv2
import torch
import copy
from typing import List, Dict, Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode
import pycocotools.mask as mask_util


def get_train_transforms():
    """
    Аугментации для ортопантомограмм
    """
    return A.Compose([
        # ------------------------------------------------------------
        # 0. ГОРИЗОНТАЛЬНЫЙ ФЛИП
        # ------------------------------------------------------------
        # ВАЖНО: флип применяется ОТДЕЛЬНО от этого Compose — в custom_mapper,
        # потому что после флипа нужно менять category_ids (FDI номера зубов).
        # Albumentations не поддерживает изменение labels внутри трансформации.
        # Здесь флип НЕ включён намеренно — см. custom_mapper.
        # Грубо говоря - после флипа надо зеркалить метки классов.
        # ------------------------------------------------------------
        # 1. ГЕОМЕТРИЧЕСКИЕ ПРЕОБРАЗОВАНИЯ
        # ------------------------------------------------------------
        A.Affine(
            scale=(0.95, 1.05),               # Масштаб: случайное увеличение/уменьшение до ±5%
            translate_percent=(0.03, 0.03),   # Сдвиг по горизонтали/вертикали до 3% от размера
            rotate=(-3, 3),                   # Поворот до ±3 градусов
            p=0.5,                            # Вероятность применения 50%
            border_mode=cv2.BORDER_CONSTANT,  # Новые области заполняются константой (чёрным)
            fill=0,                           # Значение для заполнения на изображении (0 — чёрный)
            fill_mask=0                       # Значение для заполнения на маске (0 — фон)
        ),
        # ОПИСАНИЕ: лёгкие аффинные искажения имитируют небольшие повороты головы пациента,
        # разный масштаб съёмки и смещения челюсти в кадре. Все изменения незначительны,
        # чтобы не нарушить анатомическую структуру зубного ряда.
        
        # ------------------------------------------------------------
        # 2. МЯГКИЕ ЭЛАСТИЧНЫЕ ДЕФОРМАЦИИ 
        # ------------------------------------------------------------
        A.ElasticTransform(
            alpha=0.5,                            # Интенсивность сдвига пикселей (малое значение -> слабая деформация)
            sigma=25,                             # Степень размытия поля деформации (сглаживает искажения)
            p=0.1,                                # Вероятность 10% — очень осторожное применение
            mask_interpolation=cv2.INTER_NEAREST  # Для масок — интерполяция без сглаживания
        ),
        # ОПИСАНИЕ: ElasticTransform моделирует незначительные неоднородности тканей или
        # естественные искривления челюсти. Параметры подобраны так, чтобы зубы оставались
        # узнаваемыми, но контуры слегка «дышали». 
        
        # ------------------------------------------------------------
        # 3. КОРРЕКЦИЯ КОНТРАСТА И ЯРКОСТИ (РЕНТГЕН-СПЕЦИФИЧНЫЕ)
        # ------------------------------------------------------------
        A.CLAHE(
            clip_limit=2.0,             # Порог ограничения гистограммы (выше — сильнее контраст)
            tile_grid_size=(8, 8),      # Размер ячеек для локального выравнивания на изображении (ОПТГ)
            p=0.5
        ),
        # ОПИСАНИЕ: CLAHE (Contrast Limited Adaptive Histogram Equalization) —
        # стандартный метод для рентгеновских снимков. Улучшает локальный контраст,
        # делая более чёткими границы зубов и корней, особенно на затемнённых участках.
        # Параметры (8,8) и clip_limit=2.0 дают умеренное усиление.
        
        A.RandomBrightnessContrast(
            brightness_limit=0.08,      # Изменение яркости до ±8%
            contrast_limit=0.08,        # Изменение контраста до ±8%
            p=0.5
        ),
        # Описание: Имитирует вариации экспозиции при съёмке — разные аппараты,
        # настройки яркости, толщина мягких тканей.
        
        # ------------------------------------------------------------
        # 4. ИМИТАЦИЯ АРТЕФАКТОВ (CoarseDropout) 
        # ------------------------------------------------------------
        A.CoarseDropout(
            num_holes_range=(1, 2),         # Количество прямоугольных артефактов: 1 или 2
            hole_height_range=(0.02, 0.04), # Высота артефакта 2–4% от высоты изображения
            hole_width_range=(0.02, 0.04),  # Ширина артефакта 2–4% от ширины
            fill=0,                         # Заливка артефакта чёрным (0) на изображении
            fill_mask=0,                    # Заливка артефакта фоном (0) на маске
            p=0.05                          # Вероятность 5% — редкое событие
        ),
        A.CoarseDropout(
            num_holes_range=(1, 2),
            hole_height_range=(0.02, 0.04),
            hole_width_range=(0.02, 0.04),
            fill=128,                       # Заливка артефакта серым (128) на изображении
            fill_mask=0,
            p=0.05
        ),
        A.CoarseDropout(
            num_holes_range=(1, 2),
            hole_height_range=(0.02, 0.04),
            hole_width_range=(0.02, 0.04),
            fill=255,                       # Заливка артефакта белым (255) на изображении
            fill_mask=0,
            p=0.05
        ),
        # ОПИСАНИЕ: CoarseDropout заменяет случайные прямоугольные области на указанный цвет.
        # В контексте ортопантомограмм это имитирует:
        #   - артефакты движения (смазанные участки);
        #   - наложение посторонних предметов (зажимы, маркеры);
        #   - отсутствие части зуба (например, разрушенная коронка);
        #   - переэкспонированные участки, пломбы, коронки, металлические вкладки.
        # Маленькая вероятность и небольшие размеры артефактов (2–4%) предотвращают
        # чрезмерное зашумление датасета и позволяют модели научиться игнорировать
        # локальные дефекты.
        
        # ------------------------------------------------------------
        # 5. ДОБАВЛЕНИЕ ШУМА (GaussNoise)
        # ------------------------------------------------------------
        A.GaussNoise(
            std_range=(0.01, 0.04),  
            p=0.2
        ),
        # ОПИСАНИЕ: Добавляет гауссов шум, имитируя зернистость рентгеновской
        # плёнки или электронные шумы. Параметры 
        # дают умеренный уровень шума, достаточный для повышения робастности модели,
        # но не разрушающий мелкие детали (например, периодонтальную щель).
        
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['category_ids'],
        min_area=1.0,
        min_visibility=0.3
    ))


# Маппинг category_id (1-based COCO) при горизонтальном флипе.
# При флипе левая и правая стороны меняются: квадрант 1<->2, квадрант 3<->4.
# FDI: 11-18 <-> 21-28,  31-38 <-> 41-48
# category_id = class_id + 1, где class_id: 11->0, 12->1, ..., 48->31
_FLIP_CATEGORY_ID_MAP = {
    # Квадрант 1 (11-18, cat_id 1-8) <-> Квадрант 2 (21-28, cat_id 9-16)
    1: 9,   2: 10,  3: 11,  4: 12,  5: 13,  6: 14,  7: 15,  8: 16,
    9: 1,  10: 2,  11: 3,  12: 4,  13: 5,  14: 6,  15: 7,  16: 8,
    # Квадрант 3 (31-38, cat_id 17-24) <-> Квадрант 4 (41-48, cat_id 25-32)
    17: 25, 18: 26, 19: 27, 20: 28, 21: 29, 22: 30, 23: 31, 24: 32,
    25: 17, 26: 18, 27: 19, 28: 20, 29: 21, 30: 22, 31: 23, 32: 24,
}


def flip_category_ids(category_ids):
    """Заменяет category_ids после горизонтального флипа."""
    return [_FLIP_CATEGORY_ID_MAP.get(cid, cid) for cid in category_ids]


def get_flip_transform():
    """
    Отдельный Compose только для горизонтального флипа.
    Применяется в custom_mapper с последующей заменой category_ids.
    p=0.5 - флипается половина изображений.
    """
    return A.Compose([
        A.HorizontalFlip(p=1.0),  # p=1.0 — сам Compose вызывается с вероятностью 0.5
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['category_ids'],
        min_area=1.0,
        min_visibility=0.3,
    ))


def polygon_to_mask(segmentation, height, width):
    """Конвертирует полигон в бинарную маску"""
    if isinstance(segmentation, list):
        # Полигон формат
        rles = mask_util.frPyObjects(segmentation, height, width)
        rle = mask_util.merge(rles)
    else:
        # RLE формат
        rle = segmentation
    
    mask = mask_util.decode(rle)
    return mask


def mask_to_polygon(mask):
    """Конвертирует бинарную маску в полигон"""
    # Убеждаемся что маска бинарная
    mask = (mask > 0).astype(np.uint8)
    
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    for contour in contours:
        # Минимум 3 точки для полигона
        if len(contour) >= 3:
            # Упрощаем контур
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 3:
                polygon = approx.flatten().tolist()
                # Проверяем что полигон валидный
                if len(polygon) >= 6:  # минимум 3 точки (x,y)
                    polygons.append(polygon)
    
    return polygons


def custom_mapper(dataset_dict, augmentations=None, flip_transform=None):
    """
    mapper для DataLoader с поддержкой Albumentations

    Args:
        dataset_dict: словарь с данными изображения из датасета
        augmentations: Albumentations compose объект (геометрия, цвет, шум)
        flip_transform: Albumentations compose для горизонтального флипа
                        (применяется отдельно с заменой FDI category_ids)

    Returns:
        dict: обработанные данные для Detectron2
    """
    dataset_dict = copy.deepcopy(dataset_dict)
    
    # Загружаем изображение
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    height, width = image.shape[:2]
    
    # Конвертируем в RGB для Albumentations
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Подготавливаем аннотации для Albumentations
    if "annotations" in dataset_dict:
        annos = dataset_dict["annotations"]
        
        # Извлекаем bbox и category_ids
        bboxes = []
        category_ids = []
        masks = []
        
        for anno in annos:
            # Bbox в формате COCO [x, y, width, height]
            bbox = anno["bbox"]
            bboxes.append(bbox)
            category_ids.append(anno["category_id"])
            
            # Конвертируем полигон в маску
            if "segmentation" in anno:
                mask = polygon_to_mask(anno["segmentation"], height, width)
                masks.append(mask)
        
        # Применяем аугментации
        if augmentations is not None and len(bboxes) > 0 and len(masks) > 0:
            try:
                transformed = augmentations(
                    image=image_rgb,
                    masks=masks,
                    bboxes=bboxes,
                    category_ids=category_ids
                )
                
                image_rgb = transformed["image"]
                bboxes = transformed["bboxes"]
                category_ids = transformed["category_ids"]
                masks = transformed["masks"]

            except Exception as e:
                # Если аугментация не удалась, используем оригинальные данные
                print(f"Augmentation failed: {e}")

        # Горизонтальный флип с заменой FDI category_ids (p=0.5)
        if flip_transform is not None and len(bboxes) > 0 and len(masks) > 0:
            import random
            if random.random() < 0.5:
                try:
                    flipped = flip_transform(
                        image=image_rgb,
                        masks=masks,
                        bboxes=bboxes,
                        category_ids=category_ids,
                    )
                    image_rgb  = flipped["image"]
                    bboxes     = flipped["bboxes"]
                    masks      = flipped["masks"]
                    # Заменяем category_ids: квадрант 1↔2, квадрант 3↔4
                    category_ids = flip_category_ids(list(flipped["category_ids"]))
                except Exception as e:
                    print(f"Flip augmentation failed: {e}")
        
        # Конвертируем обратно в BGR для Detectron2
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Обновляем аннотации после аугментаций
        new_annos = []
        for i, (bbox, cat_id) in enumerate(zip(bboxes, category_ids)):
            anno = {
                "bbox": list(bbox),
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": cat_id,
                "iscrowd": 0
            }
            
            # Добавляем маску в формате полигона
            if i < len(masks):
                mask = masks[i]
                if isinstance(mask, np.ndarray):
                    polygons = mask_to_polygon(mask)
                    if polygons:
                        anno["segmentation"] = polygons
                    else:
                        # Если не удалось конвертировать в полигон, пропускаем
                        continue
            
            new_annos.append(anno)
        
        dataset_dict["annotations"] = new_annos
    else:
        # Если нет аннотаций, просто применяем аугментации к изображению
        if augmentations is not None:
            transformed = augmentations(image=image_rgb, masks=[], bboxes=[], category_ids=[])
            image_rgb = transformed["image"]
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Конвертируем изображение в тензор
    dataset_dict["image"] = torch.as_tensor(
        image.transpose(2, 0, 1).astype("float32")
    )
    
    # Обрабатываем аннотации для Detectron2
    if "annotations" in dataset_dict:
        annos = [
            utils.transform_instance_annotations(obj, T.NoOpTransform(), image.shape[:2])
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        
        # Проверяем что аннотации не пустые
        if len(annos) > 0:
            instances = utils.annotations_to_instances(annos, image.shape[:2])
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        else:
            # Если аннотаций нет, создаем пустой Instances
            from detectron2.structures import Instances
            dataset_dict["instances"] = Instances(image.shape[:2])
    
    return dataset_dict


def build_train_loader_with_augmentations(cfg, mapper=None):
    """
    Создает DataLoader с аугментациями для обучения
    
    Args:
        cfg: конфигурация Detectron2
        mapper: кастомный mapper (если None, используется mapper с аугментациями)
    
    Returns:
        DataLoader
    """
    from detectron2.data import build_detection_train_loader
    from functools import partial
    
    if mapper is None:
        augmentations = get_train_transforms()
        flip_transform = get_flip_transform()
        print("+ Используется горизонтальный flip с зеркальной заменой FDI номеров (p=0.5)")
        mapper = partial(custom_mapper, augmentations=augmentations, flip_transform=flip_transform)
    
    return build_detection_train_loader(cfg, mapper=mapper)


def build_val_loader(cfg):
    """
    Создает DataLoader для валидации (без аугментаций)
    
    Args:
        cfg: конфигурация Detectron2
    
    Returns:
        DataLoader
    """
    from detectron2.data import build_detection_test_loader
    from functools import partial
    
    mapper = partial(custom_mapper, augmentations=None)
    return build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=mapper)



