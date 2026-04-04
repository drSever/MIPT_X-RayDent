###############################################
# Конвертер YOLO формата в COCO для Mask R-CNN
###############################################

import os
import json
import yaml
from pathlib import Path
from PIL import Image
import numpy as np


class YOLOtoCOCOConverter:
    def __init__(self, dataset_path, data_yaml_path):
        """
        Args:
            dataset_path: путь к корневой папке датасета
            data_yaml_path: путь к data.yaml с именами классов
        """
        self.dataset_path = Path(dataset_path)
        self.data_yaml_path = Path(data_yaml_path)
        self.class_names = self._load_class_names()
        
    def _load_class_names(self):
        """Загружает имена классов из data.yaml"""
        with open(self.data_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data['names']
    
    def convert_split(self, split='train'):
        """
        Конвертирует один split (train/valid/test) в COCO формат
        
        Args:
            split: 'train', 'valid' или 'test'
        
        Returns:
            dict: COCO формат аннотаций
        """
        images_dir = self.dataset_path / split / 'images'
        labels_dir = self.dataset_path / split / 'labels'
        
        coco_format = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Создаем категории (классы зубов по FDI)
        for class_id, class_name in enumerate(self.class_names):
            coco_format['categories'].append({
                'id': class_id + 1,  # COCO использует 1-based индексы
                'name': class_name,
                'supercategory': 'tooth'
            })
        
        annotation_id = 1
        image_id = 1
        
        # Обрабатываем все изображения
        image_files = sorted(images_dir.glob('*.jpg')) + sorted(images_dir.glob('*.png'))
        
        for img_path in image_files:
            # Загружаем изображение для получения размеров
            img = Image.open(img_path)
            width, height = img.size
            
            # Добавляем информацию об изображении
            coco_format['images'].append({
                'id': image_id,
                'file_name': img_path.name,
                'width': width,
                'height': height
            })
            
            # Ищем соответствующий файл разметки
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        
                        class_id = int(parts[0])
                        # Координаты полигона (нормализованные)
                        polygon = list(map(float, parts[1:]))
                        
                        # ВАЛИДАЦИЯ: проверка нормализованных координат
                        if any(c < 0 or c > 1.1 for c in polygon):  # 10% допуск
                            print(f"!!! Пропущена аннотация в {img_path.name}: координаты вне [0,1]")
                            continue
                        
                        # Конвертируем в абсолютные координаты с обрезкой
                        abs_polygon = []
                        for i in range(0, len(polygon), 2):
                            x = max(0.0, min(float(width), polygon[i] * width))
                            y = max(0.0, min(float(height), polygon[i + 1] * height))
                            abs_polygon.extend([x, y])
                        
                        # Вычисляем bbox из полигона
                        x_coords = abs_polygon[0::2]
                        y_coords = abs_polygon[1::2]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        bbox_width = x_max - x_min
                        bbox_height = y_max - y_min
                        
                        # ВАЛИДАЦИЯ bbox
                        if bbox_width <= 0 or bbox_height <= 0:
                            print(f"!!! Пропущена аннотация в {img_path.name}: нулевой bbox")
                            continue
                        
                        if bbox_width > width * 2 or bbox_height > height * 2:
                            print(f"!!! Пропущена аннотация в {img_path.name}: bbox слишком большой ({bbox_width}x{bbox_height})")
                            continue
                        
                        # Вычисляем площадь полигона
                        area = self._polygon_area(x_coords, y_coords)
                        
                        if area <= 0:
                            print(f"  ⚠ Пропущена аннотация в {img_path.name}: нулевая площадь")
                            continue
                        
                        # Добавляем аннотацию
                        coco_format['annotations'].append({
                            'id': annotation_id,
                            'image_id': image_id,
                            'category_id': class_id + 1,
                            'segmentation': [abs_polygon],  # список полигонов
                            'area': float(area),
                            'bbox': [float(x_min), float(y_min), float(bbox_width), float(bbox_height)],
                            'iscrowd': 0
                        })
                        
                        annotation_id += 1
            
            image_id += 1
        
        print(f"Конвертировано {split}: {len(coco_format['images'])} изображений, "
              f"{len(coco_format['annotations'])} аннотаций")
        
        return coco_format
    
    def _polygon_area(self, x_coords, y_coords):
        """Вычисляет площадь полигона по формуле Shoelace"""
        n = len(x_coords)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += x_coords[i] * y_coords[j]
            area -= x_coords[j] * y_coords[i]
        return abs(area) / 2.0
    
    def save_coco_json(self, coco_format, output_path):
        """Сохраняет COCO формат в JSON файл"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_format, f, indent=2)
        print(f"Сохранено: {output_path}")
    
    def convert_all(self, output_dir):
        """Конвертирует все splits и сохраняет в output_dir"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split in ['train', 'valid', 'test']:
            split_dir = self.dataset_path / split
            if split_dir.exists():
                coco_format = self.convert_split(split)
                output_path = output_dir / f"annotations_{split}.json"
                self.save_coco_json(coco_format, output_path)


if __name__ == "__main__":
    # Пример использования
    converter = YOLOtoCOCOConverter(
        dataset_path="dataset",
        data_yaml_path="dataset/data.yaml"
    )
    converter.convert_all(output_dir="dataset/coco_annotations")
