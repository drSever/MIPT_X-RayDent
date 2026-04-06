#######################################
# Инференс обученной Mask R-CNN модели
#######################################

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from detectron2.config import get_cfg, CfgNode as CN
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog


class MaskRCNNInference:
    """Класс для инференса Mask R-CNN"""
    
    # Маппинг class_id -> FDI номер зуба
    CLASS_TO_FDI = {
        0: 11, 1: 12, 2: 13, 3: 14, 4: 15, 5: 16, 6: 17, 7: 18,
        8: 21, 9: 22, 10: 23, 11: 24, 12: 25, 13: 26, 14: 27, 15: 28,
        16: 31, 17: 32, 18: 33, 19: 34, 20: 35, 21: 36, 22: 37, 23: 38,
        24: 41, 25: 42, 26: 43, 27: 44, 28: 45, 29: 46, 30: 47, 31: 48
    }
    
    def __init__(self, model_path, num_classes=32, score_threshold=0.5,
                 test_image_size=640,
                 swin_model_name="swin_base_patch4_window7_224"):
        """
        Args:
            model_path: путь к обученной модели (model_final.pth)
            num_classes: количество классов
            score_threshold: порог уверенности для предсказаний
            test_image_size: размер короткой стороны для inference (должен совпадать с обучением)
            swin_model_name: имя модели timm (должно совпадать с тем, что использовалось при обучении)
        """
        # Регистрируем Swin backbone
        import swin_backbone  # noqa: F401

        self.cfg = self._setup_config(model_path, num_classes, score_threshold,
                                      test_image_size, swin_model_name)
        self.predictor = DefaultPredictor(self.cfg)

        # Создаем metadata с FDI номерами
        self.metadata = MetadataCatalog.get("teeth_inference")
        self.metadata.thing_classes = [str(self.CLASS_TO_FDI[i]) for i in range(num_classes)]
    
    def _setup_config(self, model_path, num_classes, score_threshold,
                      test_image_size, swin_model_name):
        """Настройка конфигурации для инференса"""
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        ))

        # Swin-специфичные ключи
        cfg.MODEL.SWIN = CN()
        cfg.MODEL.SWIN.MODEL_NAME = swin_model_name
        cfg.MODEL.SWIN.PRETRAINED = False  # при inference не нужно — грузим чекпоинт

        cfg.MODEL.BACKBONE.NAME = "build_swin_fpn_backbone"
        cfg.MODEL.BACKBONE.FREEZE_AT = 0
        cfg.MODEL.FPN.OUT_CHANNELS = 256

        cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
        cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]

        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3

        cfg.INPUT.MIN_SIZE_TEST = test_image_size
        cfg.INPUT.MAX_SIZE_TEST = test_image_size

        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
        cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000

        return cfg
    
    def predict(self, image_path, score_threshold=None):
        """
        Предсказание на одном изображении

        Args:
            image_path: путь к изображению
            score_threshold: порог уверенности (если None, использует из конфига)

        Returns:
            outputs: результаты предсказания
            image: исходное изображение
        """
        image = cv2.imread(str(image_path))

        # Меняем порог напрямую в модели — это работает без пересоздания predictor
        if score_threshold is not None:
            self.predictor.model.roi_heads.box_predictor.test_score_thresh = score_threshold

        outputs = self.predictor(image)
        return outputs, image
    
    def visualize(self, image, outputs, show_boxes=True, show_masks=True, 
                  show_labels=True, alpha=0.5):
        """
        Визуализация результатов
        
        Args:
            image: исходное изображение (BGR)
            outputs: результаты предсказания
            show_boxes: показывать bbox
            show_masks: показывать маски
            show_labels: показывать метки классов
            alpha: прозрачность масок
        
        Returns:
            visualized_image: изображение с визуализацией (RGB)
        """
        # Конвертируем BGR в RGB для Visualizer
        image_rgb = image[:, :, ::-1].copy()
        
        v = Visualizer(
            image_rgb,
            metadata=self.metadata,
            scale=1.0,
            instance_mode=ColorMode.IMAGE
        )
        
        instances = outputs["instances"].to("cpu")
        
        # Создаем копию instances для модификации
        if not show_boxes and not show_masks:
            # Если ничего не показываем, возвращаем исходное
            return image_rgb
        
        # Если нужно показать только bbox или только маски
        if not show_masks and show_boxes:
            # Удаляем маски из instances
            instances_copy = instances[:]
            if instances_copy.has("pred_masks"):
                instances_copy.remove("pred_masks")
            out = v.draw_instance_predictions(instances_copy)
        elif not show_boxes and show_masks:
            # Удаляем bbox из instances
            instances_copy = instances[:]
            if instances_copy.has("pred_boxes"):
                instances_copy.remove("pred_boxes")
            out = v.draw_instance_predictions(instances_copy)
        else:
            # Показываем все
            out = v.draw_instance_predictions(instances)
        
        # Возвращаем RGB изображение
        return out.get_image()[:, :, ::-1]
    
    def predict_and_visualize(self, image_path, save_path=None, show=True, score_threshold=None):
        """
        Предсказание и визуализация: исходное, с bbox, с масками
        
        Args:
            image_path: путь к изображению
            save_path: путь для сохранения результата
            show: показать результат
            score_threshold: порог уверенности (если None, использует из конфига)
        """
        outputs, image = self.predict(image_path, score_threshold=score_threshold)
        
        # 1. Исходное изображение (RGB)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Изображение с bbox (RGB)
        vis_bbox = self.visualize(image, outputs, show_boxes=True, show_masks=False)
        
        # 3. Изображение с масками (RGB)
        vis_masks = self.visualize(image, outputs, show_boxes=True, show_masks=True)
        
        if show:
            # Вертикальное расположение (в столбик)
            fig, axes = plt.subplots(3, 1, figsize=(12, 18))
            
            # Исходное
            axes[0].imshow(original_image)
            axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # С bbox
            axes[1].imshow(vis_bbox)
            axes[1].set_title('Predictions: Bounding Boxes', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            # С масками
            axes[2].imshow(vis_masks)
            axes[2].set_title('Predictions: Masks + Boxes', fontsize=14, fontweight='bold')
            axes[2].axis('off')
            
            plt.suptitle(f'Results: {Path(image_path).name}', fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()
            plt.show()
        
        if save_path:
            # Сохраняем все три варианта
            save_path = Path(save_path)
            
            # Исходное
            cv2.imwrite(str(save_path.parent / f"{save_path.stem}_original{save_path.suffix}"), 
                       cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
            
            # С bbox
            cv2.imwrite(str(save_path.parent / f"{save_path.stem}_bbox{save_path.suffix}"), 
                       cv2.cvtColor(vis_bbox, cv2.COLOR_RGB2BGR))
            
            # С масками
            cv2.imwrite(str(save_path.parent / f"{save_path.stem}_masks{save_path.suffix}"), 
                       cv2.cvtColor(vis_masks, cv2.COLOR_RGB2BGR))
            
            print(f"Результаты сохранены:")
            print(f"  - {save_path.parent / f'{save_path.stem}_original{save_path.suffix}'}")
            print(f"  - {save_path.parent / f'{save_path.stem}_bbox{save_path.suffix}'}")
            print(f"  - {save_path.parent / f'{save_path.stem}_masks{save_path.suffix}'}")
        
        return outputs, (original_image, vis_bbox, vis_masks)
    
    def visualize_comparison(self, image_paths, save_path=None):
        """
        Визуализация нескольких изображений в сетке
        
        Args:
            image_paths: список путей к изображениям
            save_path: путь для сохранения результата
        """
        num_images = len(image_paths)
        fig, axes = plt.subplots(num_images, 3, figsize=(20, 7 * num_images))
        
        if num_images == 1:
            axes = axes.reshape(1, -1)
        
        for idx, image_path in enumerate(image_paths):
            outputs, image = self.predict(image_path)
            
            # Исходное
            original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axes[idx, 0].imshow(original_image)
            axes[idx, 0].set_title(f'{Path(image_path).name}\nOriginal', fontsize=12)
            axes[idx, 0].axis('off')
            
            # С bbox
            vis_bbox = self.visualize(image, outputs, show_boxes=True, show_masks=False)
            axes[idx, 1].imshow(vis_bbox)
            axes[idx, 1].set_title('Bounding Boxes', fontsize=12)
            axes[idx, 1].axis('off')
            
            # С масками
            vis_masks = self.visualize(image, outputs, show_boxes=True, show_masks=True)
            axes[idx, 2].imshow(vis_masks)
            axes[idx, 2].set_title('Masks + Boxes', fontsize=12)
            axes[idx, 2].axis('off')
        
        plt.suptitle('Predictions Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Сравнение сохранено: {save_path}")
        
        plt.show()
    
    def predict_batch(self, image_dir, output_dir=None, extensions=['.jpg', '.png', '.JPG', '.PNG'],
                      score_threshold=None):
        """
        Предсказание на папке с изображениями

        Args:
            image_dir: директория с изображениями
            output_dir: директория для сохранения результатов
            extensions: расширения файлов изображений
            score_threshold: порог уверенности (если None, использует из конфига)
        """
        image_dir = Path(image_dir)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = []
        for ext in extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))
        
        print(f"Найдено {len(image_files)} изображений")
        
        results = []
        for img_path in image_files:
            print(f"Обработка: {img_path.name}")
            outputs, image = self.predict(img_path, score_threshold=score_threshold)
            
            if output_dir:
                vis_image = self.visualize(image, outputs)
                save_path = output_dir / f"pred_{img_path.name}"
                cv2.imwrite(str(save_path), vis_image)
            
            results.append({
                'image_path': str(img_path),
                'outputs': outputs
            })
        
        print(f"Обработано {len(results)} изображений")
        return results
    
    def get_predictions_info(self, outputs):
        """
        Извлекает информацию о предсказаниях
        
        Args:
            outputs: результаты предсказания
        
        Returns:
            dict: информация о предсказаниях с FDI номерами
        """
        instances = outputs["instances"].to("cpu")
        
        # Конвертируем class_id в FDI номера
        class_ids = instances.pred_classes.numpy().tolist()
        fdi_numbers = [self.CLASS_TO_FDI[class_id] for class_id in class_ids]
        
        info = {
            'num_instances': len(instances),
            'fdi_numbers': fdi_numbers,  # FDI номера зубов
            'class_ids': class_ids,      # Оригинальные class_id (для отладки)
            'scores': instances.scores.numpy().tolist(),
            'boxes': instances.pred_boxes.tensor.numpy().tolist(),
        }
        
        return info


def create_submission(inference_results, output_path):
    """
    Создает файл submission с FDI номерами
    
    Args:
        inference_results: результаты инференса
        output_path: путь для сохранения submission
    """
    import json
    
    # Маппинг class_id -> FDI
    CLASS_TO_FDI = MaskRCNNInference.CLASS_TO_FDI
    
    submission = []
    for result in inference_results:
        instances = result['outputs']["instances"].to("cpu")
        
        for i in range(len(instances)):
            class_id = int(instances.pred_classes[i])
            fdi_number = CLASS_TO_FDI[class_id]
            
            submission.append({
                'image_path': result['image_path'],
                'fdi_number': fdi_number,  # FDI номер зуба
                'class_id': class_id,      # Оригинальный class_id (для отладки)
                'score': float(instances.scores[i]),
                'bbox': instances.pred_boxes.tensor[i].numpy().tolist(),
                'mask': instances.pred_masks[i].numpy().tolist()
            })
    
    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"Submission сохранен: {output_path}")
    print(f"Всего предсказаний: {len(submission)}")


if __name__ == "__main__":
    # Параметры
    MODEL_PATH = "./output_maskrcnn/model_final.pth"
    NUM_CLASSES = 32
    SCORE_THRESHOLD = 0.5
    TEST_IMAGE_SIZE = 640       # Должен совпадать с test_image_size при обучении!
    SWIN_MODEL_NAME = "swin_base_patch4_window7_224"  # Должен совпадать с обучением!

    # Создаем инференс объект
    inference = MaskRCNNInference(
        model_path=MODEL_PATH,
        num_classes=NUM_CLASSES,
        score_threshold=SCORE_THRESHOLD,
        test_image_size=TEST_IMAGE_SIZE,
        swin_model_name=SWIN_MODEL_NAME,
    )
    
    # Пример 1: Предсказание на одном изображении (3 варианта визуализации)
    image_path = "dataset/test/images/example.jpg"
    outputs, (original, bbox_vis, masks_vis) = inference.predict_and_visualize(
        image_path=image_path,
        save_path="./output_maskrcnn/prediction_example.jpg",
        show=True
    )
    
    # Информация о предсказаниях
    info = inference.get_predictions_info(outputs)
    print(f"\nНайдено зубов: {info['num_instances']}")
    print(f"FDI номера: {info['fdi_numbers']}")
    print(f"Уверенность: {[f'{s:.3f}' for s in info['scores']]}")
    
    # Пример 2: Сравнение нескольких изображений
    # image_paths = [
    #     "dataset/test/images/example1.jpg",
    #     "dataset/test/images/example2.jpg",
    #     "dataset/test/images/example3.jpg"
    # ]
    # inference.visualize_comparison(
    #     image_paths=image_paths,
    #     save_path="./output_maskrcnn/comparison.jpg"
    # )
    
    # Пример 3: Batch предсказание
    # results = inference.predict_batch(
    #     image_dir="dataset/test/images",
    #     output_dir="./output_maskrcnn/predictions"
    # )
    
    # Пример 4: Создание submission
    # create_submission(results, "./output_maskrcnn/submission.json")
