##################################################################
# Оценка обученной модели Mask R-CNN на тестовой подвыборке
# Метрики: mAP50, mAP75, mAP50-95, Dice, IoU, Precision, Recall
##################################################################

import os
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.structures import BoxMode
import pycocotools.mask as mask_util


class SegmentationMetrics:
    """Вычисление метрик сегментации"""
    
    def __init__(self, num_classes=32):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Сброс накопленных метрик"""
        self.tp = np.zeros(self.num_classes)  # True Positives
        self.fp = np.zeros(self.num_classes)  # False Positives
        self.fn = np.zeros(self.num_classes)  # False Negatives
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)
        self.dice_scores = defaultdict(list)
        self.iou_scores = defaultdict(list)
    
    def update(self, pred_masks, pred_classes, gt_masks, gt_classes, iou_threshold=0.5):
        """
        Обновление метрик на основе предсказаний и ground truth
        
        Args:
            pred_masks: список предсказанных масок (numpy arrays)
            pred_classes: список предсказанных классов (используется для группировки метрик)
            gt_masks: список ground truth масок
            gt_classes: список ground truth классов 
            iou_threshold: порог IoU для считания предсказания правильным
        """
        # Конвертируем в numpy если нужно
        if isinstance(pred_classes, torch.Tensor):
            pred_classes = pred_classes.cpu().numpy()
        if isinstance(gt_classes, torch.Tensor):
            gt_classes = gt_classes.cpu().numpy()
        
        # Создаем матрицу IoU между всеми предсказаниями и GT
        iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
        
        for i, pred_mask in enumerate(pred_masks):
            for j, gt_mask in enumerate(gt_masks):
                iou_matrix[i, j] = self._compute_iou(pred_mask, gt_mask)
        
        # Сопоставление предсказаний с GT (жадный алгоритм)
        matched_gt = set()
        matched_pred = set()
        
        # Сортируем по убыванию IoU
        matches = []
        for i in range(len(pred_masks)):
            for j in range(len(gt_masks)):
                if iou_matrix[i, j] > 0:
                    matches.append((iou_matrix[i, j], i, j))
        matches.sort(reverse=True)
        
        # Сопоставление
        for iou_val, pred_idx, gt_idx in matches:
            if pred_idx in matched_pred or gt_idx in matched_gt:
                continue
            
            gt_class = gt_classes[gt_idx]
            
            # Проверяем IoU 
            if iou_val >= iou_threshold:
                # True Positive
                self.tp[gt_class] += 1
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)
                
                # Вычисляем Dice и IoU для этой пары
                dice = self._compute_dice(pred_masks[pred_idx], gt_masks[gt_idx])
                self.dice_scores[gt_class].append(dice)
                self.iou_scores[gt_class].append(iou_val)
                
                # Накапливаем intersection и union
                intersection = np.logical_and(pred_masks[pred_idx], gt_masks[gt_idx]).sum()
                union = np.logical_or(pred_masks[pred_idx], gt_masks[gt_idx]).sum()
                self.intersection[gt_class] += intersection
                self.union[gt_class] += union
        
        # False Positives: предсказания без соответствующего GT
        # Распределяем по классам GT пропорционально
        unmatched_preds = len(pred_masks) - len(matched_pred)
        if unmatched_preds > 0:
            for gt_class in set(gt_classes):
                class_count = np.sum(gt_classes == gt_class)
                class_ratio = class_count / len(gt_classes)
                self.fp[gt_class] += unmatched_preds * class_ratio
        
        # False Negatives: GT без соответствующего предсказания
        for gt_idx in range(len(gt_masks)):
            if gt_idx not in matched_gt:
                gt_class = gt_classes[gt_idx]
                self.fn[gt_class] += 1
    
    def _compute_iou(self, mask1, mask2):
        """Вычисление IoU между двумя масками"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return intersection / union
    
    def _compute_dice(self, mask1, mask2):
        """Вычисление Dice коэффициента между двумя масками"""
        intersection = np.logical_and(mask1, mask2).sum()
        sum_masks = mask1.sum() + mask2.sum()
        if sum_masks == 0:
            return 0.0
        return 2.0 * intersection / sum_masks
    
    def get_metrics(self):
        """Получение всех метрик"""
        metrics = {}
        
        # Per-class метрики
        for class_id in range(self.num_classes):
            # Detection metrics 
            precision_det = self.tp[class_id] / (self.tp[class_id] + self.fp[class_id] + 1e-10)
            recall_det = self.tp[class_id] / (self.tp[class_id] + self.fn[class_id] + 1e-10)
            f1_det = 2 * precision_det * recall_det / (precision_det + recall_det + 1e-10)
            
            # Segmentation quality metrics 
            dice_list = self.dice_scores.get(class_id, [])
            iou_list = self.iou_scores.get(class_id, [])
            
            dice_mean = np.mean(dice_list) if dice_list else 0.0
            iou_mean = np.mean(iou_list) if iou_list else 0.0
            
            # IoU из накопленных intersection и union
            iou_accumulated = self.intersection[class_id] / (self.union[class_id] + 1e-10)
            
            metrics[class_id] = {
                'precision_det': precision_det,  # Detection precision
                'recall_det': recall_det,        # Detection recall
                'f1_det': f1_det,
                'dice': dice_mean,               # Segmentation quality (Dice)
                'iou': iou_mean,                 # Segmentation quality (IoU)
                'iou_accumulated': iou_accumulated,  # IoU from accumulated pixels
                'tp': int(self.tp[class_id]),
                'fp': int(self.fp[class_id]),
                'fn': int(self.fn[class_id]),
                'num_instances': int(self.tp[class_id] + self.fn[class_id])
            }
        
        # Macro-averaged метрики (среднее по классам)
        valid_classes = [cid for cid in range(self.num_classes) 
                        if metrics[cid]['num_instances'] > 0]
        
        if valid_classes:
            metrics['macro'] = {
                'precision_det': np.mean([metrics[cid]['precision_det'] for cid in valid_classes]),
                'recall_det': np.mean([metrics[cid]['recall_det'] for cid in valid_classes]),
                'f1_det': np.mean([metrics[cid]['f1_det'] for cid in valid_classes]),
                'dice': np.mean([metrics[cid]['dice'] for cid in valid_classes]),
                'iou': np.mean([metrics[cid]['iou'] for cid in valid_classes]),
            }
        else:
            metrics['macro'] = {
                'precision_det': 0.0, 'recall_det': 0.0, 'f1_det': 0.0,
                'dice': 0.0, 'iou': 0.0
            }
        
        # Micro-averaged метрики (взвешенное по количеству экземпляров)
        total_tp = self.tp.sum()
        total_fp = self.fp.sum()
        total_fn = self.fn.sum()
        
        # Detection-level micro metrics
        precision_det_micro = total_tp / (total_tp + total_fp + 1e-10)
        recall_det_micro = total_tp / (total_tp + total_fn + 1e-10)
        f1_det_micro = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-10)
        
        # Segmentation-level micro metrics (accumulated IoU)
        iou_micro = self.intersection.sum() / (self.union.sum() + 1e-10)
        
        metrics['micro'] = {
            'precision_det': precision_det_micro,
            'recall_det': recall_det_micro,
            'f1_det': f1_det_micro,
            'iou': iou_micro,
        }
        
        return metrics


def evaluate_model(
    model_path,
    dataset_path,
    coco_annotations_path,
    output_dir,
    num_classes=32,
    test_image_size=640,
    score_threshold=0.5,
    dataset_name="test"
):
    """
    Полная оценка модели на тестовой подвыборке
    
    Args:
        model_path: путь к весам модели (.pth)
        dataset_path: путь к датасету
        coco_annotations_path: путь к COCO аннотациям
        output_dir: директория для сохранения результатов
        num_classes: количество классов
        test_image_size: размер изображений для inference
        score_threshold: порог уверенности для предсказаний
        dataset_name: имя подвыборки ("test", "valid")
    
    Returns:
        dict: словарь с метриками
    """
    
    print("="*70)
    print("ОЦЕНКА МОДЕЛИ НА ТЕСТОВОЙ ПОДВЫБОРКЕ")
    print("="*70)
    
    # Создаем output директорию
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Регистрируем датасет
    dataset_key = f"teeth_{dataset_name}_eval"
    
    # Удаляем если уже зарегистрирован
    if dataset_key in DatasetCatalog:
        DatasetCatalog.remove(dataset_key)
        MetadataCatalog.remove(dataset_key)
    
    register_coco_instances(
        dataset_key,
        {},
        str(Path(coco_annotations_path) / f"annotations_{dataset_name}.json"),
        str(Path(dataset_path) / dataset_name / "images")
    )
    
    # Загружаем датасет чтобы получить маппинг категорий
    dataset_dicts_temp = DatasetCatalog.get(dataset_key)
    metadata = MetadataCatalog.get(dataset_key)
    
    # Создаем маппинг из COCO JSON напрямую
    import json
    coco_json_path = Path(coco_annotations_path) / f"annotations_{dataset_name}.json"
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Создаем маппинг: category_id (COCO) -> contiguous_id (модель использует 0-based)
    # Detectron2 сортирует категории по id и присваивает contiguous_id
    categories_sorted = sorted(coco_data['categories'], key=lambda x: x['id'])
    category_id_to_contiguous_id = {cat['id']: idx for idx, cat in enumerate(categories_sorted)}
    contiguous_id_to_category_id = {idx: cat['id'] for idx, cat in enumerate(categories_sorted)}
    
    print(f"\nМаппинг классов (все 32):")
    for cont_id in sorted(contiguous_id_to_category_id.keys()):
        cat_id = contiguous_id_to_category_id[cont_id]
        cat_name = next(c['name'] for c in categories_sorted if c['id'] == cat_id)
        print(f"  contiguous_id={cont_id:2d} -> category_id={cat_id:2d} -> class_id={cat_id-1:2d} (FDI {cat_name})")
    
    # Настройка конфигурации
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    ))
    
    cfg.MODEL.WEIGHTS = str(model_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    
    # здесь квадратное изображение (размер должен быть как на обучении)
    # если прямоугольное, то значения будут отличаться 
    cfg.INPUT.MIN_SIZE_TEST = test_image_size
    cfg.INPUT.MAX_SIZE_TEST = test_image_size
    
    # Anchor параметры - ДОЛЖНЫ СОВПАДАТЬ с параметрами при обучении!
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    
    cfg.DATASETS.TEST = (dataset_key,)
    
    print(f"\nМодель: {model_path}")
    print(f"Датасет: {dataset_name}")
    print(f"Score threshold: {score_threshold}")
    print(f"Image size: {test_image_size}")
    
    # ========== 1. COCO МЕТРИКИ (mAP) ==========
    print(f"\n{'='*70}")
    print("1. ВЫЧИСЛЕНИЕ COCO МЕТРИК (mAP)")
    print(f"{'='*70}")
    
    evaluator = COCOEvaluator(
        dataset_key,
        output_dir=str(output_dir / "coco_eval")
    )
    
    val_loader = build_detection_test_loader(cfg, dataset_key)
    
    # Создаем модель
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    
    model = build_model(cfg)
    model.eval()
    
    # Загружаем веса
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    
    coco_results = inference_on_dataset(model, val_loader, evaluator)
    
    # Извлекаем метрики
    bbox_metrics = coco_results.get('bbox', {})
    segm_metrics = coco_results.get('segm', {})
    
    print(f"\nBBox метрики:")
    print(f"  AP (mAP50-95): {bbox_metrics.get('AP', 0):.4f}")
    print(f"  AP50: {bbox_metrics.get('AP50', 0):.4f}")
    print(f"  AP75: {bbox_metrics.get('AP75', 0):.4f}")
    
    print(f"\nSegmentation метрики:")
    print(f"  AP (mAP50-95): {segm_metrics.get('AP', 0):.4f}")
    print(f"  AP50: {segm_metrics.get('AP50', 0):.4f}")
    print(f"  AP75: {segm_metrics.get('AP75', 0):.4f}")
    
    # ========== 2. DICE, IoU, PRECISION, RECALL ==========
    print(f"\n{'='*70}")
    print("2. ВЫЧИСЛЕНИЕ DICE, IoU, PRECISION, RECALL")
    print(f"{'='*70}")
    
    # Создаем predictor
    predictor = DefaultPredictor(cfg)
    
    # Загружаем датасет
    dataset_dicts = DatasetCatalog.get(dataset_key)
    
    # Инициализируем метрики для разных порогов IoU
    metrics_iou50 = SegmentationMetrics(num_classes)
    metrics_iou75 = SegmentationMetrics(num_classes)
    
    print(f"\nОбработка {len(dataset_dicts)} изображений...")
    
    # Для отладки
    total_predictions = 0
    total_gt = 0
    pred_class_counts = defaultdict(int)
    gt_class_counts = defaultdict(int)
    
    for data in tqdm(dataset_dicts):
        # Загружаем изображение
        import cv2
        img = cv2.imread(data["file_name"])
        height, width = img.shape[:2]
        
        # Предсказание
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        
        # Извлекаем предсказанные маски и классы (уже отфильтрованы по score_threshold)
        if len(instances) > 0:
            pred_masks = instances.pred_masks.numpy()
            pred_classes_contiguous = instances.pred_classes.numpy()  # contiguous_id от модели
            pred_scores = instances.scores.numpy()
            
            # Дополнительная проверка score threshold (на всякий случай)
            valid_idx = pred_scores >= score_threshold
            pred_masks = pred_masks[valid_idx]
            pred_classes_contiguous = pred_classes_contiguous[valid_idx]
            
            # Конвертируем contiguous_id в category_id, затем в 0-based class_id
            pred_classes = []
            
            for idx, cont_id in enumerate(pred_classes_contiguous):
                cat_id = contiguous_id_to_category_id.get(int(cont_id), None)
                if cat_id is not None:
                    # category_id (1-32) -> class_id (0-31)
                    class_id = cat_id - 1
                    pred_classes.append(class_id)
                else:
                    # Если маппинг не найден, используем contiguous_id напрямую
                    print(f"WARNING: contiguous_id {cont_id} not found in mapping!")
                    pred_classes.append(int(cont_id))
            
            pred_classes = np.array(pred_classes)
            total_predictions += len(pred_masks)
            
            # Подсчет классов в предсказаниях
            for cls in pred_classes:
                pred_class_counts[int(cls)] += 1
        else:
            pred_masks = []
            pred_classes = np.array([])
        
        # Извлекаем ground truth маски и классы
        if "annotations" in data:
            gt_masks = []
            gt_classes = []
            gt_category_ids = []  # Для debug
            
            for anno in data["annotations"]:
                # Проверяем валидность category_id
                cat_id = anno.get("category_id", 0)
                if cat_id < 1 or cat_id > num_classes:
                    # Пропускаем невалидные аннотации
                    continue
                
                # Конвертируем полигон в маску
                if "segmentation" in anno:
                    segmentation = anno["segmentation"]
                    rles = mask_util.frPyObjects(segmentation, height, width)
                    rle = mask_util.merge(rles)
                    mask = mask_util.decode(rle).astype(bool)
                    
                    gt_masks.append(mask)
                    # category_id в COCO 1-based (1-32), конвертируем в 0-based (0-31)
                    gt_class = cat_id - 1
                    gt_classes.append(gt_class)
                    gt_category_ids.append(cat_id)
                    gt_class_counts[gt_class] += 1
            
            total_gt += len(gt_masks)
            
            # Обновляем метрики
            if len(gt_masks) > 0:
                metrics_iou50.update(pred_masks, pred_classes, gt_masks, gt_classes, iou_threshold=0.5)
                metrics_iou75.update(pred_masks, pred_classes, gt_masks, gt_classes, iou_threshold=0.75)
    
    print(f"\nСтатистика предсказаний:")
    print(f"  Всего GT объектов: {total_gt}")
    print(f"  Всего предсказаний: {total_predictions}")
    print(f"  Среднее GT на изображение: {total_gt / len(dataset_dicts):.1f}")
    print(f"  Среднее предсказаний на изображение: {total_predictions / len(dataset_dicts):.1f}")
    
    # Проверяем распределение классов
    print(f"\nПроверка распределения классов (первые 5):")
    print(f"  GT classes: {dict(sorted(gt_class_counts.items())[:5])}")
    print(f"  Pred classes: {dict(sorted(pred_class_counts.items())[:5])}")
    
    # Проверяем, есть ли предсказания для всех классов
    gt_classes_set = set(gt_class_counts.keys())
    pred_classes_set = set(pred_class_counts.keys())
    print(f"\n  Уникальных GT классов: {len(gt_classes_set)}")
    print(f"  Уникальных Pred классов: {len(pred_classes_set)}")
    print(f"  Классы только в GT: {sorted(gt_classes_set - pred_classes_set)}")
    print(f"  Классы только в Pred: {sorted(pred_classes_set - gt_classes_set)}")
    
    # Получаем метрики
    results_iou50 = metrics_iou50.get_metrics()
    results_iou75 = metrics_iou75.get_metrics()
    
    # ========== 3. ВЫВОД РЕЗУЛЬТАТОВ ==========
    print(f"\n{'='*70}")
    print("3. СВОДКА МЕТРИК")
    print(f"{'='*70}")
    
    # FDI номера зубов
    CLASS_TO_FDI = {
        0: 11, 1: 12, 2: 13, 3: 14, 4: 15, 5: 16, 6: 17, 7: 18,
        8: 21, 9: 22, 10: 23, 11: 24, 12: 25, 13: 26, 14: 27, 15: 28,
        16: 31, 17: 32, 18: 33, 19: 34, 20: 35, 21: 36, 22: 37, 23: 38,
        24: 41, 25: 42, 26: 43, 27: 44, 28: 45, 29: 46, 30: 47, 31: 48
    }
    
    print(f"\n{'='*70}")
    print("ОБЩИЕ МЕТРИКИ")
    print(f"{'='*70}")
    
    print(f"\nCOCO Segmentation:")
    print(f"  mAP50-95: {segm_metrics.get('AP', 0):.4f}")
    print(f"  mAP50:    {segm_metrics.get('AP50', 0):.4f}")
    print(f"  mAP75:    {segm_metrics.get('AP75', 0):.4f}")
    
    print(f"\nMacro-averaged (IoU@0.5):")
    print(f"  Segmentation Quality:")
    print(f"    Dice:      {results_iou50['macro']['dice']:.4f}")
    print(f"    IoU:       {results_iou50['macro']['iou']:.4f}")
    print(f"  Detection Performance:")
    print(f"    Precision: {results_iou50['macro']['precision_det']:.4f}")
    print(f"    Recall:    {results_iou50['macro']['recall_det']:.4f}")
    print(f"    F1:        {results_iou50['macro']['f1_det']:.4f}")
    
    print(f"\nMacro-averaged (IoU@0.75):")
    print(f"  Segmentation Quality:")
    print(f"    Dice:      {results_iou75['macro']['dice']:.4f}")
    print(f"    IoU:       {results_iou75['macro']['iou']:.4f}")
    print(f"  Detection Performance:")
    print(f"    Precision: {results_iou75['macro']['precision_det']:.4f}")
    print(f"    Recall:    {results_iou75['macro']['recall_det']:.4f}")
    print(f"    F1:        {results_iou75['macro']['f1_det']:.4f}")
    
    print(f"\nMicro-averaged (IoU@0.5):")
    print(f"  Segmentation IoU: {results_iou50['micro']['iou']:.4f}")
    print(f"  Detection Precision: {results_iou50['micro']['precision_det']:.4f}")
    print(f"  Detection Recall:    {results_iou50['micro']['recall_det']:.4f}")
    print(f"  Detection F1:        {results_iou50['micro']['f1_det']:.4f}")
    
    # Per-class метрики
    print(f"\n{'='*70}")
    print("МЕТРИКИ ПО КЛАССАМ (IoU@0.5)")
    print(f"{'='*70}")
    print(f"\n{'Class':<8} {'FDI':<5} {'Dice':<7} {'IoU':<7} {'TP':<6} {'FP':<6} {'FN':<6} {'P_det':<7} {'R_det':<7} {'F1_det':<7}")
    print("-" * 90)
    
    for class_id in range(num_classes):
        if results_iou50[class_id]['num_instances'] > 0:
            fdi = CLASS_TO_FDI[class_id]
            m = results_iou50[class_id]
            print(f"{class_id:<8} {fdi:<5} {m['dice']:<7.3f} {m['iou']:<7.3f} "
                  f"{m['tp']:<6} {m['fp']:<6} {m['fn']:<6} "
                  f"{m['precision_det']:<7.3f} {m['recall_det']:<7.3f} "
                  f"{m['f1_det']:<7.3f}")
    
    # ========== 4. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ==========
    print(f"\n{'='*70}")
    print("4. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print(f"{'='*70}")
    
    # Собираем все результаты
    all_results = {
        'model_path': str(model_path),
        'dataset': dataset_name,
        'score_threshold': score_threshold,
        'test_image_size': test_image_size,
        'coco_metrics': {
            'bbox': bbox_metrics,
            'segmentation': segm_metrics
        },
        'segmentation_metrics_iou50': {
            'macro': results_iou50['macro'],
            'micro': results_iou50['micro'],
            'per_class': {
                CLASS_TO_FDI[cid]: results_iou50[cid] 
                for cid in range(num_classes)
                if results_iou50[cid]['num_instances'] > 0
            }
        },
        'segmentation_metrics_iou75': {
            'macro': results_iou75['macro'],
            'micro': results_iou75['micro'],
            'per_class': {
                CLASS_TO_FDI[cid]: results_iou75[cid]
                for cid in range(num_classes)
                if results_iou75[cid]['num_instances'] > 0
            }
        }
    }
    
    # Сохраняем в JSON
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Результаты сохранены: {results_path}")
    
    # Сохраняем в CSV для удобства
    csv_path = output_dir / "per_class_metrics.csv"
    with open(csv_path, 'w') as f:
        f.write("Class,FDI,Dice_IoU50,IoU_IoU50,Precision_det_IoU50,Recall_det_IoU50,F1_det_IoU50,"
                "Dice_IoU75,IoU_IoU75,Precision_det_IoU75,Recall_det_IoU75,F1_det_IoU75,Num_Instances\n")
        
        for class_id in range(num_classes):
            if results_iou50[class_id]['num_instances'] > 0:
                fdi = CLASS_TO_FDI[class_id]
                m50 = results_iou50[class_id]
                m75 = results_iou75[class_id]
                
                f.write(f"{class_id},{fdi},"
                       f"{m50['dice']:.4f},{m50['iou']:.4f},"
                       f"{m50['precision_det']:.4f},{m50['recall_det']:.4f},{m50['f1_det']:.4f},"
                       f"{m75['dice']:.4f},{m75['iou']:.4f},"
                       f"{m75['precision_det']:.4f},{m75['recall_det']:.4f},{m75['f1_det']:.4f},"
                       f"{m50['num_instances']}\n")
    
    print(f"+ CSV сохранен: {csv_path}")
    
    print(f"\n{'='*70}")
    print("ОЦЕНКА ЗАВЕРШЕНА")
    print(f"{'='*70}")
    
    return all_results


if __name__ == "__main__":
    # Пример использования
    results = evaluate_model(
        model_path="maskrcnn_output/model_best.pth",
        dataset_path="dataset",
        coco_annotations_path="dataset/coco_annotations",
        output_dir="evaluation_results",
        num_classes=32,
        test_image_size=640,
        score_threshold=0.5,
        dataset_name="test"
    )
