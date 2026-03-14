######################################################################
# Обучение Mask R-CNN X-101-32x8d-FPN для сегментации зубов
######################################################################

import os
import json

# Отключаем torch.compile ПЕРЕД импортом torch,
# что решает проблему "AssertionError: 'XBLOCK' too large" в Google Colab
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch

# Дополнительно страхуемся: отключаем Dynamo и Inductor
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

import torch.nn.functional as F
import numpy as np
from pathlib import Path
import shutil
from collections import Counter

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import get_event_storage
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers


class BestModelSaver(HookBase):
    """Hook для сохранения лучшей модели по Valid AP"""
    
    def __init__(self, checkpointer, output_dir):
        self.checkpointer = checkpointer
        self.output_dir = output_dir
        self.best_ap = 0.0
        self.best_iter = 0
    
    def update_best_model(self, iteration, current_ap):
        """Обновляет лучшую модель если текущий AP выше"""
        if current_ap > self.best_ap:
            previous_best = self.best_ap
            self.best_ap = current_ap
            self.best_iter = iteration
            
            # Сохраняем лучшую модель
            best_model_path = os.path.join(self.output_dir, "model_best.pth")
            self.checkpointer.save("model_best")
            
            print(f"\n{'='*70}")
            print(f"*** НОВАЯ ЛУЧШАЯ МОДЕЛЬ! ***")
            print(f"{'='*70}")
            print(f"  Итерация: {iteration}")
            print(f"  Valid AP: {current_ap/100:.4f} (предыдущий лучший: {previous_best/100:.4f})")
            print(f"  Сохранено: {best_model_path}")
            print(f"{'='*70}\n")
            
            return True
        return False


class LossLogger(HookBase):
    """Hook для логирования losses и learning rate"""
    
    def __init__(self, history, log_period=20):
        self.history = history
        self.log_period = log_period
    
    def after_step(self):
        """Вызывается после каждой итерации"""
        iteration = self.trainer.iter + 1
        
        # Логируем каждые log_period итераций
        if iteration % self.log_period == 0:
            storage = get_event_storage()
            
            # Получаем losses из storage
            losses = {}
            loss_dict = storage.latest()
            
            # Собираем все losses
            if 'total_loss' in loss_dict:
                losses['total_loss'] = loss_dict['total_loss'][0]
            if 'loss_cls' in loss_dict:
                losses['loss_cls'] = loss_dict['loss_cls'][0]
            if 'loss_box_reg' in loss_dict:
                losses['loss_box_reg'] = loss_dict['loss_box_reg'][0]
            if 'loss_mask' in loss_dict:
                losses['loss_mask'] = loss_dict['loss_mask'][0]
            if 'loss_rpn_cls' in loss_dict:
                losses['loss_rpn_cls'] = loss_dict['loss_rpn_cls'][0]
            if 'loss_rpn_loc' in loss_dict:
                losses['loss_rpn_loc'] = loss_dict['loss_rpn_loc'][0]
            
            # Получаем learning rate
            lr = self.trainer.optimizer.param_groups[0]['lr']
            
            # Сохраняем в историю
            self.history.add_iteration(iteration, losses, lr)
            
            # Сохраняем каждые 100 итераций
            if iteration % 100 == 0:
                self.history.save()


class PeriodicEvaluator(HookBase):
    """Hook для периодической оценки на validation"""
    
    def __init__(self, eval_period, model, val_loader, val_dataset_name, output_dir, history, best_model_saver):
        self.eval_period = eval_period
        self.model = model
        self.val_loader = val_loader
        self.val_dataset_name = val_dataset_name
        self.output_dir = output_dir
        self.history = history
        self.best_model_saver = best_model_saver
    
    def after_step(self):
        next_iter = self.trainer.iter + 1
        
        if next_iter % self.eval_period == 0:
            print(f"\n{'='*70}")
            print(f"ОЦЕНКА НА VALIDATION (итерация {next_iter})")
            print(f"{'='*70}\n")
            
            # Оценка
            evaluator = COCOEvaluator(self.val_dataset_name, output_dir=self.output_dir)
            results = inference_on_dataset(self.model, self.val_loader, evaluator)
            
            # Сохраняем метрики (с проверкой наличия ключей)
            metrics = {
                'iteration': next_iter,
                'bbox_AP': results.get('bbox', {}).get('AP', 0.0),
                'bbox_AP50': results.get('bbox', {}).get('AP50', 0.0),
                'bbox_AP75': results.get('bbox', {}).get('AP75', 0.0),
                'segm_AP': results.get('segm', {}).get('AP', 0.0),
                'segm_AP50': results.get('segm', {}).get('AP50', 0.0),
                'segm_AP75': results.get('segm', {}).get('AP75', 0.0),
            }
            
            self.history.add_validation_metrics(next_iter, metrics)
            self.history.save()
            
            # Выводим результаты
            print(f"\nРезультаты:")
            if 'segm' in results:
                print(f"  Segmentation AP (mAP50-95): {metrics['segm_AP']/100:.4f}")
                print(f"  Segmentation AP50: {metrics['segm_AP50']/100:.4f}")
                print(f"  Segmentation AP75: {metrics['segm_AP75']/100:.4f}")
            else:
                print(f"  !!!  Segmentation метрики недоступны (возможно, слишком рано)")
            
            if 'bbox' in results:
                print(f"  BBox AP: {metrics['bbox_AP']/100:.4f}")
            
            # Проверяем и сохраняем лучшую модель (только если есть segm метрики)
            if 'segm' in results and metrics['segm_AP'] > 0:
                self.best_model_saver.update_best_model(next_iter, metrics['segm_AP'])
            else:
                print(f"  !!!  Пропуск сохранения лучшей модели (недостаточно метрик)")
            
            print(f"\n{'='*70}\n")


class TrainingHistory:
    """Класс для сохранения истории обучения"""
    
    def __init__(self, save_path):
        self.save_path = Path(save_path)
        self.history = {
            'iterations': [],
            'total_loss': [],
            'loss_cls': [],
            'loss_box_reg': [],
            'loss_mask': [],
            'loss_rpn_cls': [],
            'loss_rpn_loc': [],
            'learning_rate': [],
            'validation_metrics': []
        }
    
    def add_iteration(self, iteration, losses, lr):
        """Добавляет данные об итерации"""
        self.history['iterations'].append(iteration)
        self.history['total_loss'].append(losses.get('total_loss', 0))
        self.history['loss_cls'].append(losses.get('loss_cls', 0))
        self.history['loss_box_reg'].append(losses.get('loss_box_reg', 0))
        self.history['loss_mask'].append(losses.get('loss_mask', 0))
        self.history['loss_rpn_cls'].append(losses.get('loss_rpn_cls', 0))
        self.history['loss_rpn_loc'].append(losses.get('loss_rpn_loc', 0))
        self.history['learning_rate'].append(lr)
    
    def add_validation_metrics(self, iteration, metrics):
        """Добавляет метрики валидации"""
        metrics['iteration'] = iteration
        self.history['validation_metrics'].append(metrics)
    
    def save(self):
        """Сохраняет историю в JSON"""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self):
        """Загружает историю из JSON"""
        if self.save_path.exists():
            with open(self.save_path, 'r') as f:
                self.history = json.load(f)
            return True
        return False


class MaskRCNNTrainer(DefaultTrainer):
    """Кастомный trainer с периодической оценкой и class weights"""
    
    def __init__(self, cfg, val_loader, val_dataset_name, history, class_weights=None, use_augmentations=True):
        self.val_loader = val_loader
        self.val_dataset_name = val_dataset_name
        self.history = history
        self.class_weights = class_weights
        self.use_augmentations = use_augmentations
        super().__init__(cfg)
        
        # Применяем class weights к модели после инициализации
        if self.class_weights is not None:
            self._apply_class_weights()
    
    @classmethod
    def build_train_loader(cls, cfg):
        """Создает DataLoader с аугментациями или без них"""
        # Проверяем флаг use_augmentations из конфига
        use_augmentations = getattr(cfg, 'USE_AUGMENTATIONS', True)
        
        if use_augmentations:
            try:
                from augmentations import build_train_loader_with_augmentations
                print("+ Используются аугментации из augmentations.py")
                return build_train_loader_with_augmentations(cfg)
            except ImportError:
                print("!!!  augmentations.py не найден, используется стандартный loader")
                return super(MaskRCNNTrainer, cls).build_train_loader(cfg)
        else:
            print("+ Аугментации отключены, используется стандартный loader")
            return super(MaskRCNNTrainer, cls).build_train_loader(cfg)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)
    
    def _apply_class_weights(self):
        """Применяет class weights к loss функции классификации"""
        if self.class_weights is None:
            return
        
        # Переносим веса на нужное устройство
        device = next(self.model.parameters()).device
        self.class_weights = self.class_weights.to(device)
        
        # Находим FastRCNNOutputLayers в модели
        roi_heads = self.model.roi_heads
        
        # Сохраняем оригинальный метод losses
        original_losses = roi_heads.box_predictor.losses
        
        # Создаем wrapper для добавления class weights
        def weighted_losses(predictions, proposals):
            """Wrapper для добавления class weights в classification loss"""
            # Получаем оригинальные losses (БЕЗ classification loss)
            losses = original_losses(predictions, proposals)
            
            # Модифицируем classification loss с учетом весов
            if 'loss_cls' in losses:
                # Получаем предсказания и ground truth
                scores, _ = predictions
                
                # Получаем ground truth labels из proposals
                gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
                
                # Создаем расширенные веса (добавляем вес 1.0 для background класса)
                weights_with_bg = torch.cat([
                    torch.tensor([1.0], device=self.class_weights.device),
                    self.class_weights
                ])
                
                # Вычисляем weighted cross entropy loss
                # F.cross_entropy с weight уже применяет веса и усредняет правильно
                # reduction='mean' (по умолчанию) автоматически нормализует по сумме весов
                loss_cls = F.cross_entropy(
                    scores, 
                    gt_classes, 
                    weight=weights_with_bg,
                    reduction='mean'  # усреднение с учетом весов
                )
                
                # Перезаписываем classification loss взвешенной версией
                losses['loss_cls'] = loss_cls
            
            return losses
        
        # Заменяем метод losses на weighted версию
        roi_heads.box_predictor.losses = weighted_losses
        
        print(f"+ Class weights применены к модели")
    
    def build_hooks(self):
        """Добавляем кастомные hooks"""
        hooks = super().build_hooks()
        
        # Hook для логирования losses и LR
        hooks.insert(-1, LossLogger(
            history=self.history,
            log_period=20  # Логируем каждые 20 итераций
        ))
        
        # Hook для сохранения лучшей модели (создаем первым)
        best_model_saver = BestModelSaver(
            checkpointer=self.checkpointer,
            output_dir=self.cfg.OUTPUT_DIR
        )
        
        # Hook для периодической оценки (передаем best_model_saver)
        hooks.insert(-1, PeriodicEvaluator(
            eval_period=self.cfg.TEST.EVAL_PERIOD,
            model=self.model,
            val_loader=self.val_loader,
            val_dataset_name=self.val_dataset_name,
            output_dir=self.cfg.OUTPUT_DIR,
            history=self.history,
            best_model_saver=best_model_saver
        ))
        
        return hooks


def setup_config(num_classes, output_dir, resume_from=None, eval_period=1000,
                 max_iter=10000, batch_size=2, base_lr=0.00025, 
                 warmup_iters=500, lr_steps=None, checkpoint_period=1000,
                 use_augmentations=True,
                 train_image_sizes=(640, 720, 800),
                 test_image_size=800):
    """
    Настройка конфигурации Mask R-CNN X-101
    
    Args:
        num_classes: количество классов
        output_dir: директория для сохранения
        resume_from: путь к чекпоинту для продолжения обучения
        eval_period: период оценки на валидации (итераций)
        max_iter: максимальное количество итераций
        batch_size: размер батча 
        base_lr: начальный learning rate
        warmup_iters: количество итераций для warmup
        lr_steps: кортеж итераций для снижения LR (если None, вычисляется автоматически)
        checkpoint_period: период сохранения чекпоинтов
        use_augmentations: использовать ли аугментации
        train_image_sizes: размеры для обучения (tuple или int). 
                          Если tuple - multi-scale training 
        test_image_size: размер для валидации и inference (int).
                        Используется для:
                        1) Оценки на val/test подвыборке во время обучения
                        2) Inference на новых изображениях после обучения
                        Рекомендуется: средний размер из train_image_sizes
                        в случае multi-scale training
    """
    cfg = get_cfg()
    
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    ))
    
    if resume_from is None:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
        )
    else:
        cfg.MODEL.WEIGHTS = resume_from
    
    cfg.DATASETS.TRAIN = ("teeth_train",)
    cfg.DATASETS.TEST = ("teeth_val",)
    
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.WARMUP_ITERS = warmup_iters
    cfg.SOLVER.MAX_ITER = max_iter
    
    # Автоматический расчет lr_steps если не указаны
    if lr_steps is None:
        # По умолчанию: снижаем LR на 60% и 80% от max_iter
        lr_steps = (int(max_iter * 0.6), int(max_iter * 0.8))
    cfg.SOLVER.STEPS = lr_steps
    
    # Добавляем флаг use_augmentations в конфиг
    cfg.USE_AUGMENTATIONS = use_augmentations
    
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    
    # Настройка размеров изображений
    if isinstance(train_image_sizes, int):
        train_image_sizes = (train_image_sizes,)
    
    cfg.INPUT.MIN_SIZE_TRAIN = train_image_sizes
    cfg.INPUT.MAX_SIZE_TRAIN = max(train_image_sizes)  # Должно быть одно число, если передан кортеж
    cfg.INPUT.MIN_SIZE_TEST = test_image_size
    cfg.INPUT.MAX_SIZE_TEST = test_image_size  # В нашем случае изображение квадратное
    
    # Anchor параметры - конфигурация для FPN
    # Каждый уровень FPN отвечает за свой диапазон размеров объектов
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    
    # Оптимизация памяти для предотвращения OOM
    # Уменьшаем количество proposals для экономии памяти
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000  # По умолчанию 12000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000   # По умолчанию 6000
    
    # Периодическая оценка
    cfg.TEST.EVAL_PERIOD = eval_period
    
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg


def register_datasets(dataset_path, coco_annotations_path):
    """Регистрирует датасеты в Detectron2"""
    register_coco_instances(
        "teeth_train",
        {},
        str(Path(coco_annotations_path) / "annotations_train.json"),
        str(Path(dataset_path) / "train" / "images")
    )
    
    register_coco_instances(
        "teeth_val",
        {},
        str(Path(coco_annotations_path) / "annotations_valid.json"),
        str(Path(dataset_path) / "valid" / "images")
    )
    
    test_annotations = Path(coco_annotations_path) / "annotations_test.json"
    if test_annotations.exists():
        register_coco_instances(
            "teeth_test",
            {},
            str(test_annotations),
            str(Path(dataset_path) / "test" / "images")
        )
    
    print("Датасеты зарегистрированы")


def compute_class_weights(coco_annotations_path, num_classes=32, method='inverse_freq', power=1.0):
    """
    Вычисляет веса классов на основе частоты встречаемости в датасете
    
    Args:
        coco_annotations_path: путь к COCO аннотациям
        num_classes: количество классов
        method: метод вычисления весов
            - 'inverse_freq': обратная частота (1 / freq)
            - 'sqrt_inverse_freq': корень из обратной частоты (1 / sqrt(freq))
            - 'effective_samples': Effective Number of Samples (ENS)
        power: степень для усиления весов (по умолчанию 1.0)
    
    Returns:
        torch.Tensor: веса классов размера [num_classes]
    """
    # Загружаем аннотации
    train_json = Path(coco_annotations_path) / "annotations_train.json"
    with open(train_json, 'r') as f:
        data = json.load(f)
    
    # Подсчитываем количество экземпляров каждого класса
    class_counts = Counter()
    for ann in data['annotations']:
        # category_id в COCO 1-based, конвертируем в 0-based
        class_id = ann['category_id'] - 1
        class_counts[class_id] += 1
    
    # Создаем массив с количеством экземпляров для каждого класса
    counts = np.zeros(num_classes)
    for class_id, count in class_counts.items():
        if 0 <= class_id < num_classes:
            counts[class_id] = count
    
    # Проверяем наличие отсутствующих классов
    missing_classes = np.where(counts == 0)[0]
    if len(missing_classes) > 0:
        print(f"\n!!!  ПРЕДУПРЕЖДЕНИЕ: Обнаружены классы без экземпляров в train:")
        for class_id in missing_classes:
            fdi_number = class_id + 11 if class_id < 8 else class_id + 13 if class_id < 16 else class_id + 15 if class_id < 24 else class_id + 17
            print(f"    Класс {class_id} (FDI {fdi_number})")
    
    # Вычисляем веса в зависимости от метода
    if method == 'inverse_freq':
        # Обратная частота: weight = 1 / count
        weights = 1.0 / (counts + 1e-6)  # +epsilon для избежания деления на 0
        
    elif method == 'sqrt_inverse_freq':
        # Корень из обратной частоты: weight = 1 / sqrt(count)
        weights = 1.0 / (np.sqrt(counts) + 1e-6)
        
    elif method == 'effective_samples':
        # Effective Number of Samples (ENS)
        # weight = (1 - beta) / (1 - beta^n)
        # где beta = (N - 1) / N, N - общее количество образцов
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-6)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Применяем степень для усиления весов
    if power != 1.0:
        weights = np.power(weights, power)
    
    # Обрабатываем отсутствующие классы: заменяем их вес на максимальный среди присутствующих
    if len(missing_classes) > 0:
        present_classes_mask = counts > 0
        if np.any(present_classes_mask):
            max_weight_present = weights[present_classes_mask].max()
            weights[missing_classes] = max_weight_present
            print(f"    * Веса отсутствующих классов установлены в {max_weight_present:.3f} (max среди присутствующих)")
        else:
            # Если все классы отсутствуют (крайне маловероятно), устанавливаем единичные веса
            weights[:] = 1.0
            print(f"    !!! Все классы отсутствуют! Установлены единичные веса.")
    
    # Нормализуем веса так, чтобы среднее было 1.0
    weights = weights / weights.mean()
    
    # Конвертируем в torch.Tensor
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    
    # Выводим статистику
    print(f"\n{'='*70}")
    print("CLASS WEIGHTS СТАТИСТИКА")
    print(f"{'='*70}")
    print(f"Метод: {method}")
    if power != 1.0:
        print(f"Степень усиления: {power}")
    print(f"\nРаспределение классов:")
    
    # Группируем зубы по типам
    wisdom_teeth = [17, 27, 37, 47]  # Зубы мудрости (18, 28, 38, 48 в FDI)
    
    for i in range(num_classes):
        fdi_number = i + 11 if i < 8 else i + 13 if i < 16 else i + 15 if i < 24 else i + 17
        tooth_type = " Зуб мудрости" if i in [7, 15, 23, 31] else " Обычный зуб"
        status = "!!! ОТСУТСТВУЕТ" if counts[i] == 0 else ""
        print(f"  Класс {i:2d} (FDI {fdi_number}): count={int(counts[i]):4d}, weight={weights[i]:.3f} {tooth_type} {status}")
    
    print(f"\nСтатистика весов:")
    print(f"  Минимальный вес: {weights.min():.3f}")
    print(f"  Максимальный вес: {weights.max():.3f}")
    print(f"  Средний вес: {weights.mean():.3f}")
    print(f"  Соотношение max/min: {weights.max() / weights.min():.2f}x")
    
    # Дополнительная статистика по отсутствующим классам
    if len(missing_classes) > 0:
        print(f"\n!  Отсутствующих классов: {len(missing_classes)}/{num_classes}")
    else:
        print(f"\n+ Все {num_classes} классов присутствуют в датасете")
    
    print(f"{'='*70}\n")
    
    return weights_tensor


def train_model(
    dataset_path,
    coco_annotations_path,
    output_dir,
    num_classes=32,
    resume_from=None,
    eval_period=1000,
    use_augmentations=True,
    max_iter=10000,
    batch_size=2,
    base_lr=0.00025,
    warmup_iters=500,
    lr_steps=None,
    checkpoint_period=1000,
    class_weights=False,
    use_class_weights=None,  
    class_weight_method='inverse_freq',
    class_weight_power=1.0,
    train_image_sizes=(640, 720, 800),
    test_image_size=800
):
    """
    Основная функция обучения с периодической оценкой
    
    Args:
        dataset_path: путь к датасету
        coco_annotations_path: путь к COCO аннотациям
        output_dir: директория для сохранения
        num_classes: количество классов (зубов)
        resume_from: путь к чекпоинту для дообучения
        eval_period: период оценки на валидации (итераций)
        use_augmentations: использовать ли аугментации (требует augmentations.py)
        max_iter: максимальное количество итераций обучения
        batch_size: размер батча (количество изображений за итерацию)
        base_lr: начальный learning rate
        warmup_iters: количество итераций для warmup (постепенное увеличение LR)
        lr_steps: кортеж итераций для снижения LR (если None, вычисляется как 60% и 80% от max_iter)
        checkpoint_period: период сохранения чекпоинтов (итераций)
        class_weights: использовать ли веса классов для балансировки
        class_weight_method: метод вычисления весов ('inverse_freq', 'sqrt_inverse_freq', 'effective_samples')
        class_weight_power: степень усиления весов (1.0 = без усиления, >1.0 = сильнее)
        train_image_sizes: размеры для обучения (tuple или int). Если tuple - multi-scale training
        test_image_size: размер для валидации и inference (int)
    """
    
    setup_logger()
    
    register_datasets(dataset_path, coco_annotations_path)
    
    # Вычисляем class weights если нужно
    computed_class_weights = None
    if class_weights:
        computed_class_weights = compute_class_weights(
            coco_annotations_path=coco_annotations_path,
            num_classes=num_classes,
            method=class_weight_method,
            power=class_weight_power
        )
    
    # Получаем размер датасета
    train_json = Path(coco_annotations_path) / "annotations_train.json"
    with open(train_json, 'r') as f:
        train_data = json.load(f)
    num_train_images = len(train_data['images'])
    
    cfg = setup_config(
        num_classes=num_classes,
        output_dir=output_dir,
        resume_from=resume_from,
        eval_period=eval_period,
        max_iter=max_iter,
        batch_size=batch_size,
        base_lr=base_lr,
        warmup_iters=warmup_iters,
        lr_steps=lr_steps,
        checkpoint_period=checkpoint_period,
        use_augmentations=use_augmentations,
        train_image_sizes=train_image_sizes,
        test_image_size=test_image_size
    )
    
    # Расчет эпох
    iterations_per_epoch = num_train_images / cfg.SOLVER.IMS_PER_BATCH
    num_epochs = cfg.SOLVER.MAX_ITER / iterations_per_epoch
    
    # История обучения
    history = TrainingHistory(Path(output_dir) / "training_history.json")
    if resume_from:
        history.load()
    
    # Validation loader
    val_loader = build_detection_test_loader(cfg, "teeth_val")
    
    # Создаем trainer с class weights и флагом аугментаций
    trainer = MaskRCNNTrainer(
        cfg, 
        val_loader, 
        "teeth_val", 
        history, 
        class_weights=computed_class_weights,
        use_augmentations=use_augmentations
    )
    trainer.resume_or_load(resume=resume_from is not None)
    
    print(f"\n{'='*70}")
    print(f"НАЧАЛО ОБУЧЕНИЯ")
    print(f"{'='*70}")
    print(f"\nДатасет:")
    print(f"  Изображений в train: {num_train_images}")
    print(f"  Изображений в valid: {len(DatasetCatalog.get('teeth_val'))}")
    print(f"\nМодель:")
    print(f"  Архитектура: Mask R-CNN X-101-32x8d-FPN")
    print(f"  Количество классов: {num_classes}")
    print(f"  Аугментации: {'+ Включены' if use_augmentations else '- Отключены'}")
    print(f"  Class Weights: {'+ Включены' if class_weights else '- Отключены'}")
    if class_weights:
        print(f"    Метод: {class_weight_method}")
        if class_weight_power != 1.0:
            print(f"    Степень усиления: {class_weight_power}")
    print(f"\nПараметры обучения:")
    print(f"  Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"  Warmup iterations: {cfg.SOLVER.WARMUP_ITERS}")
    print(f"  LR decay steps: {cfg.SOLVER.STEPS}")
    print(f"  LR decay gamma: {cfg.SOLVER.GAMMA}")
    print(f"  Максимум итераций: {cfg.SOLVER.MAX_ITER:,}")
    print(f"  Итераций в эпохе: {iterations_per_epoch:.0f}")
    print(f"  Количество эпох: {num_epochs:.2f}")
    print(f"\nРазмеры изображений:")
    print(f"  Train: {cfg.INPUT.MIN_SIZE_TRAIN} (multi-scale)" if len(cfg.INPUT.MIN_SIZE_TRAIN) > 1 else f"  Train: {cfg.INPUT.MIN_SIZE_TRAIN[0]}")
    print(f"  Test/Inference: {cfg.INPUT.MIN_SIZE_TEST}")
    print(f"  Max size train: {cfg.INPUT.MAX_SIZE_TRAIN}")
    print(f"  Max size test: {cfg.INPUT.MAX_SIZE_TEST}")
    print(f"\nОценка:")
    print(f"  Период оценки: каждые {eval_period} итераций")
    print(f"  Всего оценок: {cfg.SOLVER.MAX_ITER // eval_period}")
    print(f"\nСохранение:")
    print(f"  Чекпоинты: каждые {cfg.SOLVER.CHECKPOINT_PERIOD} итераций")
    print(f"  Лучшая модель: model_best.pth (по Valid AP)")
    print(f"  Последняя модель: model_final.pth")
    print(f"  Директория: {output_dir}")
    print(f"{'='*70}\n")
    
    # Обучение
    trainer.train()
    
    # Финальная оценка
    print("\nФинальная оценка модели...")
    evaluator = COCOEvaluator("teeth_val", output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, "teeth_val")
    final_results = inference_on_dataset(trainer.model, val_loader, evaluator)
    
    final_metrics = {
        'iteration': cfg.SOLVER.MAX_ITER,
        'bbox_AP': final_results.get('bbox', {}).get('AP', 0.0),
        'bbox_AP50': final_results.get('bbox', {}).get('AP50', 0.0),
        'bbox_AP75': final_results.get('bbox', {}).get('AP75', 0.0),
        'segm_AP': final_results.get('segm', {}).get('AP', 0.0),
        'segm_AP50': final_results.get('segm', {}).get('AP50', 0.0),
        'segm_AP75': final_results.get('segm', {}).get('AP75', 0.0),
    }
    
    history.add_validation_metrics(cfg.SOLVER.MAX_ITER, final_metrics)
    history.save()
    
    print(f"\n{'='*70}")
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"{'='*70}")
    print(f"\nСтатистика:")
    print(f"  Обучено итераций: {cfg.SOLVER.MAX_ITER:,}")
    print(f"  Обучено эпох: {num_epochs:.2f}")
    print(f"\nФинальные метрики:")
    if 'segm' in final_results:
        print(f"  Segmentation AP (mAP50-95): {final_metrics['segm_AP']/100:.4f}")
        print(f"  Segmentation AP50: {final_metrics['segm_AP50']/100:.4f}")
        print(f"  Segmentation AP75: {final_metrics['segm_AP75']/100:.4f}")
    else:
        print(f"  !!!  Segmentation метрики недоступны")
    
    if 'bbox' in final_results:
        print(f"  BBox AP (mAP50-95): {final_metrics['bbox_AP']/100:.4f}")
        print(f"  BBox AP50: {final_metrics['bbox_AP50']/100:.4f}")
        print(f"  BBox AP75: {final_metrics['bbox_AP75']/100:.4f}")
    
    print(f"\nСохраненные модели:")
    print(f"  model_final.pth - последняя модель")
    print(f"  model_best.pth - лучшая модель по Valid AP")
    print(f"  model_XXXXXXX.pth - чекпоинты каждые {cfg.SOLVER.CHECKPOINT_PERIOD} итераций")
    print(f"\nДиректория: {output_dir}")
    print(f"{'='*70}\n")
    
    # Автоматическая визуализация истории обучения
    print("Создание графиков истории обучения...")
    try:
        from visualize_training import TrainingVisualizer
        
        plots_dir = Path(output_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        visualizer = TrainingVisualizer(Path(output_dir) / "training_history.json")
        
        # Сохраняем все графики без показа (show=False)
        visualizer.plot_losses(save_path=plots_dir / "losses.png", show=False)
        visualizer.plot_learning_rate(save_path=plots_dir / "learning_rate.png", show=False)
        visualizer.plot_validation_metrics(save_path=plots_dir / "validation_metrics.png", show=False)
        visualizer.plot_train_val_comparison(save_path=plots_dir / "train_val_comparison.png", show=False)
        
        print(f"\n* Графики сохранены в: {plots_dir}")
        print(f"  - losses.png")
        print(f"  - learning_rate.png")
        print(f"  - validation_metrics.png")
        print(f"  - train_val_comparison.png")
        
    except Exception as e:
        print(f"\n!!!  Не удалось создать графики: {e}")
        print("Вы можете создать их вручную используя visualize_training.py")
    
    return cfg, trainer, history


if __name__ == "__main__":
    # ============================================================
    # КОНФИГУРАЦИЯ ОБУЧЕНИЯ
    # ============================================================
    
    # Пути
    DATASET_PATH = "dataset"
    COCO_ANNOTATIONS_PATH = "dataset/coco_annotations"
    OUTPUT_DIR = "./output_maskrcnn"
    
    # Модель
    NUM_CLASSES = 32
    RESUME_FROM = None  # Путь к чекпоинту для продолжения обучения (или None)
    
    # Параметры обучения
    MAX_ITER = 10000           # Количество итераций (увеличьте до 20000-30000 для лучшего качества)
    BATCH_SIZE = 2             # Размер батча (увеличьте если позволяет GPU память)
    BASE_LR = 0.00025          # Learning rate (для batch_size=2)
    WARMUP_ITERS = 500         # Warmup итераций
    LR_STEPS = None            # Шаги снижения LR (None = авто: 60% и 80% от MAX_ITER)
    
    # Оценка и сохранение
    EVAL_PERIOD = 1000         # Оценка на валидации каждые N итераций
    CHECKPOINT_PERIOD = 1000   # Сохранение чекпоинтов каждые N итераций
    
    # Аугментации
    USE_AUGMENTATIONS = True   # True = использовать аугментации из augmentations.py, False = без аугментаций
    
    # Class Weights (балансировка редких зубов)
    CLASS_WEIGHTS = False  # False = выключено, True = включить веса классов
    CLASS_WEIGHT_METHOD = 'inverse_freq'  # Метод: 'inverse_freq', 'sqrt_inverse_freq', 'effective_samples'
    CLASS_WEIGHT_POWER = 1.0   # Степень усиления (1.0 = без усиления, 1.5 = умеренное, 2.0 = сильное)
    
    
    
    cfg, trainer, history = train_model(
        dataset_path=DATASET_PATH,
        coco_annotations_path=COCO_ANNOTATIONS_PATH,
        output_dir=OUTPUT_DIR,
        num_classes=NUM_CLASSES,
        resume_from=RESUME_FROM,
        eval_period=EVAL_PERIOD,
        use_augmentations=USE_AUGMENTATIONS,
        max_iter=MAX_ITER,
        batch_size=BATCH_SIZE,
        base_lr=BASE_LR,
        warmup_iters=WARMUP_ITERS,
        lr_steps=LR_STEPS,
        checkpoint_period=CHECKPOINT_PERIOD,
        class_weights=CLASS_WEIGHTS,
        class_weight_method=CLASS_WEIGHT_METHOD,
        class_weight_power=CLASS_WEIGHT_POWER
    )
