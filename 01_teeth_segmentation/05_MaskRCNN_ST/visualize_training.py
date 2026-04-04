############################################
# Визуализация истории обучения Mask R-CNN
############################################

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class TrainingVisualizer:
    """Класс для визуализации истории обучения"""
    
    def __init__(self, history_path):
        """
        Args:
            history_path: путь к training_history.json
        """
        self.history_path = Path(history_path)
        self.history = self._load_history()
    
    def _load_history(self):
        """Загружает историю из JSON"""
        with open(self.history_path, 'r') as f:
            return json.load(f)
    
    def plot_losses(self, save_path=None, figsize=(15, 10), show=True):
        """Строит графики всех losses — train и val на одном графике"""
        if not self.history.get('iterations') or len(self.history['iterations']) == 0:
            print("!!! Нет данных о losses для визуализации")
            return

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Training vs Validation Losses', fontsize=16, fontweight='bold')

        train_iters = self.history['iterations']

        # Извлекаем val losses из validation_metrics (если есть)
        val_metrics = self.history.get('validation_metrics', [])
        # Фильтруем только те точки где есть val losses (на случай смешанной истории)
        val_metrics_with_losses = [m for m in val_metrics if 'val_total_loss' in m]
        has_val_losses = len(val_metrics_with_losses) > 0
        if has_val_losses:
            val_iters   = [m['iteration']       for m in val_metrics_with_losses]
            val_total   = [m['val_total_loss']   for m in val_metrics_with_losses]
            val_cls     = [m['val_loss_cls']     for m in val_metrics_with_losses]
            val_box_reg = [m['val_loss_box_reg'] for m in val_metrics_with_losses]
            val_mask    = [m['val_loss_mask']    for m in val_metrics_with_losses]
            val_rpn_cls = [m['val_loss_rpn_cls'] for m in val_metrics_with_losses]
            val_rpn_loc = [m['val_loss_rpn_loc'] for m in val_metrics_with_losses]
        def _plot(ax, title, train_vals, val_vals=None, train_color='b', val_color='r'):
            ax.plot(train_iters, train_vals, color=train_color, linewidth=1.5,
                    alpha=0.8, label='Train')
            if val_vals is not None:
                ax.plot(val_iters, val_vals, color=val_color, linewidth=2,
                        marker='o', markersize=4, label='Val')
                ax.legend(fontsize=9)
            ax.set_title(title)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)

        _plot(axes[0, 0], 'Total Loss',
              self.history['total_loss'],
              val_total if has_val_losses else None)

        _plot(axes[0, 1], 'Classification Loss',
              self.history['loss_cls'],
              val_cls if has_val_losses else None, 'b', 'r')

        _plot(axes[0, 2], 'Box Regression Loss',
              self.history['loss_box_reg'],
              val_box_reg if has_val_losses else None, 'b', 'r')

        _plot(axes[1, 0], 'Mask Loss',
              self.history['loss_mask'],
              val_mask if has_val_losses else None, 'b', 'r')

        _plot(axes[1, 1], 'RPN Classification Loss',
              self.history['loss_rpn_cls'],
              val_rpn_cls if has_val_losses else None, 'b', 'r')

        _plot(axes[1, 2], 'RPN Localization Loss',
              self.history['loss_rpn_loc'],
              val_rpn_loc if has_val_losses else None, 'b', 'r')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График losses сохранен: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def plot_learning_rate(self, save_path=None, figsize=(10, 6), show=True):
        """Строит график learning rate"""
        # Проверка наличия данных
        if not self.history.get('iterations') or len(self.history['iterations']) == 0:
            print("!!! Нет данных о learning rate для визуализации")
            return
        
        fig = plt.figure(figsize=figsize)
        plt.plot(self.history['iterations'], self.history['learning_rate'], 
                'b-', linewidth=2)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График LR сохранен: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def plot_validation_metrics(self, save_path=None, figsize=(15, 10), show=True):
        """Строит графики метрик валидации"""
        if not self.history.get('validation_metrics') or len(self.history['validation_metrics']) == 0:
            print("!!! Нет данных валидации для визуализации")
            return
        
        val_metrics = self.history['validation_metrics']
        iterations = [m['iteration'] for m in val_metrics]
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Validation Metrics', fontsize=16, fontweight='bold')
        
        # Segmentation AP (mAP50-95) - делим на 100 для отображения в долях
        segm_ap = [m['segm_AP']/100 for m in val_metrics]
        axes[0, 0].plot(iterations, segm_ap, 'b-o', linewidth=2, markersize=8)
        axes[0, 0].set_title('Segmentation AP (mAP50-95)', fontweight='bold')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('AP')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1])
        
        # Segmentation AP50 - делим на 100
        segm_ap50 = [m['segm_AP50']/100 for m in val_metrics]
        axes[0, 1].plot(iterations, segm_ap50, 'r-o', linewidth=2, markersize=8)
        axes[0, 1].set_title('Segmentation AP50', fontweight='bold')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('AP50')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
        
        # Segmentation AP75 - делим на 100
        segm_ap75 = [m['segm_AP75']/100 for m in val_metrics]
        axes[0, 2].plot(iterations, segm_ap75, 'g-o', linewidth=2, markersize=8)
        axes[0, 2].set_title('Segmentation AP75', fontweight='bold')
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('AP75')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim([0, 1])
        
        # BBox AP (mAP50-95) - делим на 100
        bbox_ap = [m['bbox_AP']/100 for m in val_metrics]
        axes[1, 0].plot(iterations, bbox_ap, 'm-o', linewidth=2, markersize=8)
        axes[1, 0].set_title('BBox AP (mAP50-95)', fontweight='bold')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('AP')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        
        # BBox AP50 - делим на 100
        bbox_ap50 = [m['bbox_AP50']/100 for m in val_metrics]
        axes[1, 1].plot(iterations, bbox_ap50, 'c-o', linewidth=2, markersize=8)
        axes[1, 1].set_title('BBox AP50', fontweight='bold')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('AP50')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])
        
        # BBox AP75 - делим на 100
        has_bbox_ap75 = any('bbox_AP75' in m for m in val_metrics)
        if has_bbox_ap75:
            bbox_ap75 = [m.get('bbox_AP75', 0)/100 for m in val_metrics]
            axes[1, 2].plot(iterations, bbox_ap75, 'y-o', linewidth=2, markersize=8)
            axes[1, 2].set_title('BBox AP75', fontweight='bold')
        else:
            # Если нет bbox_AP75, показываем segm_AP еще раз для симметрии
            axes[1, 2].plot(iterations, segm_ap, 'b-o', linewidth=2, markersize=8, alpha=0.5)
            axes[1, 2].set_title('Segmentation AP (duplicate)', fontweight='bold')
        
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('AP75')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График метрик сохранен: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def plot_train_val_comparison(self, save_path=None, figsize=(15, 6), show=True):
        """Сравнивает train loss и validation metrics"""
        if not self.history.get('validation_metrics') or len(self.history['validation_metrics']) == 0:
            print("!!! Нет данных валидации для сравнения")
            return
        
        if not self.history.get('iterations') or len(self.history['iterations']) == 0:
            print("!!! Нет данных обучения для сравнения")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Training Loss vs Validation Metrics', fontsize=16, fontweight='bold')
        
        # График 1: Total Loss (train) и Segmentation AP (validation)
        ax1 = axes[0]
        ax2 = ax1.twinx()
        
        # Train loss
        train_iters = self.history['iterations']
        train_loss = self.history['total_loss']
        line1 = ax1.plot(train_iters, train_loss, 'b-', linewidth=1, alpha=0.6, label='Train Total Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Total Loss', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        
        # Validation AP - делим на 100 для отображения в долях
        val_metrics = self.history['validation_metrics']
        val_iters = [m['iteration'] for m in val_metrics]
        val_ap = [m['segm_AP']/100 for m in val_metrics]
        line2 = ax2.plot(val_iters, val_ap, 'r-o', linewidth=2, markersize=8, label='Valid Segm AP')
        ax2.set_ylabel('Segmentation AP', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim([0, 1])
        
        # Легенда
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        ax1.set_title('Loss vs AP')
        
        # График 2: Mask Loss (train) и Segmentation AP50 (validation)
        ax3 = axes[1]
        ax4 = ax3.twinx()
        
        # Train mask loss
        mask_loss = self.history['loss_mask']
        line3 = ax3.plot(train_iters, mask_loss, 'g-', linewidth=1, alpha=0.6, label='Train Mask Loss')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Mask Loss', color='g')
        ax3.tick_params(axis='y', labelcolor='g')
        ax3.grid(True, alpha=0.3)
        
        # Validation AP50 - делим на 100 для отображения в долях
        val_ap50 = [m['segm_AP50']/100 for m in val_metrics]
        line4 = ax4.plot(val_iters, val_ap50, 'm-o', linewidth=2, markersize=8, label='Valid Segm AP50')
        ax4.set_ylabel('Segmentation AP50', color='m')
        ax4.tick_params(axis='y', labelcolor='m')
        ax4.set_ylim([0, 1])
        
        # Легенда
        lines = line3 + line4
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper left')
        ax3.set_title('Mask Loss vs AP50')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сравнения сохранен: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def plot_all(self, output_dir=None, show=True):
        """Строит все графики"""
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.plot_losses(save_path=output_dir / "losses.png", show=show)
            self.plot_learning_rate(save_path=output_dir / "learning_rate.png", show=show)
            self.plot_validation_metrics(save_path=output_dir / "validation_metrics.png", show=show)
            self.plot_train_val_comparison(save_path=output_dir / "train_val_comparison.png", show=show)
        else:
            self.plot_losses(show=show)
            self.plot_learning_rate(show=show)
            self.plot_validation_metrics(show=show)
            self.plot_train_val_comparison(show=show)
    
    def print_summary(self):
        """Выводит сводку обучения"""
        print("\n" + "="*60)
        print("СВОДКА ОБУЧЕНИЯ")
        print("="*60)
        
        # Проверка наличия данных обучения
        has_training_data = self.history.get('iterations') and len(self.history['iterations']) > 0
        has_validation_data = self.history.get('validation_metrics') and len(self.history['validation_metrics']) > 0
        
        if has_training_data:
            print(f"\nВсего итераций: {len(self.history['iterations'])}")
            print(f"Последняя итерация: {self.history['iterations'][-1]}")
            print(f"\nФинальные losses:")
            print(f"  Total Loss: {self.history['total_loss'][-1]:.4f}")
            print(f"  Classification Loss: {self.history['loss_cls'][-1]:.4f}")
            print(f"  Box Regression Loss: {self.history['loss_box_reg'][-1]:.4f}")
            print(f"  Mask Loss: {self.history['loss_mask'][-1]:.4f}")
        else:
            print("\n!!! Нет данных об обучении (losses)")
        
        if has_validation_data:
            last_val = self.history['validation_metrics'][-1]
            print(f"\nФинальные метрики валидации:")
            print(f"  Segmentation AP (mAP50-95): {last_val['segm_AP']/100:.4f}")
            print(f"  Segmentation AP50: {last_val['segm_AP50']/100:.4f}")
            print(f"  Segmentation AP75: {last_val['segm_AP75']/100:.4f}")
            print(f"  BBox AP (mAP50-95): {last_val['bbox_AP']/100:.4f}")
            print(f"  BBox AP50: {last_val['bbox_AP50']/100:.4f}")
            if 'bbox_AP75' in last_val:
                print(f"  BBox AP75: {last_val['bbox_AP75']/100:.4f}")
            
            # Лучшие результаты
            best_ap = max(self.history['validation_metrics'], key=lambda x: x['segm_AP'])
            print(f"\nЛучший Segmentation AP: {best_ap['segm_AP']/100:.4f} "
                  f"(итерация {best_ap['iteration']})")
        else:
            print("\n!!! Нет данных валидации")
        
        print("="*60 + "\n")


def compare_models(history_paths, labels, metric='segm_AP', save_path=None):
    """
    Сравнивает несколько моделей по выбранной метрике
    
    Args:
        history_paths: список путей к training_history.json
        labels: список названий моделей
        metric: метрика для сравнения
        save_path: путь для сохранения графика
    """
    plt.figure(figsize=(12, 6))
    
    for history_path, label in zip(history_paths, labels):
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        if history['validation_metrics']:
            val_metrics = history['validation_metrics']
            iterations = [m['iteration'] for m in val_metrics]
            # Делим на 100 для отображения в долях (0-1)
            values = [m[metric]/100 for m in val_metrics]
            plt.plot(iterations, values, '-o', linewidth=2, markersize=6, label=label)
    
    plt.title(f'Model Comparison: {metric}', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сравнения сохранен: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Пример использования
    visualizer = TrainingVisualizer("./output_maskrcnn/training_history.json")
    
    # Вывод сводки
    visualizer.print_summary()
    
    # Построение всех графиков
    visualizer.plot_all(output_dir="./output_maskrcnn/plots")
    
    # Или по отдельности
    # visualizer.plot_losses()
    # visualizer.plot_learning_rate()
    # visualizer.plot_validation_metrics()
