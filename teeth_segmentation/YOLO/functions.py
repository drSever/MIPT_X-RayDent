#########################################################################################
# Функции для решения задачи сегментации зубов на ортопантомогаммах с использованием YOLO
#########################################################################################

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
    

### КРОП для исходного датасета ###

def calculate_crop_coordinates(image_shape: Tuple[int, int], crop_settings: Dict[str, float]) -> Tuple[int, int, int, int]:
    """
    Вычисляет координаты для кропа изображения на основе настроек.

    Args:
        image_shape: Размеры изображения (height, width)
        crop_settings: Словарь с настройками кропа

    Returns:
        Tuple с координатами (y1, y2, x1, x2) для кропа
    """
    height, width = image_shape[:2]

    # Вычисляем пиксели для обрезки
    left_pixels = int(width * crop_settings['left_crop'])
    right_pixels = int(width * crop_settings['right_crop'])
    top_pixels = int(height * crop_settings['top_crop'])
    bottom_pixels = int(height * crop_settings['bottom_crop'])

    # Координаты для кропа
    y1 = top_pixels
    y2 = height - bottom_pixels
    x1 = left_pixels
    x2 = width - right_pixels

    # Проверяем, что координаты корректны
    if y2 <= y1 or x2 <= x1:
        raise ValueError(f"Некорректные настройки кропа. Результирующий размер: {y2-y1}x{x2-x1}")

    return y1, y2, x1, x2


def crop_image(image: np.ndarray, crop_settings: Dict[str, float]) -> np.ndarray:
    """
    Обрезает изображение согласно настройкам кропа.

    Args:
        image: Входное изображение (numpy array)
        crop_settings: Словарь с настройками кропа

    Returns:
        Обрезанное изображение
    """
    y1, y2, x1, x2 = calculate_crop_coordinates(image.shape, crop_settings)
    return image[y1:y2, x1:x2]


def crop_yolo_annotations(annotation_path: str, crop_settings: Dict[str, float],
                         original_shape: Tuple[int, int]) -> List[str]:
    """
    Обрезает YOLO аннотации согласно настройкам кропа.

    Args:
        annotation_path: Путь к файлу аннотации
        crop_settings: Словарь с настройками кропа
        original_shape: Оригинальные размеры изображения (height, width)

    Returns:
        Список строк с обновленными аннотациями
    """
    # Если аннотаций нет
    if not os.path.exists(annotation_path):
        return []

    height, width = original_shape[:2]
    y1, y2, x1, x2 = calculate_crop_coordinates(original_shape, crop_settings)

    # Новые размеры после кропа
    new_height = y2 - y1
    new_width = x2 - x1

    cropped_annotations = []

    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        data = line.strip().split()
        if len(data) < 3:
            continue

        class_id = data[0]
        polygon_coords = list(map(float, data[1:]))

        # Преобразуем нормализованные координаты в абсолютные
        abs_coords = []
        for i in range(0, len(polygon_coords), 2):
            if i + 1 < len(polygon_coords):
                x_abs = polygon_coords[i] * width
                y_abs = polygon_coords[i + 1] * height
                abs_coords.extend([x_abs, y_abs])

        # Применяем кроп к координатам
        cropped_coords = []
        valid_points = 0

        for i in range(0, len(abs_coords), 2):
            if i + 1 < len(abs_coords):
                x_cropped = abs_coords[i] - x1
                y_cropped = abs_coords[i + 1] - y1

                # Проверяем, что точка находится в пределах обрезанного изображения
                if 0 <= x_cropped <= new_width and 0 <= y_cropped <= new_height:
                    # Нормализуем координаты относительно нового размера
                    x_norm = x_cropped / new_width
                    y_norm = y_cropped / new_height
                    cropped_coords.extend([x_norm, y_norm])
                    valid_points += 1

        # Сохраняем аннотацию только если осталось достаточно точек для полигона
        if valid_points >= 3:
            coord_str = ' '.join([f'{coord:.6f}' for coord in cropped_coords])
            cropped_annotations.append(f'{class_id} {coord_str}')

    return cropped_annotations


def process_dataset_with_crop(dataset_dir: str, crop_settings: Dict[str, float],
                             output_dir: str = None) -> str:
    """
    Обрабатывает весь датасет с применением кропа к изображениям и аннотациям.

    Args:
        dataset_dir: Путь к исходному датасету
        crop_settings: Настройки кропа
        output_dir: Путь для сохранения обработанного датасета

    Returns:
        Путь к обработанному датасету
    """
    if output_dir is None:
        output_dir = dataset_dir + '_cropped'

    print(f"Обработка датасета с кропом...")
    print(f"Исходный датасет: {dataset_dir}")
    print(f"Выходной датасет: {output_dir}")

    # Создаем структуру директорий
    os.makedirs(output_dir, exist_ok=True)

    # Обрабатываем каждый split (train, valid, test)
    splits = ['train', 'valid', 'test']
    total_processed = 0
    total_annotations_kept = 0
    total_annotations_removed = 0

    for split in splits:
        images_dir = os.path.join(dataset_dir, split, 'images')
        labels_dir = os.path.join(dataset_dir, split, 'labels')

        if not os.path.exists(images_dir):
            print(f"Пропускаем {split} - директория не найдена: {images_dir}")
            continue

        # Создаем выходные директории
        output_images_dir = os.path.join(output_dir, split, 'images')
        output_labels_dir = os.path.join(output_dir, split, 'labels')
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        # Получаем список изображений
        image_files = [f for f in os.listdir(images_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"\nОбработка {split}: {len(image_files)} изображений")

        split_processed = 0
        split_annotations_kept = 0
        split_annotations_removed = 0

        for image_file in tqdm(image_files, desc=f'Обработка {split}'):
            # Пути к файлам
            image_path = os.path.join(images_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)

            # Выходные пути
            output_image_path = os.path.join(output_images_dir, image_file)
            output_label_path = os.path.join(output_labels_dir, label_file)

            try:
                # Загружаем и обрезаем изображение
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Не удалось загрузить изображение: {image_path}")
                    continue

                original_shape = image.shape
                cropped_image = crop_image(image, crop_settings)

                # Сохраняем обрезанное изображение
                cv2.imwrite(output_image_path, cropped_image)

                # Обрабатываем аннотации
                original_annotations_count = 0
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        original_annotations_count = len(f.readlines())

                cropped_annotations = crop_yolo_annotations(label_path, crop_settings, original_shape)

                # Сохраняем обновленные аннотации
                with open(output_label_path, 'w') as f:
                    for annotation in cropped_annotations:
                        f.write(annotation + '\n')

                # Статистика
                kept = len(cropped_annotations)
                removed = original_annotations_count - kept

                split_processed += 1
                split_annotations_kept += kept
                split_annotations_removed += removed

            except Exception as e:
                print(f"Ошибка при обработке {image_file}: {str(e)}")
                continue

        print(f"  Обработано изображений: {split_processed}")
        print(f"  Аннотаций сохранено: {split_annotations_kept}")
        print(f"  Аннотаций удалено: {split_annotations_removed}")

        total_processed += split_processed
        total_annotations_kept += split_annotations_kept
        total_annotations_removed += split_annotations_removed

    # Копируем data.yaml и обновляем пути
    original_yaml = os.path.join(dataset_dir, 'data.yaml')
    output_yaml = os.path.join(output_dir, 'data.yaml')

    if os.path.exists(original_yaml):
        shutil.copy2(original_yaml, output_yaml)

        # Обновляем пути в yaml файле
        with open(output_yaml, 'r') as f:
            yaml_content = f.read()

        # Заменяем пути на новые
        yaml_content = yaml_content.replace(dataset_dir, output_dir)

        with open(output_yaml, 'w') as f:
            f.write(yaml_content)

    print(f"\n=== ИТОГОВАЯ СТАТИСТИКА ===")
    print(f"Всего обработано изображений: {total_processed}")
    print(f"Всего аннотаций сохранено: {total_annotations_kept}")
    print(f"Всего аннотаций удалено: {total_annotations_removed}")
    print(f"Обработанный датасет сохранен в: {output_dir}")

    return output_dir


def visualize_crop_effect(image_path: str, annotation_path: str, crop_settings: Dict[str, float]):
    """
    Визуализирует эффект кропа на изображении и аннотациях.

    Args:
        image_path: Путь к изображению
        annotation_path: Путь к аннотации
        crop_settings: Настройки кропа
    """
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape

    # Применяем кроп
    cropped_image = crop_image(image_rgb, crop_settings)

    # Обрабатываем аннотации
    cropped_annotations = crop_yolo_annotations(annotation_path, crop_settings, original_shape)

    # Создаем визуализацию
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Оригинальное изображение
    axes[0].imshow(image_rgb)
    axes[0].set_title('Оригинальное изображение', fontsize=16, fontweight='bold')
    axes[0].axis('off')

    # Показываем область кропа
    y1, y2, x1, x2 = calculate_crop_coordinates(original_shape, crop_settings)
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=3, edgecolor='red', facecolor='none')
    axes[0].add_patch(rect)
    axes[0].text(x1, y1-10, 'Область кропа', color='red', fontsize=12, fontweight='bold')

    # Обрезанное изображение
    axes[1].imshow(cropped_image)
    axes[1].set_title('Обрезанное изображение', fontsize=16, fontweight='bold')
    axes[1].axis('off')

    # Добавляем информацию о размерах
    orig_h, orig_w = original_shape[:2]
    crop_h, crop_w = cropped_image.shape[:2]

    info_text = f"Оригинал: {orig_w}x{orig_h}\nОбрезанное: {crop_w}x{crop_h}\nАннотаций сохранено: {len(cropped_annotations)}"
    axes[1].text(10, crop_h-50, info_text, color='white', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.7))

    plt.tight_layout()
    plt.show()

    return cropped_image, cropped_annotations


def demonstrate_crop_effect_on_random_image(dataset_dir, crop_settings, apply_crop=True):
    """
    Демонстрирует эффект кропа на случайном изображении из случайной подвыборки

    Args:
        dataset_dir: Путь к датасету
        crop_settings: Словарь с настройками кропа
        apply_crop: Применять ли демонстрацию кропа

    Returns:
        tuple: (cropped_image, cropped_annotations) или (None, None) если демонстрация не выполнена
    """
    if not apply_crop:
        print("Демонстрация кропа пропущена (кроп не применяется)")
        return None, None

    # Находим случайное изображение из случайной подвыборки для демонстрации
    demo_splits = ['train', 'valid', 'test']

    # Получаем путь к оригинальному датасету (убираем '_cropped' если есть)
    original_dataset_dir = dataset_dir.replace('_cropped', '') if '_cropped' in dataset_dir else dataset_dir

    # Собираем все доступные изображения из всех подвыборок
    all_available_images = []

    for split in demo_splits:
        images_dir = os.path.join(original_dataset_dir, split, 'images')
        labels_dir = os.path.join(original_dataset_dir, split, 'labels')

        if os.path.exists(images_dir):
            image_files = [f for f in os.listdir(images_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for image_file in image_files:
                image_path = os.path.join(images_dir, image_file)
                annotation_path = os.path.join(labels_dir,
                                             os.path.splitext(image_file)[0] + '.txt')
                all_available_images.append({
                    'split': split,
                    'image_path': image_path,
                    'annotation_path': annotation_path,
                    'filename': image_file
                })

    if not all_available_images:
        print("Не найдено изображений для демонстрации кропа")
        return None, None

    # Выбираем случайное изображение
    random_image = random.choice(all_available_images)
    demo_image_path = random_image['image_path']
    demo_annotation_path = random_image['annotation_path']

    print(f"Демонстрация кропа на случайном изображении:")
    print(f"  Подвыборка: {random_image['split']}")
    print(f"  Файл: {random_image['filename']}")
    print(f"  Путь: {demo_image_path}")

    if not os.path.exists(demo_image_path):
        print(f"Файл не найден: {demo_image_path}")
        return None, None

    try:
        cropped_demo, cropped_demo_annotations = visualize_crop_effect(
            demo_image_path, demo_annotation_path, crop_settings
        )

        print(f"  Результат: обработано {len(cropped_demo_annotations)} аннотаций")
        return cropped_demo, cropped_demo_annotations

    except Exception as e:
        print(f"Ошибка при демонстрации кропа: {str(e)}")
        return None, None
    
    
### Для полной оценки модели на тестовой подвыборке ###

### Для поиска и отображения масок

def parse_yolo_segmentation_mask(txt_path, img_width, img_height):
    """
    Парсинг YOLO формата сегментации и создание бинарной маски
    Формат: class_id x1 y1 x2 y2 ... xn yn
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    if not os.path.exists(txt_path):
        return mask

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        data = line.strip().split()
        if len(data) < 3:  # минимум class_id + 2 координаты
            continue

        class_id = int(data[0])
        polygon = list(map(float, data[1:]))

        # Преобразование нормализованных координат в абсолютные
        polygon_abs = []
        for i in range(0, len(polygon), 2):
            if i + 1 < len(polygon):
                x = int(polygon[i] * img_width)
                y = int(polygon[i + 1] * img_height)
                polygon_abs.append([x, y])

        if len(polygon_abs) >= 3:  # минимум 3 точки для полигона
            polygon_array = np.array(polygon_abs, dtype=np.int32)
            cv2.fillPoly(mask, [polygon_array], color=class_id + 1)  # +1 чтобы избежать 0 (фон)

    return mask


def create_multiclass_mask_from_yolo(txt_path, img_shape, num_classes):
    """
    Создание многоклассовой маски из YOLO аннотаций
    """
    height, width = img_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    if not os.path.exists(txt_path):
        return mask

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        data = line.strip().split()
        if len(data) < 3:
            continue

        class_id = int(data[0])
        polygon = list(map(float, data[1:]))

        # Создание полигона для текущего класса
        polygon_abs = []
        for i in range(0, len(polygon), 2):
            if i + 1 < len(polygon):
                x = int(polygon[i] * width)
                y = int(polygon[i + 1] * height)
                polygon_abs.append([x, y])

        if len(polygon_abs) >= 3:
            polygon_array = np.array(polygon_abs, dtype=np.int32)
            cv2.fillPoly(mask, [polygon_array], color=class_id + 1)

    return mask

### Метрики

def calculate_dice(mask1, mask2):
    """
    Расчет DICE coefficient между двумя бинарными масками
    """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    union = mask1.sum() + mask2.sum()

    if union == 0:
        return 1.0  # обе маски пустые

    dice = 2 * intersection / union
    return dice

def calculate_iou(mask1, mask2):
    """
    Расчет IoU (Intersection over Union)
    """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 1.0

    return intersection / union

def calculate_micro_macro_dice(all_pred_masks, all_true_masks, num_classes):
    """
    Корректный расчет Micro и Macro DICE
    """
    total_intersection = 0
    total_pixels_gt = 0
    total_pixels_pred = 0
    macro_dice_per_class = []
    class_presence_count = [0] * (num_classes + 1)

    for class_id in range(1, num_classes + 1):
        class_intersection = 0
        class_pixels_gt = 0
        class_pixels_pred = 0

        for pred_mask, true_mask in zip(all_pred_masks, all_true_masks):
            pred_class = (pred_mask == class_id).astype(bool)
            true_class = (true_mask == class_id).astype(bool)

            # Учитываем только если класс присутствует в ground truth
            if true_class.sum() > 0:
                intersection = np.logical_and(pred_class, true_class).sum()
                pixels_gt = true_class.sum()
                pixels_pred = pred_class.sum()

                class_intersection += intersection
                class_pixels_gt += pixels_gt
                class_pixels_pred += pixels_pred
                class_presence_count[class_id] += 1

        # DICE для класса (только где он присутствует)
        if (class_pixels_gt + class_pixels_pred) > 0 and class_presence_count[class_id] > 0:
            class_dice = 2.0 * class_intersection / (class_pixels_gt + class_pixels_pred)
            macro_dice_per_class.append(class_dice)

            # Для micro DICE
            total_intersection += class_intersection
            total_pixels_gt += class_pixels_gt
            total_pixels_pred += class_pixels_pred

    # Micro DICE (по всем пикселям всех классов)
    if (total_pixels_gt + total_pixels_pred) > 0:
        micro_dice = 2.0 * total_intersection / (total_pixels_gt + total_pixels_pred)
    else:
        micro_dice = 0

    # Macro DICE (среднее по классам)
    macro_dice = np.mean(macro_dice_per_class) if macro_dice_per_class else 0

    return micro_dice, macro_dice, macro_dice_per_class

def calculate_micro_macro_iou(all_pred_masks, all_true_masks, num_classes):
    """
    Расчет Micro и Macro IoU
    """
    total_intersection = 0
    total_union = 0
    macro_iou_per_class = []
    class_presence_count = [0] * (num_classes + 1)

    for class_id in range(1, num_classes + 1):
        class_intersection = 0
        class_union = 0

        for pred_mask, true_mask in zip(all_pred_masks, all_true_masks):
            pred_class = (pred_mask == class_id).astype(bool)
            true_class = (true_mask == class_id).astype(bool)

            # Учитываем только если класс присутствует в ground truth
            if true_class.sum() > 0:
                intersection = np.logical_and(pred_class, true_class).sum()
                union = np.logical_or(pred_class, true_class).sum()

                class_intersection += intersection
                class_union += union
                class_presence_count[class_id] += 1

        # IoU для класса (только где он присутствует)
        if class_union > 0 and class_presence_count[class_id] > 0:
            class_iou = class_intersection / class_union
            macro_iou_per_class.append(class_iou)

            # Для micro IoU
            total_intersection += class_intersection
            total_union += class_union

    # Micro IoU (по всем пикселям всех классов)
    micro_iou = total_intersection / total_union if total_union > 0 else 0

    # Macro IoU (среднее по классам)
    macro_iou = np.mean(macro_iou_per_class) if macro_iou_per_class else 0

    return micro_iou, macro_iou, macro_iou_per_class

def calculate_image_level_metrics(pred_mask, true_mask, num_classes):
    """
    Вычисление метрик для отдельного изображения
    """
    # Бинарные метрики (все классы вместе)
    true_binary = (true_mask > 0).astype(bool)
    pred_binary = (pred_mask > 0).astype(bool)

    image_iou = calculate_iou(true_binary, pred_binary)
    image_dice = calculate_dice(true_binary, pred_binary)

    # Micro и Macro метрики для изображения
    total_intersection = 0
    total_union = 0
    total_pixels_gt = 0
    total_pixels_pred = 0
    macro_dice_per_class = []
    macro_iou_per_class = []

    for class_id in range(1, num_classes + 1):
        true_class = (true_mask == class_id).astype(bool)
        pred_class = (pred_mask == class_id).astype(bool)

        # Учитываем только если класс присутствует в ground truth
        if true_class.sum() > 0:
            # IoU для класса
            intersection_iou = np.logical_and(true_class, pred_class).sum()
            union_iou = np.logical_or(true_class, pred_class).sum()

            if union_iou > 0:
                class_iou = intersection_iou / union_iou
                macro_iou_per_class.append(class_iou)

            # DICE для класса
            intersection_dice = np.logical_and(true_class, pred_class).sum()
            pixels_gt = true_class.sum()
            pixels_pred = pred_class.sum()

            if (pixels_gt + pixels_pred) > 0:
                class_dice = 2.0 * intersection_dice / (pixels_gt + pixels_pred)
                macro_dice_per_class.append(class_dice)

            # Для micro метрик
            total_intersection += intersection_dice
            total_pixels_gt += pixels_gt
            total_pixels_pred += pixels_pred

    # Расчет Micro метрик
    # Micro DICE = 2 × total_intersection / (total_pixels_gt + total_pixels_pred)
    if (total_pixels_gt + total_pixels_pred) > 0:
        image_micro_dice = 2.0 * total_intersection / (total_pixels_gt + total_pixels_pred)
    else:
        image_micro_dice = 0.0

    # Micro IoU = total_intersection / total_union
    # total_union = total_pixels_gt + total_pixels_pred - total_intersection
    total_union = total_pixels_gt + total_pixels_pred - total_intersection
    if total_union > 0:
        image_micro_iou = total_intersection / total_union
    else:
        image_micro_iou = 0.0

    # Macro метрики для изображения
    image_macro_iou = np.mean(macro_iou_per_class) if macro_iou_per_class else 0
    image_macro_dice = np.mean(macro_dice_per_class) if macro_dice_per_class else 0

    return {
        'image_iou': image_iou,
        'image_dice': image_dice,
        'image_micro_iou': image_micro_iou,
        'image_micro_dice': image_micro_dice,
        'image_macro_iou': image_macro_iou,
        'image_macro_dice': image_macro_dice
    }
    
### Для оценки модели и визуализации результатов

def evaluate_yolov8_dice_yolo_format(
    model_path,
    test_images_dir,
    test_labels_dir,
    conf_threshold,
    img_size,
    seed,
    apply_crop_to_inference,
    crop_settings
    ):
    """
    Оценка YOLOv8 segmentation модели с YOLO форматом масок
    """



    print(f"Полная оценка модели...")
    print(f"Применение кропа при оценке: {'ДА' if apply_crop_to_inference else 'НЕТ'}")
    if apply_crop_to_inference:
        print(f"Настройки кропа: слева {crop_settings['left_crop']*100:.1f}%, справа {crop_settings['right_crop']*100:.1f}%, сверху {crop_settings['top_crop']*100:.1f}%, снизу {crop_settings['bottom_crop']*100:.1f}%")

    # Загрузка модели
    model = YOLO(model_path)

    # Получение информации о классах
    class_names = model.names
    num_classes = len(class_names)

    # Хранение результатов
    all_pred_masks = []
    all_true_masks = []
    image_files_processed = []

    # Списки для метрик каждого изображения
    image_metrics_list = []

    # Получение списка изображений и сортировка
    image_files = sorted([f for f in os.listdir(test_images_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for image_file in tqdm(image_files, desc="Processing images"):
        # Пути к файлам
        image_path = os.path.join(test_images_dir, image_file)
        label_path = os.path.join(test_labels_dir,
                                 os.path.splitext(image_file)[0] + '.txt')

        # Загрузка изображения для получения размеров
        image = cv2.imread(image_path)
        if image is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            continue

        img_height, img_width = image.shape[:2]
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Применяем кроп если включено (для соответствия данным обучения)
        if apply_crop_to_inference:
            image_for_inference = crop_image(original_image, crop_settings)
        else:
            image_for_inference = original_image

        # Загрузка и создание ground truth маски
        true_mask = create_multiclass_mask_from_yolo(
            label_path, (img_height, img_width), num_classes
        )

        # Инференс модели на обработанном изображении
        results = model(image_for_inference, conf=conf_threshold, imgsz=img_size, verbose=False, seed=seed)

        if len(results) == 0 or results[0].masks is None:
            # Нет предсказаний - создаем пустую маску
            pred_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        else:
            # Получение предсказанной маски
            result = results[0]
            pred_mask = result.masks.data.cpu().numpy()

            if len(pred_mask.shape) == 3:
                # Множественные маски - комбинируем их
                if apply_crop_to_inference:
                    crop_height, crop_width = image_for_inference.shape[:2]
                    pred_mask_combined = np.zeros((crop_height, crop_width), dtype=np.uint8)
                else:
                    pred_mask_combined = np.zeros((pred_mask.shape[1], pred_mask.shape[2]), dtype=np.uint8)

                for i, (mask, cls) in enumerate(zip(pred_mask, result.boxes.cls)):
                    class_id = int(cls.item()) + 1  # +1 т.к. в true_mask классы с 1
                    binary_mask = (mask > 0.5).astype(np.uint8)

                    # Масштабируем маску к размеру изображения для инференса
                    if apply_crop_to_inference:
                        if binary_mask.shape != (crop_height, crop_width):
                            binary_mask = cv2.resize(binary_mask, (crop_width, crop_height),
                                                   interpolation=cv2.INTER_NEAREST)
                    else:
                        if binary_mask.shape != pred_mask_combined.shape:
                            binary_mask = cv2.resize(binary_mask, (pred_mask_combined.shape[1], pred_mask_combined.shape[0]),
                                                   interpolation=cv2.INTER_NEAREST)

                    pred_mask_combined[binary_mask == 1] = class_id

                pred_mask = pred_mask_combined
            else:
                # Одиночная маска
                pred_mask = (pred_mask[0] > 0.5).astype(np.uint8)

            # Масштабирование маски к размеру оригинального изображения
            if apply_crop_to_inference:
                # Получаем координаты кропа
                y1, y2, x1, x2 = calculate_crop_coordinates((img_height, img_width), crop_settings)
                crop_height, crop_width = image_for_inference.shape[:2]

                # Создаем маску размера оригинального изображения
                full_pred_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                # Вставляем обрезанную маску в соответствующую область
                full_pred_mask[y1:y1+crop_height, x1:x1+crop_width] = pred_mask
                pred_mask = full_pred_mask
            else:
                # Изменение размера если необходимо (без кропа)
                if pred_mask.shape != (img_height, img_width):
                    pred_mask = cv2.resize(pred_mask, (img_width, img_height),
                                         interpolation=cv2.INTER_NEAREST)

        # Сохраняем маски для корректного расчета
        all_pred_masks.append(pred_mask)
        all_true_masks.append(true_mask)
        image_files_processed.append(image_file)

        # Вычисляем метрики для текущего изображения
        image_metrics = calculate_image_level_metrics(pred_mask, true_mask, num_classes)
        image_metrics_list.append(image_metrics)

    # Корректный расчет DICE и IoU по классам (глобальные)
    micro_dice, macro_dice, per_class_dice_corrected = calculate_micro_macro_dice(
        all_pred_masks, all_true_masks, num_classes
    )

    micro_iou, macro_iou, per_class_iou_corrected = calculate_micro_macro_iou(
        all_pred_masks, all_true_masks, num_classes
    )

    return (image_metrics_list, image_files_processed,
            class_names, micro_dice, macro_dice, per_class_dice_corrected,
            micro_iou, macro_iou, per_class_iou_corrected)

def analyze_and_visualize_results(image_metrics_list, image_files, class_names,
                                 micro_dice, macro_dice, per_class_dice_corrected,
                                 micro_iou, macro_iou, per_class_iou_corrected):
    """
    Анализ и визуализация результатов
    """
    # Извлекаем метрики из списка
    image_ious = [metrics['image_iou'] for metrics in image_metrics_list]
    image_dices = [metrics['image_dice'] for metrics in image_metrics_list]
    image_micro_ious = [metrics['image_micro_iou'] for metrics in image_metrics_list]
    image_micro_dices = [metrics['image_micro_dice'] for metrics in image_metrics_list]
    image_macro_ious = [metrics['image_macro_iou'] for metrics in image_metrics_list]
    image_macro_dices = [metrics['image_macro_dice'] for metrics in image_metrics_list]

    # Общая статистика
    print("=== МЕТРИКИ ===")
    print(f"Global Micro DICE (по пикселям): {micro_dice:.4f}")
    print(f"Global Macro DICE (по классам): {macro_dice:.4f}")
    print(f"Global Micro IoU (по пикселям): {micro_iou:.4f}")
    print(f"Global Macro IoU (по классам): {macro_iou:.4f}")
    print(f"Средний IoU по изображениям: {np.mean(image_ious):.4f} ± {np.std(image_ious):.4f}")
    print(f"Средний DICE по изображениям: {np.mean(image_dices):.4f} ± {np.std(image_dices):.4f}")
    print(f"Средний Micro IoU по изображениям: {np.mean(image_micro_ious):.4f} ± {np.std(image_micro_ious):.4f}")
    print(f"Средний Micro DICE по изображениям: {np.mean(image_micro_dices):.4f} ± {np.std(image_micro_dices):.4f}")
    print(f"Средний Macro IoU по изображениям: {np.mean(image_macro_ious):.4f} ± {np.std(image_macro_ious):.4f}")
    print(f"Средний Macro DICE по изображениям: {np.mean(image_macro_dices):.4f} ± {np.std(image_macro_dices):.4f}")

    # Статистика по классам
    print("\n=== СТАТИСТИКА ПО КЛАССАМ (ЗУБАМ) ===")
    class_stats = []
    for class_id in range(1, len(class_names) + 1):
        if (class_id - 1) < len(per_class_dice_corrected) and (class_id - 1) < len(per_class_iou_corrected):
            corrected_dice = per_class_dice_corrected[class_id - 1]
            corrected_iou = per_class_iou_corrected[class_id - 1]
            class_name = class_names[class_id - 1] if (class_id - 1) in class_names else f"Class_{class_id}"

            class_stats.append({
                'Class': class_name,
                'DICE': corrected_dice,
                'IoU': corrected_iou
            })
            print(f"{class_name}: DICE = {corrected_dice:.4f}, IoU = {corrected_iou:.4f}")

    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Сравнение Global Micro и Macro DICE
    methods = ['Micro DICE', 'Macro DICE']
    values = [micro_dice, macro_dice]
    colors = ['green', 'blue']

    bars = axes[0, 0].bar(methods, values, color=colors, alpha=0.7)
    axes[0, 0].set_ylabel('DICE Score')
    axes[0, 0].set_title('Global Micro vs Macro DICE')
    axes[0, 0].set_ylim(0, 1)

    # Добавление значений на столбцы
    for bar, value in zip(bars, values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')

    # DICE по классам
    if per_class_dice_corrected:
        class_ids = list(range(1, len(per_class_dice_corrected) + 1))
        class_labels = [class_names[i-1] if (i-1) in class_names else f"Class_{i}"
                       for i in class_ids]

        bars = axes[0, 1].bar(class_ids, per_class_dice_corrected,
                             color=['red' if x < 0.75 else 'blue' for x in per_class_dice_corrected],
                             alpha=0.7)
        axes[0, 1].set_xlabel('Class (Tooth)')
        axes[0, 1].set_ylabel('DICE Score')
        axes[0, 1].set_title('DICE Score per Tooth Class')
        axes[0, 1].axhline(y=0.75, color='red', linestyle='--', label='Порог 0.75')
        axes[0, 1].legend()

        # Добавляем номера классов на ось X
        axes[0, 1].set_xticks(class_ids)
        axes[0, 1].set_xticklabels(class_labels, rotation=45, ha='right')

    # IoU по классам
    if per_class_iou_corrected:
        class_ids = list(range(1, len(per_class_iou_corrected) + 1))
        class_labels = [class_names[i-1] if (i-1) in class_names else f"Class_{i}"
                       for i in class_ids]

        bars = axes[1, 0].bar(class_ids, per_class_iou_corrected,
                             color=['red' if x < 0.6 else 'blue' for x in per_class_iou_corrected],
                             alpha=0.7)
        axes[1, 0].set_xlabel('Class (Tooth)')
        axes[1, 0].set_ylabel('IoU Score')
        axes[1, 0].set_title('IoU Score per Tooth Class')
        axes[1, 0].axhline(y=0.6, color='red', linestyle='--', label='Порог 0.6')
        axes[1, 0].legend()

        # Добавляем номера классов на ось X
        axes[1, 0].set_xticks(class_ids)
        axes[1, 0].set_xticklabels(class_labels, rotation=45, ha='right')

    # Сравнение средних метрик по изображениям
    metrics = ['Image IoU', 'Image DICE', 'Micro IoU', 'Micro DICE', 'Macro IoU', 'Macro DICE']
    values = [
        np.mean(image_ious),
        np.mean(image_dices),
        np.mean(image_micro_ious),
        np.mean(image_micro_dices),
        np.mean(image_macro_ious),
        np.mean(image_macro_dices)
    ]
    colors = ['lightblue', 'lightgreen', 'orange', 'yellow', 'pink', 'lavender']

    bars = axes[1, 1].bar(metrics, values, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Средние метрики по изображениям')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].tick_params(axis='x', rotation=45)

    # Добавление значений на столбцы
    for bar, value in zip(bars, values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()

    return class_stats

def save_detailed_results(image_files, image_metrics_list, output_path_csv):
    """
    Сохранение детальных результатов в CSV
    """
    # Создаем DataFrame с метриками для каждого изображения
    results_data = []
    for image_file, metrics in zip(image_files, image_metrics_list):
        row = {
            'Image': image_file,
            'IoU_Score': metrics['image_iou'],
            'DICE_Score': metrics['image_dice'],
            'Micro_IoU': metrics['image_micro_iou'],
            'Micro_DICE': metrics['image_micro_dice'],
            'Macro_IoU': metrics['image_macro_iou'],
            'Macro_DICE': metrics['image_macro_dice']
        }
        results_data.append(row)

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(output_path_csv, index=False)
    print(f"Детальные результаты сохранены в: {output_path_csv}")

    return results_df

def save_metrics_summary(micro_dice, macro_dice, micro_iou, macro_iou,
                        per_class_dice, per_class_iou, class_names,
                        image_metrics_list, image_files, output_path_json):
    """
    Сохранение суммарных метрик в JSON файл
    """
    # Вычисляем средние по изображениям
    avg_image_iou = np.mean([m['image_iou'] for m in image_metrics_list])
    avg_image_dice = np.mean([m['image_dice'] for m in image_metrics_list])
    avg_image_micro_iou = np.mean([m['image_micro_iou'] for m in image_metrics_list])
    avg_image_micro_dice = np.mean([m['image_micro_dice'] for m in image_metrics_list])
    avg_image_macro_iou = np.mean([m['image_macro_iou'] for m in image_metrics_list])
    avg_image_macro_dice = np.mean([m['image_macro_dice'] for m in image_metrics_list])

    summary = {
        'global_metrics': {
            'micro_dice': micro_dice,
            'macro_dice': macro_dice,
            'micro_iou': micro_iou,
            'macro_iou': macro_iou
        },
        'average_image_metrics': {
            'image_iou': avg_image_iou,
            'image_dice': avg_image_dice,
            'image_micro_iou': avg_image_micro_iou,
            'image_micro_dice': avg_image_micro_dice,
            'image_macro_iou': avg_image_macro_iou,
            'image_macro_dice': avg_image_macro_dice
        },
        'per_class_metrics': {},
        'per_image_metrics': {}
    }

    # Метрики по классам
    for class_id in range(1, len(class_names) + 1):
        if (class_id - 1) < len(per_class_dice) and (class_id - 1) < len(per_class_iou):
            class_name = class_names[class_id - 1] if (class_id - 1) in class_names else f"Class_{class_id}"
            summary['per_class_metrics'][class_name] = {
                'dice': float(per_class_dice[class_id - 1]),
                'iou': float(per_class_iou[class_id - 1])
            }

    # Метрики по изображениям (только первые 5 для примера)
    for i, (image_file, metrics) in enumerate(zip(image_files[:5], image_metrics_list[:5])):
        summary['per_image_metrics'][image_file] = {
            'image_iou': float(metrics['image_iou']),
            'image_dice': float(metrics['image_dice']),
            'image_micro_iou': float(metrics['image_micro_iou']),
            'image_micro_dice': float(metrics['image_micro_dice']),
            'image_macro_iou': float(metrics['image_macro_iou']),
            'image_macro_dice': float(metrics['image_macro_dice'])
        }

    with open(output_path_json, 'w') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"Сводные метрики сохранены в: {output_path_json}")
    return summary

def run_yolo_segmentation_evaluation(
    model_path,
    img_size,
    apply_crop_to_inference,
    crop_settings,
    test_images_dir,
    test_labels_dir,
    conf_threshold,
    output_path_csv,
    output_path_json,
    seed
):
    """
    Полный пайплайн оценки сегментации зубов
    """



    # Проверка существования директорий
    if not os.path.exists(test_images_dir):
        print(f"Директория с изображениями не найдена: {test_images_dir}")
        return

    if not os.path.exists(test_labels_dir):
        print(f"Директория с разметкой не найдена: {test_labels_dir}")
        return

    # Запуск оценки
    (image_metrics_list, image_files, class_names,
     micro_dice, macro_dice, per_class_dice_corrected,
     micro_iou, macro_iou, per_class_iou_corrected) = evaluate_yolov8_dice_yolo_format(
        model_path,
        test_images_dir,
        test_labels_dir,
        conf_threshold,
        img_size,
        seed,
        apply_crop_to_inference,
        crop_settings
        )

    # Анализ результатов
    class_stats = analyze_and_visualize_results(
        image_metrics_list, image_files, class_names,
        micro_dice, macro_dice, per_class_dice_corrected,
        micro_iou, macro_iou, per_class_iou_corrected
    )

    # Сохранение детальных результатов
    results_df = save_detailed_results(image_files, image_metrics_list, output_path_csv)

    # Сохранение сводных метрик
    metrics_summary = save_metrics_summary(
        micro_dice, macro_dice, micro_iou, macro_iou,
        per_class_dice_corrected, per_class_iou_corrected, class_names,
        image_metrics_list, image_files, output_path_json
    )

    # Вывод сводки
    print("\n=== СВОДКА РЕЗУЛЬТАТОВ ===")
    print(f"Обработано изображений: {len(image_metrics_list)}")
    print(f"Global Micro DICE: {micro_dice:.4f}")
    print(f"Global Macro DICE: {macro_dice:.4f}")
    print(f"Global Micro IoU: {micro_iou:.4f}")
    print(f"Global Macro IoU: {macro_iou:.4f}")

    # Анализ проблемных зубов по DICE
    problem_teeth_dice = [(class_names[i-1] if (i-1) in class_names else f"Class_{i}", score)
                         for i, score in enumerate(per_class_dice_corrected, 1) if score < 0.75]
    if problem_teeth_dice:
        print(f"\nПроблемные зубы (DICE < 0.75): {len(problem_teeth_dice)}")
        for tooth, score in sorted(problem_teeth_dice, key=lambda x: x[1]):
            print(f"  {tooth}: {score:.4f}")

    # Анализ проблемных зубов по IoU
    problem_teeth_iou = [(class_names[i-1] if (i-1) in class_names else f"Class_{i}", score)
                        for i, score in enumerate(per_class_iou_corrected, 1) if score < 0.6]
    if problem_teeth_iou:
        print(f"\nПроблемные зубы (IoU < 0.6): {len(problem_teeth_iou)}")
        for tooth, score in sorted(problem_teeth_iou, key=lambda x: x[1]):
            print(f"  {tooth}: {score:.4f}")

    return (image_metrics_list, class_stats, results_df, metrics_summary,
            micro_dice, macro_dice, per_class_dice_corrected,
            micro_iou, macro_iou, per_class_iou_corrected)
    
    
### Для инференса на новых изображениях ###

def inference_orthopantomogram(
    model_path,
    image_path,
    conf_threshold,
    alpha,
    img_size,
    apply_crop_to_inference,
    crop_settings
    ):
    """
    Инференс модели YOLO segmentation на новой ортопантомограмме
    с цветной визуализацией результатов
    """

    # Загрузка модели
    model = YOLO(model_path)

    # Получение информации о классах
    class_names = model.names
    num_classes = len(class_names)

    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image_rgb.copy()

    # Применяем кроп если включено
    if apply_crop_to_inference:
        print(f"Применение кропа к изображению для инференса...")
        image_for_inference = crop_image(image_rgb, crop_settings)
        print(f"Размер до кропа: {image_rgb.shape[:2]}, после кропа: {image_for_inference.shape[:2]}")
    else:
        image_for_inference = image_rgb

    # Инференс на обработанном изображении
    results = model(image_for_inference, conf=conf_threshold, imgsz=img_size)

    if len(results) == 0 or results[0].masks is None:
        print("Не обнаружено зубов на изображении")
        return original_image, [], []

    result = results[0]
    masks_data = result.masks.data.cpu().numpy()
    classes_data = result.boxes.cls.cpu().numpy()
    confidence_scores = result.boxes.conf.cpu().numpy()

    # Создание цветовой карты для классов
    colors = colormaps['tab20'].resampled(num_classes + 1)

    # Создание изображения с наложением масок
    overlay = original_image.copy()
    mask_overlay = np.zeros_like(original_image)

    detected_teeth = []

    # Вычисляем параметры для масштабирования координат обратно к оригинальному изображению
    if apply_crop_to_inference:
        # Получаем координаты кропа
        y1, y2, x1, x2 = calculate_crop_coordinates(original_image.shape, crop_settings)
        crop_height, crop_width = image_for_inference.shape[:2]
        orig_height, orig_width = original_image.shape[:2]
    else:
        y1, x1 = 0, 0
        crop_height, crop_width = original_image.shape[:2]
        orig_height, orig_width = original_image.shape[:2]

    for i, (mask, cls, conf) in enumerate(zip(masks_data, classes_data, confidence_scores)):
        class_id = int(cls)
        class_name = class_names[class_id]
        color = np.array(colors(class_id + 1)[:3]) * 255  # +1 чтобы избежать черного цвета

        # Бинаризация маски
        binary_mask = (mask > 0.5).astype(np.uint8)

        # Масштабирование маски к размеру оригинального изображения
        if apply_crop_to_inference:
            # Сначала изменяем размер маски до размера обрезанного изображения
            if binary_mask.shape != (crop_height, crop_width):
                binary_mask = cv2.resize(binary_mask, (crop_width, crop_height),
                                       interpolation=cv2.INTER_NEAREST)

            # Создаем маску размера оригинального изображения
            full_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
            # Вставляем обрезанную маску в соответствующую область
            full_mask[y1:y1+crop_height, x1:x1+crop_width] = binary_mask
            binary_mask = full_mask
        else:
            # Изменение размера если необходимо (без кропа)
            if binary_mask.shape != original_image.shape[:2]:
                binary_mask = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)

        # Создание цветной маски
        color_mask = np.zeros_like(original_image)
        color_mask[binary_mask == 1] = color

        # Наложение маски
        mask_overlay = cv2.addWeighted(mask_overlay, 1, color_mask, 1, 0)

        # Поиск контуров для центроида
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Находим наибольший контур
            largest_contour = max(contours, key=cv2.contourArea)

            # Вычисляем центроид
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                detected_teeth.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': conf,
                    'centroid': (cx, cy),
                    'color': color,
                    'contour': largest_contour
                })

    # Наложение прозрачной маски на оригинальное изображение
    result_image = cv2.addWeighted(original_image, 1 - alpha, mask_overlay, alpha, 0)

    return result_image, detected_teeth, class_names

def visualize_detection_results(result_image, detected_teeth, class_names, show_confidence=True):
    """
    Визуализация результатов детекции с подписями и контурами
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ## 1. Изображение с наложенными масками и подписями
    ax1.imshow(result_image)
    ax1.set_title('Сегментация зубов с цветовыми метками', fontsize=16, fontweight='bold')
    ax1.axis('off')

    # Добавление подписей классов
    for tooth in detected_teeth:
        cx, cy = tooth['centroid']
        class_name = tooth['class_name']
        confidence = tooth['confidence']
        color = tooth['color'] / 255.0  # нормализация для matplotlib

        # Подпись класса
        label = f"{class_name}"
        if show_confidence:
            label += f"\n({confidence:.2f})"

        # Рамка вокруг текста
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8,
                         edgecolor='white', linewidth=2)

        ax1.text(cx, cy, label, fontsize=10, fontweight='bold',
                color='white', ha='center', va='center', bbox=bbox_props)

    ## 2. Легенда с цветовыми кодами классов
    ax2.axis('off')
    ax2.set_title('Легенда классов зубов', fontsize=16, fontweight='bold')

    # Создание легенды
    unique_teeth = {}
    for tooth in detected_teeth:
        if tooth['class_name'] not in unique_teeth:
            unique_teeth[tooth['class_name']] = tooth['color'] / 255.0

    # Сортировка классов для удобства просмотра
    sorted_teeth = sorted(unique_teeth.items(), key=lambda x: x[0])

    # Отображение легенды
    legend_elements = []
    for class_name, color in sorted_teeth:
        legend_elements.append(patches.Patch(color=color, label=class_name))

    ax2.legend(handles=legend_elements, loc='center', fontsize=12,
              framealpha=0.9, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.show()

    return fig

def create_detailed_analysis_image(original_image, detected_teeth, alpha):
    """
    Создание детального изображения анализа с контурами и правильной нумерацией
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))

    ## 1. Оригинальное изображение
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Исходная ортопантомограмма', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    ## 2. Изображение с контурами: подписываем названия классов
    contour_image = original_image.copy()
    for tooth in detected_teeth:
        color = tooth['color']
        class_name = tooth['class_name']

        # Рисуем контур
        cv2.drawContours(contour_image, [tooth['contour']], -1, color.tolist(), 3)

        # Подписываем название класса зуба
        cx, cy = tooth['centroid']

        # Размер текста для правильного позиционирования
        text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

        # фон для текста
        cv2.rectangle(contour_image,
                     (cx - text_size[0]//2 - 5, cy - text_size[1]//2 - 5),
                     (cx + text_size[0]//2 + 5, cy + text_size[1]//2 + 5),
                     color.tolist(), -1)

        # текст с названием класса
        cv2.putText(contour_image, class_name,
                   (cx - text_size[0]//2, cy + text_size[1]//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    axes[0, 1].imshow(contour_image)
    axes[0, 1].set_title('Контуры зубов с названиями классов', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    ## 3. Изображение с цветными масками
    result_image = original_image.copy()
    mask_overlay = np.zeros_like(original_image)

    for tooth in detected_teeth:
        color = tooth['color']
        # Создаем временную маску для этого зуба
        temp_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(temp_mask, [tooth['contour']], 1)

        # Создаем цветную маску
        color_mask = np.zeros_like(original_image)
        color_mask[temp_mask == 1] = color

        # Накладываем маску
        mask_overlay = cv2.addWeighted(mask_overlay, 1, color_mask, 1, 0)

    # Наложение прозрачной маски
    result_image = cv2.addWeighted(original_image, 1 - alpha, mask_overlay, alpha, 0)

    axes[1, 0].imshow(result_image)
    axes[1, 0].set_title('Цветные маски сегментации', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # 4. Таблица обнаруженных зубов
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Обнаруженные зубы', fontsize=14, fontweight='bold')

    # Создание таблицы
    if detected_teeth:
        table_data = []
        # Сортируем зубы по классам для удобства чтения
        sorted_teeth = sorted(detected_teeth, key=lambda x: x['class_name'])

        for i, tooth in enumerate(sorted_teeth):
            table_data.append([
                i + 1,  # Порядковый номер в таблице
                tooth['class_name'],
                f"{tooth['confidence']:.3f}",
                f"({tooth['centroid'][0]}, {tooth['centroid'][1]})"
            ])

        table = axes[1, 1].table(
            cellText=table_data,
            colLabels=['№', 'Класс', 'Уверенность', 'Центроид'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Цветовое выделение строк таблицы
        for i, tooth in enumerate(sorted_teeth):
            color = tooth['color'] / 255.0
            # Применяем цвет к всей строке
            for j in range(4):
                table[(i+1, j)].set_facecolor(color)
                table[(i+1, j)].set_text_props(weight='bold')

    plt.tight_layout()
    plt.show()

    return fig

def save_annotation_image(result_image, detected_teeth, output_path):
    """
    Сохранение аннотированного изображения
    """
    # Конвертируем обратно в BGR для OpenCV
    output_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

    # Добавляем подписи на сохраненное изображение
    for tooth in detected_teeth:
        cx, cy = tooth['centroid']
        class_name = tooth['class_name']
        confidence = tooth['confidence']
        color = tooth['color'].tolist()

        # Рамка для текста
        text = f"{class_name} ({confidence:.2f})"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        # Фон для текста
        cv2.rectangle(output_image,
                     (cx - text_size[0]//2 - 5, cy - text_size[1]//2 - 5),
                     (cx + text_size[0]//2 + 5, cy + text_size[1]//2 + 5),
                     color, -1)

        # Текст
        cv2.putText(output_image, text, (cx - text_size[0]//2, cy + text_size[1]//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(output_path, output_image)
    print(f"Аннотированное изображение сохранено: {output_path}")

# Основная функция для запуска инференса
def analyze_orthopantomogram(
    image_path,
    model_path,
    conf_threshold,
    alpha,
    img_size,
    apply_crop_to_inference,
    crop_settings,
    output_path=None,
    show_detailed=True
    ):
    """
    Полный анализ ортопантомограммы
    """
    print(f"Анализ изображения: {image_path}")
    print("Загрузка модели...")
    print(f"Применение кропа при инференсе: {'ДА' if apply_crop_to_inference else 'НЕТ'}")
    if apply_crop_to_inference:
        print(f"Настройки кропа: слева {crop_settings['left_crop']*100:.1f}%, справа {crop_settings['right_crop']*100:.1f}%, сверху {crop_settings['top_crop']*100:.1f}%, снизу {crop_settings['bottom_crop']*100:.1f}%")

    # Выполняем инференс
    result_image, detected_teeth, class_names = inference_orthopantomogram(
        model_path, image_path, conf_threshold, alpha, img_size, apply_crop_to_inference, crop_settings
    )

    if result_image is None:
        return

    print(f"Обнаружено зубов: {len(detected_teeth)}")

    # Вывод статистики
    if detected_teeth:
        print("\nОбнаруженные зубы:")
        for tooth in detected_teeth:
            print(f"  - {tooth['class_name']}: уверенность {tooth['confidence']:.3f}")

        # Средняя уверенность
        avg_confidence = np.mean([t['confidence'] for t in detected_teeth])
        print(f"\nСредняя уверенность: {avg_confidence:.3f}")

    # Визуализация результатов
    fig1 = visualize_detection_results(result_image, detected_teeth, class_names)

    # Детальный анализ
    if show_detailed and detected_teeth:
        original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        fig2 = create_detailed_analysis_image(original_image, detected_teeth, alpha)

    # Сохранение результата
    if output_path and detected_teeth:
        save_annotation_image(result_image, detected_teeth, output_path)

    return result_image, detected_teeth

### Дополнительные утилиты

def batch_analyze_orthopantomograms(
    image_dir,
    model_path,
    output_dir,
    apply_crop_to_inference,
    alpha,
    crop_settings,
    img_size,
    conf_threshold,
    ):
    """
    Пакетный анализ нескольких ортопантомограмм
    """
    # Создание выходной директории
    os.makedirs(output_dir, exist_ok=True)

    # Более надежный поиск изображений
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG']
    image_files = []

    print(f"Поиск изображений в: {image_dir}")
    print(f"Применение кропа при инференсе: {'ДА' if apply_crop_to_inference else 'НЕТ'}")
    if apply_crop_to_inference:
        print(f"Настройки кропа: слева {crop_settings['left_crop']*100:.1f}%, справа {crop_settings['right_crop']*100:.1f}%, сверху {crop_settings['top_crop']*100:.1f}%, снизу {crop_settings['bottom_crop']*100:.1f}%")

    # Рекурсивный поиск файлов с нужными расширениями
    for file in os.listdir(image_dir):
        file_path = os.path.join(image_dir, file)
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file_path)

    # Альтернативный вариант поиска
    if not image_files:
        print("Попытка альтернативного поиска...")
        for ext in image_extensions:
            pattern = os.path.join(image_dir, f"*{ext}")
            image_files.extend(glob.glob(pattern))
            pattern = os.path.join(image_dir, f"*{ext.upper()}")
            image_files.extend(glob.glob(pattern))

    # Убираем дубликаты и сортируем
    image_files = sorted(list(set(image_files)))

    print(f"Найдено изображений: {len(image_files)}")

    if not image_files:
        print("! Изображения не найдены! Проверьте:")
        print(f"   - Путь: {image_dir}")
        print(f"   - Существование директории: {os.path.exists(image_dir)}")
        print(f"   - Содержимое директории: {os.listdir(image_dir) if os.path.exists(image_dir) else 'Директория не существует'}")
        return []

    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Обработка: {os.path.basename(image_path)}")

        try:
            # Создание пути для сохранения
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")

            # Анализ
            result_image, detected_teeth = analyze_orthopantomogram(
                image_path,
                model_path,
                conf_threshold,
                alpha,
                img_size,
                apply_crop_to_inference,
                crop_settings,
                output_path,
                show_detailed=False
            )

            results.append({
                'image_path': image_path,
                'detected_teeth': detected_teeth,
                'num_teeth': len(detected_teeth) if detected_teeth else 0
            })

        except Exception as e:
            print(f"! Ошибка при обработке {image_path}: {str(e)}")
            continue

    # Сводка по пакетной обработке
    print("\n" + "="*50)
    print("СВОДКА ПО ПАКЕТНОЙ ОБРАБОТКЕ")
    print("="*50)

    if results:
        for result in results:
            print(f"{os.path.basename(result['image_path'])}: {result['num_teeth']} зубов")

        total_teeth = sum(r['num_teeth'] for r in results)
        avg_teeth = total_teeth / len(results)
        print(f"\nВсего обнаружено зубов: {total_teeth}")
        print(f"Среднее количество зубов на изображение: {avg_teeth:.1f}")

        # Статистика по уверенности
        all_confidences = []
        for result in results:
            if result['detected_teeth']:
                all_confidences.extend([t['confidence'] for t in result['detected_teeth']])

        if all_confidences:
            print(f"Средняя уверенность детекции: {np.mean(all_confidences):.3f} ± {np.std(all_confidences):.3f}")
    else:
        print("Нет результатов для отображения")

    return results