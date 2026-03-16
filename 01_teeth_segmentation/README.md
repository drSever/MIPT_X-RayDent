# Сегментация зубов на ортопантомограммах
## Структура раздела
- `00_Dataset` - раздел с анализом используемых датасетов и аугментаций
- `01_YOLO` - раздел с обучением моделей на базе архитектуры YOLO
- `02_U-Net` - раздел с обучением моделей на базе архитектуры U-Net
- `03_U-Net_encoders` - раздел с обучением моделей на базе архитектуры U-Net с предобученными энкодерами
- `04_MaskRCNN` - раздел с обучением моделей на базе архитектуры MaskRCNN

# Обзор литературы

Процесс сегментации зубов является критически важным этапом в цифровой стоматологии, обеспечивающим основу для диагностики, планирования имплантации, ортодонтического лечения и идентификации личности (Hsu & Wang, 2021; Kanwal et al., 2023).

## История вопроса и этапы решения задачи
История методов сегментации стоматологических снимков прошла путь от простейших алгоритмов обработки изображений до сложных нейросетевых архитектур (Brahmi et al., 2024; Kanwal et al., 2023).

1.    **Традиционные методы (до 2018 года)**: В этот период доминировали методы, основанные на характеристиках яркости пикселей и априорных знаниях о форме. Около 54% исследований того времени использовали пороговую обработку (threshold-based) (Silva & Oliveira, 2018). К другим популярным подходам относились методы разделения по регионам, кластеризация (Fuzzy C-means), алгоритмы водораздела (watershed) и модели активных контуров («змеи» или snakes) (Silva & Oliveira, 2018). Большинство работ (80%) фокусировалось на интраоральных снимках, так как панорамные изображения (ОПТГ) считались слишком сложными из-за наложения структур челюсти, позвоночника и шумов (Silva & Oliveira, 2018).
2.    **Переломный момент (2018 год)**: Настоящий прорыв произошел с публикацией работ Silva et al. (2018), которые представили первый крупный открытый набор данных UFBA-UESC из 1500 аннотированных ОПТГ (Silva & Oliveira, 2018; Silva et al., 2022). Это позволило исследователям перейти к глубокому обучению (Deep Learning) (Silva et al., 2022).

## Основные архитектуры
В области сегментации зубов сформировался стандарт использования трех основных типов архитектур:
1.    **U-Net**: Эта архитектура стала «золотым стандартом» для медицинской визуализации благодаря симметричной структуре энкодера-декодера и обходным связям (skip connections), позволяющим сохранять детали границ (Helli & Hamamci, 2022; Kanwal et al., 2023). С помощью U-Net исследователи достигали коэффициента Дайса (Dice Score) на уровне 92,8% – 93,4% (Koch et al., 2019; Helli & Hamamci, 2022). Современные модификации, такие как Teeth U-Net, используют механизмы внимания для обработки нерегулярных форм зубов (Hou et al., 2023).
2.    **Mask R-CNN**: Двухстадийная архитектура, которая одновременно выполняет детектирование объектов и предсказание попиксельной маски. Jader et al. (2018) первыми применили её для сегментации экземпляров зубов, достигнув F1-score 88% и точности (Precision) 94% (Jader et al., 2018).
3.    **YOLO (You Only Look Once)**: Одностадийные модели которые ценятся за высокую скорость работы (Budagam et al., 2024). В задачах определения углов наклона третьих моляров модели YOLO продемонстрировали выдающийся результат — до 99% mAP (Vilcapoma et al., 2024).

## Методы оптимизации и функции потерь
Большинство моделей обучались с использованием оптимизаторов Adam или SGD (Hsu & Wang, 2021; Park et al., 2022). В качестве функций потерь применяются комбинации Binary Cross-Entropy (BCE) для точности на уровне пикселей и Dice Loss для решения проблемы дисбаланса классов (когда зубы занимают малую часть снимка) (Dhar & Deb, 2024; Ghafoor et al., 2023).

## Семантическая и инстанс-сегментация
Задача сегментации зубов решается в двух основных парадигмах:
1.    **Семантическая сегментация (Semantic Segmentation)**: Относит каждый пиксель к классу «зуб» или «фон» без разделения зубов между собой (Mashayekhi et al., 2023). Это полезно для общей оценки состояния полости рта, например, при поиске пародонтита, где модель Multi-Label U-Net показала Dice 0,96 (Widyaningrum et al., 2022). Однако такой подход не позволяет нумеровать зубы по системе FDI.
2.    **Сегментация экземпляров (Instance Segmentation)**: Более сложная задача, требующая идентификации каждого отдельного зуба как уникального объекта (Silva et al., 2022). Здесь критически важной метрикой является mAP (mean Average Precision). Современные модели сегментации экземпляров, такие как HTC (Hybrid Task Cascade), достигают значений mAP 0,821 на панорамных снимках (Silva et al., 2022).

## Современные архитектуры и актуальные результаты в сегментации зубов
На современном этапе фокус исследований сместился на использование механизмов внимания (attention mechanisms) и гибридных систем, которые позволяют более точно учитывать сложную морфологию и взаимное расположение зубов (Ghafoor et al., 2023; Mashayekhi et al., 2023).
1.    **Трансформеры (Transformers)**: Архитектуры вроде Swin Transformer и DETR произвели революцию в области, позволяя эффективно улавливать глобальные зависимости и контекст в зубном ряду, что недоступно классическим сверточным сетям (Almalki & Latecki, 2023; Carion et al., 2020; Liu et al., 2021). Например, специализированная модель Mask2Former с энкодером BEIT продемонстрировала показатель mIoU 90% при семантической сегментации сразу 33 различных признаков, охватывающих зубы, корни, кариес, реставрации, пульпиты и другие аномалии (Mashayekhi et al., 2023).
2.    **Гибридные модели «Detect-then-Segment»**: Системы нового поколения, такие как OralBBNet, объединяют высокую скорость локализации YOLOv8 и точность выделения границ, характерную для U-Net (Budagam et al., 2024). Такой подход позволяет поднять показатель Dice на 15–20% по сравнению со стандартными решениями за счет использования ограничивающих рамок (bounding boxes) как пространственного априорного знания, которое направляет процесс сегментации в сложных зонах (Budagam et al., 2024).
3.    **Сети Колмогорова-Арнольда (KAN)**: Новейший архитектурный подход DE-KAN использует нелинейные обучаемые функции активации на ребрах сети вместо весов на узлах, что значительно повышает обучающую способность и прозрачность модели («white-box» подход) (Mustakim et al., 2024). Применение этой технологии позволило достичь рекордного коэффициента Дайса 97,1% на наборе данных CDPR, успешно справляясь с проблемой перекрывающихся структур и нечетких границ (Mustakim et al., 2024).
4.    **Обучение с частичным привлечением учителя (Semi-Supervised Learning)**: В условиях дефицита размеченных медицинских данных модели, обучаемые на малых выборках (например, в рамках международного челленджа STS 2024), показывают значительный рост точности благодаря использованию псевдо-меток (pseudo-labels) и интеграции мощных базовых моделей, таких как SAM (Segment Anything Model) (Wang et al., 2024). Победители STS 2024 продемонстрировали, что использование SSL позволяет улучшить показатели сегментации более чем на 60 процентных пунктов по сравнению с классическим обучением с учителем при одинаковом (малом) объеме аннотированных данных (Wang et al., 2024).
Таким образом, современные системы ИИ способны сегментировать зубы с точностью, превышающей 90–95%, и обрабатывать снимки в 79 раз быстрее, чем врачи-люди, что радикально ускоряет диагностику и планирование лечения (Hsu et al., 2024; Mustakim et al., 2024).
 
## Архитектура YOLO
 
Использование архитектуры YOLO (You Only Look Once) в стоматологической радиологии, особенно для анализа панорамных снимков (ОПТГ), приобрело значительную популярность благодаря высокой скорости работы и эффективности в задачах локализации и классификации зубов (Budagam et al., 2024; Vilcapoma et al., 2024). Модели YOLO применяются как в виде самостоятельных систем для сегментации экземпляров, так и в качестве первого этапа («детектора») в гибридных многостадийных конвейерах (Hamamci et al., 2025; Oz et al., 2026).
 
Источники выделяют несколько ключевых сценариев использования различных версий YOLO:
- **Гибридная архитектура OralBBNet (YOLOv8 + U-Net)**: В этой модели YOLOv8 используется на первой стадии для обнаружения зубов и извлечения их ограничивающих рамок (bounding boxes) (Budagam et al., 2024). Полученные рамки служат «пространственными подсказками» для сегментатора U-Net, что позволяет модели фокусироваться на конкретных объектах, избегая ошибок в зашумленных областях челюсти (Budagam et al., 2024). Такой подход «Detect-then-Segment» позволяет поднять показатель Dice за счет использования ограничивающих рамок (Budagam et al., 2024).
- **Специализированная модель YOLOrtho (модифицированная YOLOv8)**: Команда ChohoTech в рамках челенджа DENTEX адаптировала YOLOv8, добавив координатные свертки (coordinate convolutions) для лучшего понимания пространственного расположения зубов и четыре независимых классификационных головы для диагностики патологий (Hamamci et al., 2025). Модель была модифицирована для обработки частичных меток и объединения рамок для зубов с множественными заболеваниями (Hamamci et al., 2025).
- **YOLOv2 для оценки углов третьих моляров**: В сравнительном исследовании с Faster R-CNN и SSD модель YOLOv2 продемонстрировала высокую эффективность и была выбрана для классификации углов наклона зубов мудрости по критерию Винтера (Vilcapoma et al., 2024). Исследование показало, что YOLOv2 достигает до 99% mAP при определении дистоангулярного, вертикального, мезиоангулярного и горизонтального положений (Vilcapoma et al., 2024).
- **YOLOv5 для педиатрии**: Существуют решения на базе YOLOv5, специально обученные для работы с молочным и сменным прикусом у детей, где зубы имеют малый размер и часто перекрываются (Beser et al., 2024; Budagam et al., 2024). Модель способна автоматически обнаруживать, сегментировать и нумеровать зубы у педиатрических пациентов (Budagam et al., 2024).
- **YOLOv11 в сравнительных тестах**: Одна из последних версий YOLOv11 тестировалась в качестве базовой модели на наборе данных AKUDENTAL для оценки точности определения границ реставраций (имплантатов, мостов) и зубов (Oz et al., 2026). Испытания показали, что YOLOv11 обеспечивает высокую скорость и точность, сопоставимую с двухстадийными архитектурами при локализации объектов (Oz et al., 2026).

Модели YOLO демонстрируют высокие показатели, часто превосходя классические двухстадийные детекторы, такие как Mask R-CNN (Budagam et al., 2024; Vilcapoma et al., 2024).
- Точность детектирования (mAP):
В задаче определения углов зубов мудрости YOLOv2 в комбинации с ResNet-101 достигла впечатляющей точности в 99% (mean Average Precision, mAP), показав лучшие результаты среди конкурентов (Vilcapoma et al., 2024).
В рамках челенджа STS 2024 модель на базе YOLOv8 от команды ChohoTech достигла показателя Instance Affinity (IA) 88,53%, что стало лучшим результатом среди полуучительских методов сегментации (Wang et al., 2024).
Для обнаружения всех классов зубов YOLOv8 показала mAP 74,9% и AP50 94,6% на наборе данных UFBA-425 (Budagam et al., 2024).
- Качество сегментации (Dice Score):
Гибридная модель OralBBNet обеспечила улучшение коэффициента Дайса на 15–20% по сравнению со стандартным U-Net и на 2–4% по сравнению с другими SOTA-архитектурами сегментации (Budagam et al., 2024).
- Скорость и производительность: YOLOv2 продемонстрировала самое быстрое время вывода — от 0,071 до 0,092 секунды на изображение, что делает её идеальной для использования в реальном времени в клиниках (Vilcapoma et al., 2024).

### Используемые методы аугментации 
 
Для повышения обобщающей способности моделей и борьбы с нехваткой данных (особенно по редким патологиям или молочным зубам) исследователи применяли широкий спектр аугментаций (Almalki & Latecki, 2023; Kadi et al., 2024):
- Геометрические трансформации:
    - Случайное кадрирование (Random Cropping): в диапазоне от 0% до 20% площади снимка (Budagam et al., 2024).
    - Вращение (Rotation): случайные повороты в диапазоне +/- 10 градусов (Kadi et al., 2024; Vilcapoma et al., 2024).
    - Отражение (Flipping): горизонтальное (с корректным переименованием номеров зубов по FDI) и вертикальное (Almalki & Latecki, 2023; Helli & Hamamci, 2022).
- Яркостные и контрастные изменения:
    - Регулировка яркости (Brightness) и контрастности (Contrast) (Hsu & Wang, 2021; Budagam et al., 2024).
    - Изменение насыщенности (Saturation) и применение пороговых фильтров (Vilcapoma et al., 2024).
    - Гистограммная эквализация (CLAHE) для улучшения видимости деталей корней (Budagam et al., 2024).
- Шумы и искажения:
    - Добавление шума типа «соль и перец» (Salt and Pepper noise) (Helli & Hamamci, 2022).
    - Гауссово размытие (Gaussian Blur) для имитации низкого качества снимков (Hsu & Wang, 2021).
    - Эластичные деформации (Elastic Transformations) для имитации анатомической вариативности (Ahn et al., 2026; Hsu & Wang, 2021).
- Продвинутые методы:
    - Mosaic-аугментация: объединение нескольких снимков в один обучающий пример для улучшения детектирования мелких объектов (Wang et al., 2023).
    - Inpainting (дорисовка): использование генеративных моделей (например, Stable Diffusion 2) для удаления или добавления зубов на снимки, что позволило повысить mAP на 0,07 (Kadi et al., 2024).

### Выводы
YOLO является мощным инструментом для анализа ОПТГ, обеспечивающим баланс между точностью сегментации (Dice ~90%) и скоростью обработки (менее 100 мс). Однако эффективность модели напрямую зависит от качества пространственных подсказок и интенсивности аугментации данных при обучении (Budagam et al., 2024; Vilcapoma et al., 2024).
 
## Архитектура U-Net
Архитектура U-Net признана «золотым стандартом» в области сегментации медицинских изображений, включая стоматологические панорамные снимки (ОПТГ), благодаря своей симметричной структуре энкодера-декодера и обходным связям (skip connections) (Ronneberger et al., 2015). Эти связи позволяют объединять высокоуровневые семантические признаки с низкоуровневыми деталями границ, что критически важно для точного выделения контуров зубов (Widyaningrum & Candradewi, 2022).
 
В исследованиях ОПТГ применялись как классические, так и глубоко модифицированные версии архитектуры:
- **Классическая U-Net**: Использовалась для прямой семантической сегментации, разделяя изображение на классы «зуб» и «фон» (Koch et al., 2019).
- **Multi-Label U-Net**: Применялась для детекции и стадирования пародонтита, где модель классифицировала пиксели на пять уровней тяжести заболевания на основе потери костной ткани (Widyaningrum & Candradewi, 2022).
- **Teeth U-Net**: Специализированная версия, включающая блоки Squeeze-and-Excitation (SE) для калибровки каналов и механизмы внимания для обработки нерегулярных форм зубов и компенсации низкого контраста (Hou et al., 2023).
- **Гибридные и двухстадийные модели**:
    - TSASNet (Two-Stage Attention Segmentation Network): Использует механизм внимания на первой стадии для грубой локализации области зубов и U-Net на второй стадии для уточнения маски на пиксельном уровне (Zhao et al., 2020).
    - OralBBNet: Объединяет детектор YOLOv8 и сегментатор U-Net. Ограничивающие рамки от YOLOv8 подаются в U-Net как «пространственные подсказки» через специальные слои BB-Convolution, что предотвращает слияние масок соседних зубов (Budagam et al., 2024).
    - U-Net с морфологической обработкой: Для решения задачи сегментации экземпляров (разделения соприкасающихся зубов) предсказания U-Net подвергались постобработке алгоритмами водораздела (watershed) и фильтрации в OpenCV (Helli & Hamamci, 2022).
- **Продвинутые архитектуры**: Исследовались варианты U-KAN (сочетание U-Net с сетями Колмогорова-Арнольда) (Mustakim et al., 2024) и CTA-UNet (параллельная архитектура CNN-Transformer) для захвата глобальных зависимостей в зубном ряду (Chen et al., 2023).

Модели на базе U-Net демонстрируют одни из самых высоких показателей точности в литературе:
- Коэффициент Дайса (Dice Score):
Базовые модели достигают значений 92,8% – 93,4% (Koch et al., 2019).
Модифицированная модель S-R2F2U-Net показала результат 93,26% (Dhar & Deb, 2024).
Использование постобработки и искусственного разделения зубов на масках (split mask) позволило достичь 95,4% (Helli & Hamamci, 2022).
Новейшая модель DE-KAN достигла рекордного показателя 97,1% на наборе данных CDPR (Mustakim et al., 2024).
- IoU (Intersection over Union): Для семантической сегментации зубов показатели варьируются от 91,1% до 94,5% (Mustakim et al., 2024).

### Используемые методы аугментации
 
Для повышения обобщающей способности моделей применялись методы случайного изменения яркости, контрастности, аффинные и эластичные преобразования, а также размытие по Гауссу (Lee et al., 2020). Часто использовалось горизонтальное отражение, требующее пересчета номеров зубов согласно системе FDI (Pinheiro et al., 2021). Также внедрялись специализированные методы, такие как Uniform Distributed Augmentation для балансировки редких классов (Mashayekhi et al., 2023) и аугментация на основе преобразования Фурье (FTA) (Wang et al., 2024).
 
Для обучения U-Net на ограниченных стоматологических данных, таких как набор UFBA-UESC (Silva et al., 2018; Jader et al., 2018), применялся агрессивный набор аугментаций для повышения обобщающей способности моделей и компенсации малого объема выборок:
1.    **Геометрические**: Горизонтальное отражение признано наиболее естественным методом из-за горизонтальной симметрии челюсти (Helli & Hamamci, 2022; Mustakim et al., 2024). Также активно использовались случайное вращение (обычно в диапазоне +/- 10–20 градусов) (Kadi & Bendjama, 2024), случайное кадрирование (cropping) и масштабирование (Budagam et al., 2024; Chen et al., 2023).
2.    **Эластичные деформации (Elastic Deformations)**: Считаются одними из самых эффективных для медицинских снимков, так как имитируют естественную вариативность формы органов и биологических тканей (Hsu & Wang, 2021; Raith et al., 2025).
3.    **Яркостные**: Применялось случайное изменение яркости, контрастности и насыщенности для адаптации модели к снимкам, полученным с разных рентгеновских аппаратов и при различных условиях экспозиции (Chen Jiayi, 2025; Vilcapoma et al., 2024).
4.    **Шумовые**: Для повышения устойчивости к низкому качеству изображений внедрялось добавление гауссова шума, шума типа «соль и перец» и гауссова размытия (blurring) (Helli & Hamamci, 2022; Lee et al., 2020).
 
### Выводы
Использование U-Net в задачах ОПТГ эволюционировало от простых классификаторов пикселей к сложным гибридным системам. За счет использования пространственных подсказок и механизмов внимания такие модели достигают точности сегментации выше 95% по Dice (Helli & Hamamci, 2022; Mustakim et al., 2024), превосходя возможности чисто детекторных моделей (Budagam et al., 2024).
 
## Архитектура MaskRCNN
Архитектура Mask R-CNN является одной из наиболее значимых в истории сегментации зубов, так как именно на её основе были предприняты первые успешные попытки сегментации экземпляров (instance segmentation) на панорамных снимках (ОПТГ) (Jader et al., 2018). В отличие от семантической сегментации, Mask R-CNN позволяет не просто выделить область зубов, но и идентифицировать каждый зуб как отдельный объект (Widyaningrum & Candradewi, 2022).
 
Исследователи применяли Mask R-CNN в различных модификациях и сценариях:
- **Выбор backbone**: Чаще всего использовались семейства ResNet. Было установлено, что ResNet-50 часто является оптимальным выбором для ОПТГ, так как обеспечивает высокую точность (IoU 75,14%) и при этом менее склонен к переобучению на специфических данных, чем более глубокие ResNet-101 или ResNet-152 (Wang et al., 2023). Также тестировались современные варианты с Swin Transformer в качестве бэкенда для задач нумерации зубов (Almalki & Latecki, 2023).
- **Гибридные системы**: Mask R-CNN интегрировали в многостадийные конвейеры. Например, в системе DeepOPG модель использовалась для локализации зубов после предварительной функциональной сегментации (Hsu & Wang, 2021).
- **Уточнение границ**: Чтобы справиться с «зубчатыми» краями масок, возникающими из-за низкого разрешения стандартной головы Mask R-CNN (28x28), исследователи добавляли модуль PointRend, который итеративно уточняет детали границ, что особенно эффективно для зубов сложной формы (Pinheiro et al., 2021; Silva et al., 2022).
- **Специфические задачи**: Архитектуру успешно применяли не только для здоровых зубов, но и для сегментации корней при оценке исходов эндодонтического лечения (Dennis et al., 2024), а также для автоматического заполнения зубных карт (charting), включая идентификацию пломб, коронок и имплантатов (Oz et al., 2026; Vinayahalingam et al., 2021).

Mask R-CNN демонстрирует стабильно высокие показатели, хотя в некоторых тестах может уступать специализированным или более легким моделям в скорости:
- F1-score и точность: В новаторской работе Jader et al. (2018) был достигнут F1-score 88% и точность (Precision) 94% на наборе данных из 1224 изображений. В других исследованиях фиксировался F1-score на уровне 87,5% (Lee et al., 2020).
- Коэффициент Дайса (Dice): Модель показывает результаты в диапазоне 87% – 92,78% (Rubiu et al., 2023). При этом отмечается, что семантические модели (например, U-Net) могут давать более высокий Dice (до 0,96), но они не способны разделять отдельные зубы (Widyaningrum & Candradewi, 2022).
- mAP (mean Average Precision): На современных бенчмарках, таких как OdontoAI или AKUDENTAL, Mask R-CNN достигает значений mAP 0,71 – 0,749 (Silva et al., 2022; Oz et al., 2026).
- Сравнение с YOLO: Исследования на датасете AKUDENTAL показали, что Mask R-CNN и YOLOv11 имеют схожий mAP при детектировании, однако двухстадийная архитектура Mask R-CNN обеспечивает значительно большую точность в прорисовке границ объектов (Oz et al., 2026).

### Использованные аугментации
Для обучения на специфических стоматологических данных применялся широкий набор методов искусственного расширения выборки для повышения обобщающей способности моделей и компенсации ограниченного объема обучающих наборов (Park et al., 2022).
- Геометрические: Случайное вращение, обычно в пределах +/- 5–10° (Kadi & Bendjama, 2024), горизонтальное отражение с обязательной коррекцией номеров зубов по системе FDI (Almalki & Latecki, 2023; Silva et al., 2023), а также масштабирование, сдвиг (shifting) и аффинные трансформации (Budagam et al., 2024; Hsu & Wang, 2021).
- Эластичные деформации (Elastic Transformations): Данный метод признан критически важным для имитации естественной вариативности анатомии челюсти и биологических тканей на рентгенограммах (Hsu & Wang, 2021; Raith et al., 2025).
- Яркость и контраст: Регулировка яркости, контрастности и насыщенности используется для адаптации моделей к снимкам с разным уровнем экспозиции и характеристиками оборудования (Chen, 2025; Hsu & Wang, 2021).
- Шум и размытие: Добавление гауссова шума, шума «соль и перец» и гауссова размытия применяется для повышения устойчивости моделей к низкому качеству исходных рентгенограмм (Helli & Hamamci, 2022; Hsu & Wang, 2021).

### Трансферное обучение 
Использование весов, предобученных на универсальных наборах данных ImageNet и COCO, является стандартом для инициализации Mask R-CNN в стоматологических задачах, что позволяет эффективно обучать модели даже на относительно небольших выборках (Dennis et al., 2024; Helli & Hamamci, 2022; Widyaningrum & Candradewi, 2022).

### Выводы
Таким образом, Mask R-CNN остается фундаментальным инструментом для задач, где требуется высокая детализация границ и разделение зубов на отдельные экземпляры. Его двухстадийная архитектура обеспечивает значительно большую точность прорисовки контуров объектов по сравнению с современными более быстрыми одностадийными архитектурами, такими как YOLO (Hamamci et al., 2025; Oz et al., 2026).
 
## Архитектура SAM
Архитектура Segment Anything Model (SAM) в сочетании с детекторами реального времени (такими как YOLOv8) произвела революцию в сегментации зубов, особенно в условиях острой нехватки аннотированных экспертами данных (Wang et al., 2024). Благодаря своим мощным возможностям предварительного обучения на гигантских массивах данных, SAM выступает в роли «базовой модели» (foundation model), способной генерировать высокоточные маски даже для сложных анатомических структур (Wang et al., 2024).
 
Роль архитектуры SAM в сегментации зубов:
- **Связка YOLOv8 + SAM**: Стратегия «Детектируй, затем сегментируй».   
Популярная двухстадийная стратегия объединяет скорость детектора YOLOv8 и точность SAM (Wang et al., 2024). Этот конвейер решает задачу инстанс-сегментации:
Локализация (YOLOv8): На первом этапе модель YOLOv8 (например, версия YOLOv8x) быстро находит каждый зуб и создает вокруг него ограничивающую рамку (bounding box) (Wang et al., 2024).
- **Сегментация (SAM)**: Координаты этих рамок передаются в SAM в качестве визуальных подсказок (prompts) (Wang et al., 2024). Используя свою чувствительность к краям (edge sensitivity), SAM строит прецизионную пиксельную маску внутри рамки, отделяя зуб от костной ткани и десен (Wang et al., 2024).
- **Решение проблемы перекрытия**: Традиционные сверточные сети (CNN) часто ошибочно объединяют маски тесно расположенных или перекрывающихся зубов (Wang et al., 2024). Использование рамок YOLO в качестве априорного знания заставляет SAM обрабатывать каждый объект как отдельный «инстанс», исключая их склеивание (Wang et al., 2024).
 
Применение архитектур на базе SAM позволило достичь показателей, ранее недоступных при малых объемах данных:
- Точность сегментации: Использование вариантов SAM позволило командам достичь индекса Dice выше 90% при сегментации зубов на КЛКТ (3D), используя всего 9% размеченных данных (Wang et al., 2024).
- Экономия времени: В исследовании взаимодействия «человек-машина» было доказано, что использование высокоточных предсказаний моделей сокращает время ручной разметки одного панорамного снимка (ОПТГ) взрослым пациентом на 88.2%, а объемов КЛКТ — на 94.1% (Wang et al., 2024). Это превращает многочасовую работу врача в быструю процедуру проверки и коррекции (Wang et al., 2024).

### Выводы
Таким образом, архитектуры на базе SAM в сочетании с современными детекторами, такими как YOLOv8, являются наиболее перспективным направлением для автоматизации анализа стоматологических изображений. Они не только обеспечивают высокую точность (Dice > 90%), сравнимую с экспертной, но и делают передовые технологии доступными даже для исследовательских групп с ограниченными ресурсами на разметку данных.

## Использованная литература 
- Almalki, A., & Latecki, L. J. (2023). Self-supervised learning with masked image modeling for teeth numbering, detection of dental restorations, and instance segmentation in dental panoramic radiographs. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) (pp. 5594–5603). 
 
- Ahn, S., Kim, M., Kim, J., & Park, W. (2026). Application of deep learning in evaluating the anatomical relationship between the mandibular third molar and inferior alveolar nerve: A scoping review. Medicina Oral, Patología Oral y Cirugía Bucal, 31(1), e95–e103. 
 
- Beser, B., Reis, T., Berber, M. N., Topaloglu, E., Gungor, E., Kılıc, M. C., Duman, S., Çelik, Ö., Kuran, A., & Bayrakdar, I. S. (2024). YOLO-v5 based deep learning approach for tooth detection and segmentation on pediatric panoramic radiographs in mixed dentition. BMC Medical Imaging, 24(1), 172. 
 
- Brahmi, W., Jdey, I., & Drira, F. (2024). Exploring the role of convolutional neural networks (CNN) in dental radiography segmentation: A comprehensive systematic literature review. Multimedia Tools and Applications, 83, 55565–55585. 
 
- Budagam, D., Imanbayev, A. Z., Akhmetov, I. R., Sinitca, A., Antonov, S., & Kaplun, D. (2024). OralBBNet: Spatially guided dental segmentation of panoramic X-rays with bounding box priors. arXiv preprint arXiv:2401.09190. 
 
- Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-end object detection with transformers. In European Conference on Computer Vision (ECCV) (pp. 213–229). Springer. 

- Ghafoor, A., Moon, S. Y., & Lee, B. (2023). Multiclass segmentation using teeth attention modules for dental X-ray images. IEEE Access, 11, 130832–130846. 
 
- Chen, Z., Chen, S., & Hu, F. (2023). CTA-UNet: CNN-transformer architecture UNet for dental CBCT images segmentation. Physics in Medicine & Biology, 68(17), 175042. 
 
- Chen, J. (2025). Convolutional neural network for maxillary sinus segmentation based on the U-Net architecture at different planes in the Chinese population: A semantic segmentation study. BMC Oral Health, 25, 961. 
 
- Dennis, M., et al. (2024). Development and evaluation of a deep learning segmentation model for assessing non-surgical endodontic treatment outcomes on periapical radiographs: A retrospective study. PLOS ONE, 19(12), e0311235. 
 
- Dhar, M. K., & Deb, M. (2024). S-R2F2U-Net: A single-stage model for teeth segmentation. International Journal of Biomedical Engineering and Technology, 46(1), 81–100. 
 
- Ghafoor, A., Moon, S. Y., & Lee, B. (2023). Multiclass segmentation using teeth attention modules for dental X-ray images. IEEE Access, 11, 130832–130846. 
 
- Hamamci, I. E., Er, S., Durugol, O. F., Cakmak, G. R., de la Rosa, E., Simsar, E., Yuksel, A. E., Gultekin, S., Ozdemir, S. D., Yang, K., ... & Menze, B. (2025). DENTEX: Dental enumeration and tooth pathosis detection benchmark for panoramic X-rays. arXiv preprint arXiv:2305.19112v4. 
 
- Helli, S. S., & Hamamci, A. (2022). Tooth instance segmentation on panoramic dental radiographs using U-Nets and morphological processing. Düzce University Journal of Science & Technology, 10(1), 39–50. 
 
- Hou, S., Zhou, T., Liu, Y., Dang, P., Lu, H., & Shi, H. (2023). Teeth U-Net: A segmentation model of dental panoramic X-ray images for context semantics and contrast enhancement. Computers in Biology and Medicine, 152, 106296. 
 
- Hsu, T.-M. H., & Wang, Y.-C. C. (2021). DeepOPG: Improving orthopantomogram finding summarization with weak supervision. arXiv preprint arXiv:2103.08290. 
 
- Hsu, T.-M. H., Wang, Y.-C. C., et al. (2024). Artificial intelligence to assess dental findings from panoramic radiographs – A multinational study. Radiology (Preprint). 
 
- Jader, G., Fontineli, J., Ruiz, M., Abdalla, K., Pithon, M., & Oliveira, L. (2018). Deep instance segmentation of teeth in panoramic x-ray images. In 2018 31st SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI) (pp. 400–407). IEEE. 
 
- Kadi, H., Sourget, T., Kawczynski, M., Bendjama, S., Grollemund, B., & Bloch-Zupan, A. (2024). Detection transformer for teeth detection, segmentation, and numbering in oral rare diseases: Focus on data augmentation and inpainting techniques. arXiv preprint arXiv:2402.04408. 
 
- Kanwal, M., Rehman, M. M. U., Farooq, M. U., & Chae, D.-K. (2023). Mask-transformer-based networks for teeth segmentation in panoramic radiographs. Bioengineering, 10(7), 843. 
 
- Koch, T. L., Perslev, M., Igel, C., & Brandt, S. S. (2019). Accurate segmentation of dental panoramic radiographs with u-nets. In 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019) (pp. 15–19). IEEE. 
 
- Lee, J.-H., Han, S.-S., Kim, Y. H., Lee, C., & Kim, I. (2020). Application of a fully deep convolutional neural network to the automation of tooth segmentation on panoramic radiographs. Oral Surgery, Oral Medicine, Oral Pathology and Oral Radiology, 129(6), 635–642. 
 
- Liu, Z., Lin, Y., Cao, Y., Han, Hu., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin Transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (pp. 10012–10022). 
 
- Mashayekhi, M., Majd, S. A., Amiramjadi, A., & Mashayekhi, B. (2023). Radious: Unveiling the enigma of dental radiology with BEIT adaptor and Mask2Former in semantic segmentation. arXiv preprint arXiv:2305.06236. 
 
- Mustakim, M. M. R., Li, J., Bhuiyan, S., Hasan, M. M., & Han, B. (2024). DE-KAN: A Kolmogorov Arnold Network with Dual Encoder for accurate 2D Teeth Segmentation. International Journal of Imaging Systems and TechnologyWang, Y., Li, Z., Wu, C., Liu, J., Zhang, Y., Ni, J., ... & Zhou, H. (2024). MICCAI STS 2024 Challenge: Semi-supervised instance-level tooth segmentation in panoramic X-ray and CBCT images. Medical Image Analysis, 98, 103322. 
 
- Oz, M., Sengul, A., Hatipoglu, M., & Danisman, T. (2026). AKUDENTAL teeth instance segmentation dataset: a cross-dataset analysis. BMC Oral Health, 26(1), 247. 
 
- Park, S., Kim, H., Shim, E., Hwang, B. Y., Kim, Y., Lee, J. W., & Seo, H. (2022). Deep learning-based automatic segmentation of mandible and maxilla in multi-center CT images. Applied Sciences, 12(3), 1358. 
 
- Park, E. Y., Cho, H., & Kim, E. K. (2022). Caries detection with tooth surface segmentation on intraoral photographic images using deep learning. BMC Oral Health, 22, 573. 
 
- Pinheiro, L., Silva, B., Sobrinho, B., Lima, F., Cury, P., & Oliveira, L. (2021). Numbering permanent and deciduous teeth via deep instance segmentation in panoramic x-rays. Symposium on Medical Information Processing and Analysis (SIPAIM), 12088, 95–104. 
 
- Raith, S., Pankert, T., Jaganathan, S., Pankert, K., Lee, H., Peters, F., Hölzle, F., & Modabber, A. (2025). Segmenting beyond the imaging data: Creation of anatomically valid edentulous mandibular geometries for surgical planning using artificial intelligence. Clinical Oral Investigations, 29, 501. 
 
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, 234–241. 
 
- Rubiu, G., et al. (2023). Teeth segmentation in panoramic dental x-ray using mask regional convolutional neural network. Applied Sciences, 13(14), 7947. 
 
- Silva, B., Pinheiro, L., Pinheiro, B., Lima, F., Sobrinho, B., Abdalla, K., Pithon, M., Cury, P., & Oliveira, L. (2023). Boosting research on dental panoramic radiographs: A challenging data set, baselines, and a task central online platform for benchmark. Computer Methods in Biomechanics and Biomedical Engineering: Imaging & Visualization, 11(4), 1327–1347. 
 
- Silva, G., & Oliveira, L. (2018). Automatic segmenting teeth in X-ray images: Trends, a novel data set, benchmarking and future perspectives. Expert Systems with Applications, 107, 15–31. 
 
- Silva, B. P. M., et al. (2022). Boosting research on dental panoramic radiographs: A challenging data set, baselines, and a task central online platform for benchmark. Computers in Biology and Medicine, 11(4), 1327–1347. 
 
- Singh, P., & Sehgal, P. (2020). Numbering and classification of panoramic dental images using 6-layer convolutional neural network. Pattern Recognition and Image Analysis, 30, 125–133. 
 
- Vilcapoma, P., Meléndez, D. P., Fernández, A., Vásconez, I. N., Hillmann, N. C., Gatica, G., & Vásconez, J. P. (2024). Comparison of Faster R-CNN, YOLO, and SSD for third molar angle detection in dental panoramic X-rays. Sensors, 24(18), 6053. 
 
- Vinayahalingam, S., et al. (2021). Automated chart filing on panoramic radiographs using deep learning. Journal of Dentistry, 115, 103864. 
 
- Wang, Y., Li, Z., Wu, C., Liu, J., Zhang, Y., Ni, J., ... & Zhou, H. (2024). MICCAI STS 2024 Challenge: Semi-supervised instance-level tooth segmentation in panoramic X-ray and CBCT images. Medical Image Analysis, 98, 103322. 
 
- Wang, Y., Zhang, Y., Chen, X., Wang, S., Qian, D., Ye, F., ... & Zhou, H. (2023). STS MICCAI 2023 Challenge: Grand challenge on 2D and 3D semi-supervised tooth segmentation. Medical Image Analysis. 
 
- Wang, Y., Ye, F., Chen, Y., Wang, C., Wu, C., Xu, F., Ma, Z., Liu, Y., Zhang, Y., Cao, M., & Chen, X. (2025). A multi-modal dental dataset for semi-supervised deep learning image segmentation. Scientific Data, 12, 117. 
 
- Wang, X., et al. (2023). ViSTooth: A visualization framework for tooth segmentation on panoramic radiograph. Proceedings of the IEEE Visualization and Visual Analytics (VIS).

- Widyaningrum, R., & Candradewi, I. (2022). Comparison of Multi-Label U-Net and Mask R-CNN for panoramic radiograph segmentation to detect periodontitis. Imaging Science in Dentistry, 52(4), 383–391. 
 
- Zhao, Y., Li, P., Gao, C., Liu, Y., Chen, Q., Yang, F., & Meng, D. (2020). TSASNet: Tooth segmentation on dental panoramic X-ray images by two-stage attention segmentation network. Knowledge-Based Systems, 206, 106338. 
 
- Zhu, S., Hu, M., Pan, T., Hong, Y., Li, B., Zhou, Z., & Xu, T. (2024). ViSTooth: A visualization framework for tooth segmentation on panoramic radiograph. 
