# Сегментация зубоврадикулярных кист на ортопантомограммах
## Структура раздела
- `` - раздел 
- `` - раздел 
- `` - раздел 
- `` - раздел 
- `` - раздел 

# Обзор литературы

## История вопроса и актуальность
 
**Радикулярная киста** (также называемая периапикальной) является наиболее распространенным типом кист челюстно-лицевой области и возникает как следствие хронического воспаления, вызванного кариесом, травмой или некорректным лечением корневых каналов (Rašić et al., 2024b). ОПТГ остается основным инструментом первичной диагностики благодаря своей доступности, низкой стоимости и широкому охвату анатомических структур (Rašić et al., 2023; Kaygısız et al., 2025).
Традиционный ручной анализ ОПТГ стоматологами сопряжен с высокой субъективностью, утомляемостью и риском диагностических ошибок, особенно при поиске мелких очагов или на фоне наложения сложных анатомических теней (Mashayekhi et al., 2023; Rašić et al., 2023). Внедрение систем автоматизированного анализа на базе глубокого обучения (Deep Learning) призвано минимизировать человеческий фактор и обеспечить раннее обнаружение патологий (Brahmbhatt & Shah, 2026).
 
## Использованные архитектуры и модели

В исследованиях последних лет наблюдается переход от классических сверточных нейронных сетей (CNN) к более сложным гибридным и трансформерным архитектурам:
- Семейство YOLO (You Only Look Once):
    - YOLOv8: Эта модель способна одновременно выполнять детекцию (ограничивающие рамки) и сегментацию масок. Она использует подход «без анкоров/якорей» (anchor-free), что упрощает обучение и повышает точность локализации (Rašić et al., 2023).
    - YOLOv11: Одна из современных версий (представлена в сентябре 2024 г.), в которую включены модули C2PSA (Cross-Stage Partial with Self-Attention) и C3k2. Это позволило модели лучше извлекать признаки мелких объектов и эффективнее работать в анатомически сложных зонах (Kaygısız et al., 2025).
- Архитектуры на базе трансформеров:
    - Radious (BEIT-Adapter + Mask2Former): Использование трансформеров (Vision Transformers, ViT) позволяет модели улавливать глобальный контекст изображения, который часто теряется в обычных CNN. Mask2Former выступает в роли универсального декодера, использующего маскированное внимание для точного выделения границ патологии (Mashayekhi et al., 2023).
- Специализированные модификации U-Net:
    - MARes-Net (Multi-scale Attention Residual Network): Модель, объединяющая остаточные связи (для предотвращения затухания градиентов) и механизмы внимания (Attention Gates). Модуль SFEM (Scale-aware Feature Extraction) расширяет поле восприятия сети, позволяя захватывать кисты различных размеров (Ding et al., 2024).
    - nnU-Net: Самоадаптирующаяся версия U-Net, показавшая высокую эффективность в задачах медицинской сегментации (Zhu et al., 2022).
- Другие модели: DenseNet121-CBAM, усиленная модулем внимания к каналам и пространству, показала высокую точность в классификации шести типов патологий челюсти (Brahmbhatt & Shah, 2026). Для оценки состояния гайморовых пазух успешно применялась модель EfficientDet-D4 (Ha et al., 2023).

## Методы аугментации данных

Для преодоления проблемы ограниченности и дисбаланса медицинских выборок исследователи применяют широкий спектр методов аугментации:
- Базовые методы: Вращение (обычно ±5–10°), горизонтальное отражение, масштабирование, сдвиг и изменение экспозиции/яркости (Kaygısız et al., 2025; Rašić et al., 2024b).
- Продвинутые методы:
    - Mosaic: Объединение четырех разных снимков в один тренировочный образец, что помогает модели лучше обобщать признаки в разном контексте (Rašić et al., 2023).
    - Uniform Distributed Augmentation: Специальная стратегия, при которой количество аугментированных копий для редких классов (например, определенных типов кист) увеличивается сильнее, чем для частых, для достижения равномерного распределения данных (Mashayekhi et al., 2023).
    - CLAHE (Contrast Limited Adaptive Histogram Equalization): Улучшение локального контраста для выделения нечетких границ кист на фоне костной ткани (Kaygısız et al., 2025; Ding et al., 2024).
    - Cut-and-pasting: Вырезание области патологии и вставка ее на снимки здоровых пациентов для синтетического увеличения обучающей выборки (Yu et al., 2022).

## Сложности
Сложности в области пазух: Основной проблемой остается верхняя челюсть, где наложение скуловой кости, твердого неба и гайморовых пазух создает «шум», мешающий ИИ. Карты EigenCAM выявили наличие «холодных зон» (неуверенности) алгоритмов именно в области пазух, где патологии часто путают с естественными полостями или ретенционными псевдокистами (Rašić et al., 2024b; Ha et al., 2023).
 
 
## Наиболее эффективные решения

Наилучшие показатели метрик в задачах анализа стоматологических изображений (сегментация, детекция и классификация) продемонстрировали следующие архитектуры и модели:

1. Сегментация (Semantic and Instance Segmentation)
В задачах точного выделения границ патологий наивысшие результаты показали:
    - Radious (BEIT-Adaptor + Mask2Former): Эта система достигла показателя mIoU (Mean Intersection over Union) 90% и средней точности (mAcc) 65% (Mashayekhi et al., 2023). Она значительно превзошла такие модели, как DeepLabv3+ (на 9%) и Segformer (на 33%) при анализе 33 различных признаков на рентгенограммах.
    - YOLOv8: При сегментации рентгенопрозрачных поражений нижней челюсти использование аугментации данных позволило достичь точности (Precision) 100% и полноты (Recall) 94,5% (Rašić et al., 2023).
    - MARes-Net: Специализированная сеть на базе U-Net, усиленная механизмами внимания и остаточными связями, показала IoU 86,17%, точность 93,84% и F1-меру 93,21% при сегментации кист челюсти (Ding et al., 2024).

2. Обнаружение объектов (Object Detection)
Для локализации поражений с помощью ограничивающих рамок лучшие результаты зафиксированы у:
    - YOLOv8: В задаче детекции поражений нижней челюсти с применением аугментации модель достигла mAP@50 на уровне 97,5% (Rašić et al., 2023).
    - YOLOv11: Самая современная версия архитектуры показала mAP 86% в мультиклассовой конфигурации для трех типов кист, продемонстрировав особую эффективность в обнаружении дентигеральных кист с AP 91% (Kaygısız et al., 2025).
    - EfficientDet-D4: Эта модель показала общую точность обнаружения 92% при классификации состояния гайморовых пазух, при этом точность для здоровых пазух достигла 98% (Ha et al., 2023).

3. Классификация (Classification)
Наилучшие метрики в задачах отнесения изображений к определенным категориям патологий показали:
    - VGG16 (Transfer Learning): При использовании трансферного обучения модель достигла исключительной точности 98,48% в классификации периодонтальных кист (Lakshmi & Dheeba, 2023).
    - GLCM + SVM: Применение текстурных признаков (Grey Level Co-occurrence Matrix) в сочетании с классификатором SVM позволило достичь точности 98% при различении кист, опухолей и абсцессов (Kumar et al., 2023).
    - DenseNet121 + CBAM: Модель, усиленная механизмом внимания, показала общую точность 92,54% в классификации шести распространенных патологий челюсти (Brahmbhatt & Shah, 2026).

Сводная таблица лучших результатов:
| Модель               | Задача                         | Ключевая метрика               | Источник                 |
|----------------------|--------------------------------|--------------------------------|--------------------------|
| Radious (Mask2Former) | Семантическая сегментация       | mIoU 90%                       | Mashayekhi et al., 2023  |
| YOLOv8               | Детекция / Сегментация          | mAP 97,5% / Precision 100%     | Rašić et al., 2023       |
| VGG16                | Классификация                   | Accuracy 98,48%                | Lakshmi & Dheeba, 2023   |
| GLCM + SVM           | Текстурная классификация        | Accuracy 98%                   | Kumar et al., 2023       |
| MARes-Net            | Сегментация кист                | IoU 86,17%                     | Ding et al., 2024        |

## Выводы:
- Обнаружено достаточно мало работ, посвященных сегментации именно радикулярных (периапикальных) кист.
- Имеются объективные сложности для сегментации кист на верхней челюсти из-за наложения гайморовых пазух.
- Исследователи использовали различные аугментации в решении данной задачи.
- На данный заявлено 3 эффективных архитектуры для решения задачи сегментации кист:
    - YOLOv8
    - MARes-Net (на базе U-Net)
    - Radious (BEIT-Adaptor + Mask2Former)

## Использованная литература

- Brahmbhatt, D. K., & Shah, J. S. (2026). An automated diagnostic support system for jaw pathologies on panoramic radiographs: A DenseNet121-CBAM deep learning study with histopathological correlation. Journal of Stomatology, Oral and Maxillofacial Surgery, 127(2), 102604. 
 
- Ding, X., Jiang, X., Zheng, H., Shi, H., Wang, B., & Chan, S. (2024). MARes-Net: Multi-scale attention residual network for jaw cyst image segmentation. Frontiers in Bioengineering and Biotechnology, 12, 1454728. 
 
- Ha, E. G., Jeon, K. J., Choi, H., Lee, C., Choi, Y. J., & Han, S. S. (2023). Automatic diagnosis of retention pseudocyst in the maxillary sinus on panoramic radiographs using a convolutional neural network algorithm. Scientific Reports, 13, 2734. 
 
- Kaygısız, Ö. F., Uranbey, Ö., Gürsoytrak, B., Gür, Z. B., Çiçek, A., & Canbal, M. A. (2025). A deep learning approach based on YOLO v11 for automatic detection of jaw cysts. BMC Oral Health, 25, 1518. 
 
- Kwon, O., Yong, T. H., Kang, S. R., Kim, J. E., Huh, K. H., Heo, M. S., Lee, S. S., Choi, S. C., & Yi, W. J. (2020). Automatic diagnosis for cysts and tumors of both jaws on panoramic radiographs using a deep convolution neural network. Dentomaxillofacial Radiology, 49(8), 20200185. 
 
- Kumar, V. S., Kumar, P. R., Yadalam, P. K., Anegundi, R. V., Shrivastava, D., Alfurhud, A. A., Almaktoom, I. T., Alftaikhah, S. A. A., Alsharari, A. H. L., & Srivastava, K. C. (2023). Machine learning in the detection of dental cyst, tumor, and abscess lesions. BMC Oral Health, 23, 833. 
 
- Lakshmi, T. K., & Dheeba, J. (2023). Classification and Segmentation of Periodontal Cyst for Digital Dental Diagnosis Using Deep Learning. Computer Assisted Methods in Engineering and Science, 30(2), 131–149. 
 
- Mashayekhi, M., Ahmadi Majd, S., Amiramjadi, A., & Mashayekhi, B. (2023). Radious: Unveiling the enigma of dental radiology with BEIT adaptor and Mask2Former in semantic segmentation. arXiv:2305.06236.
 
- Rašić, M., Tropčić, M., Karlović, P., Gabrić, D., Subašić, M., & Knežević, P. (2023). Detection and segmentation of radiolucent lesions in the lower jaw on panoramic radiographs using deep neural networks. Medicina, 59(12), 2138. 
 
- Rašić, M., Tropčić, M., Pupić-Bakrač, J., Subašić, M., Čvrljević, I., & Dediol, E. (2024b). Utilizing deep learning for diagnosing radicular cysts. Diagnostics, 14(13), 1443. 
 
- Yang, H., Jo, E., Kim, H. J., Cha, I. H., Jung, Y. S., Nam, W., Kim, J. Y., Kim, J. K., Kim, Y. H., Oh, T. G., & Kim, D. (2020). Deep learning for automated detection of cyst and tumors of the jaw in panoramic radiographs. Journal of Clinical Medicine, 9(6), 1839. 
 
- Yu, D., Hu, J., Feng, Z., Song, M., & Zhu, H. (2022). Deep learning based diagnosis for cysts and tumors of jaw with massive healthy samples. Scientific Reports, 12, 1855. 
 
- Zhu, H., Yu, H., Zhang, F., Cao, Z., Wu, F., & Zhu, F. (2022). Automatic segmentation and detection of ectopic eruption of first permanent molars on panoramic radiographs based on nnU-Net. International Journal of Paediatric Dentistry, 32, 785–792. 
 
