# Домашнее задание: Semi-Supervised Learning с Multi-Branch MLP

## Описание проекта

Реализация semi-supervised learning подхода с multi-branch нейронной сетью для мультиклассовой классификации. Основная цель - достижение F1-score (macro) не менее 32% через использование как размеченных, так и неразмеченных данных и методов борьбы с дисбалансом классов.

### Multi-Branch MLP с Semi-Supervised Learning

Реализована улучшенная архитектура **MultiBranchMLP** с тремя параллельными ветками, каждая с BatchNorm и residual connections:

- **Bottleneck Branch** - сужение размерности (dim → dim//4 → dim) с BatchNorm на каждом слое
- **Inverted Bottleneck Branch** - расширение размерности (dim → dim×4 → dim) с BatchNorm на каждом слое  
- **Regular Branch** - обычный residual блок (dim → hidden_dim×2 → dim) с BatchNorm на каждом слое

Архитектурные особенности:
- Входная проекция: Linear → BatchNorm1d → GELU → Dropout
- Параллельная обработка через три ветки (по 4 блока каждая)
- Объединение через конкатенацию (`concat`) или суммирование
- Выходная проекция: BatchNorm1d → Linear

### Semi-Supervised методы

Реализована стратегия **Pseudo-labeling** с адаптивным взвешиванием:

- **Confidence-based Filtering**: использование только предсказаний с уверенностью > 0.95
- **Curriculum Learning**: 10 эпох warmup только на размеченных данных
- **Progressive Weighting**: постепенное увеличение веса unsupervised loss от 0 до 0.3
- **Adaptive Thresholding**: `pseudo_label_threshold=0.95` для качества псевдометок

### Подобраны оптимальные значения

Гиперпараметры, настроенные для стабильного обучения:

- **Архитектура**: `num_blocks=4`, `dropout=0.1`, `combine_mode='concat'`
- **Оптимизация**: `learning_rate=1e-3`, `AdamW` с `weight_decay=1e-4`
- **Scheduling**: `CosineAnnealingLR` с `eta_min=1e-6`
- **SSL параметры**: `pseudo_label_threshold=0.95`, `consistency_weight=0.3`, `warmup_epochs=10`
- **Data loading**: `batch_size=256`, `unlabeled_batch_ratio=2`

### Проект использует

**Ключевые компоненты системы:**

- **SemiSupervisedDataModule** - обработка размеченных и неразмеченных данных с параллельными DataLoader'ами
- **SemiSupervisedLightningModule** - обучение с псевдоразметкой и прогрессивным взвешиванием
- **MultiBranchMLP** - улучшенная архитектура с BatchNorm и residual connections
- **Comprehensive Metrics** - F1-macro, accuracy для multiclass классификации
- **Error Handling** - устойчивая загрузка данных с обработкой исключений
- **Memory Optimization** - очистка pandas DataFrame после создания тензоров

## Структура проекта
```text
homework_3_semi_supervised_learning/
├── hw_semi_supervised_learning.ipynb          # Основной ноутбук с реализацией semi-supervised learning
├── model.py                                   # Содержит MultiBranchMLP архитектуру
├── data_module.py                             # DataModule для работы с размеченными и неразмеченными данными
├── lightning_module.py                        # Содержит BaseLightningModule для обучения моделей в PyTorch Lightning
├── requirements.txt                           # Список Python-пакетов и их версий, необходимых для запуска
└── README.md                                  # Документация