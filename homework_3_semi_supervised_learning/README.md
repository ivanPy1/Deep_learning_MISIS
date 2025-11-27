# Домашнее задание: Semi-Supervised Learning с Multi-Branch MLP

## Описание проекта

Реализация semi-supervised learning подхода с multi-branch нейронной сетью для мультиклассовой классификации. Основная цель - достижение F1-score (macro) не менее 32% через использование как размеченных, так и неразмеченных данных и методов борьбы с дисбалансом классов.

## Что сделано

### Multi-Branch MLP с Semi-Supervised Learning

Реализована модель с тремя параллельными ветками, расширенная для semi-supervised обучения:

- **Bottleneck Branch** - сужение размерности (dim → dim//4 → dim)
- **Inverted Bottleneck Branch** - расширение размерности (dim → dim×4 → dim) 
- **Regular Branch** - обычный residual блок (dim → hidden_dim → dim)

```Multi-Branch MLP```:
- Принимает вход и проецирует в hidden_dim
- Пропускает через три параллельные ветки (каждая из num_blocks блоков своего типа)
- Объединяет результаты через конкатенацию (concat) или суммирование (sum)
- Проецирует в выходную размерность

### Semi-Supervised методы:

- **Pseudo-labeling** - автоматическая разметка неразмеченных данных
- **Consistency Regularization** - обучение устойчивости к аугментациям

### Подобраны оптимальные значения:

- Глубина модели (num_blocks)
- Ширина модели (hidden_dim) 
- Learning rate и scheduler
- Оптимизатор
- Вес consistency loss
- Порог уверенности для pseudo-labeling

### Проект использует:
- BaseLightningModule из lightning_module.py
- F1-оптимизированные веса 
- Early stopping для предотвращения переобучения
- Semi-supervised стратегии для использования неразмеченных данных

## Структура проекта
```text
homework_3_semi_supervised_learning/
├── hw_semi_supervised_learning.ipynb          # Основной ноутбук с реализацией semi-supervised learning
├── model.py                                   # Содержит MultiBranchMLP архитектуру
├── data_module.py                             # DataModule для работы с размеченными и неразмеченными данными
├── lightning_module.py                        # Содержит BaseLightningModule для обучения моделей в PyTorch Lightning
├── requirements.txt                           # Список Python-пакетов и их версий, необходимых для запуска
└── README.md                                  # Документация