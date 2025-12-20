# Домашнее задание: Multi-Branch MLP для классификации Wine Quality

## Описание проекта

Реализация multi-branch нейронной сети для мультиклассовой классификации качества вина на датасете Wine Quality. Основная цель - достижение F1-score (macro) не менее 40% через использование продвинутой архитектуры и методов борьбы с дисбалансом классов.

## Что сделано 

### Multi-Branch MLP

Реализована модель с тремя параллельными ветками, результаты которых объединяются:

- **Bottleneck Branch** - сужение размерности (dim → dim//4 → dim)
- **Inverted Bottleneck Branch** - расширение размерности (dim → dim×4 → dim) 
- **Regular Branch** - обычный residual блок (dim → hidden_dim → dim)

```Multi-Branch MLP```:
- Принимает вход и проецирует в hidden_dim
- Пропускает через три параллельные ветки (каждая из num_blocks блоков своего типа)
- Объединяет результаты через конкатенацию (concat) или суммирование (sum)
- Проецирует в выходную размерность

### Подобраны оптимальные значения:

- Глубина модели (num_blocks)
- Ширина модели (hidden_dim)
- Learning rate
- Оптимизатор

### Проект использует:
- BaseLightningModule из lightning_module.py
- F1-оптимизированные веса 
- early stopping для предотвращения переобучения

## Структура проекта
```text
homework_2_multi_branch_classification/
├── hw_multi_branch_classification.ipynb # Основной ноутбук с реализацией
├── lightning_module.py 
├── mlp_model.py 
├── model.py 
├── utils.py 
├── wine_quality_data.py 
├── requirements.txt 
└── README.md 
```