# Deep learning Course - Homeworks

Репозиторий с домашними заданиями по курсу "Глубокое обучение" (Deep Learning). Содержит реализации различных нейросетевых архитектур и алгоритмов машинного обучения.

## 📋 Список домашних заданий

### 🏠 Homework 1: Pytorch lightning classification
**Цель:** Освоение фреймворка PyTorch Lightning для создания структурированных и эффективных конвейеров глубокого обучения на примере задачи многоклассовой классификации вин.


**Основные задачи:**
- Реализация класса для управления данными - **WineDataModule**
- Реализация **NeuralNetLightning** - модель на PyTorch Lightning с методами:
   - `__init__` - инициализация модели и параметров
   - `forward` - прямой проход
   - `training_step` - шаг обучения
   - `validation_step` - шаг валидации
   - `test_step` - шаг тестирования
   - `configure_optimizers` - настройка оптимизатора

**Технологии:** Python, PyTorch Lightning, PyTorch, scikit-learn, NumPy, matplotlib

---

### 🏠 Homework 2: Multi-Branch MLP для классификации Wine Quality
**Цель:** Достижение F1-score (macro) не менее 40% через использование продвинутой архитектуры и методов борьбы с дисбалансом классов

**Multi-Branch MLP модель:**
- Реализована архитектура с тремя параллельными ветками:
  - **Bottleneck Branch** - сужение размерности (dim → dim//4 → dim)
  - **Inverted Bottleneck Branch** - расширение размерности (dim → dim×4 → dim)
  - **Regular Branch** - обычный residual блок (dim → hidden_dim → dim)
- Модель принимает вход, проецирует в hidden_dim, пропускает через ветки и объединяет результаты через конкатенацию или суммирование

**Оптимизация гиперпараметров:**
- Подбор глубины модели (num_blocks)
- Подбор ширины модели (hidden_dim)
- Выбор learning rate
- Выбор оптимизатора

**Мониторинг метрик:**
- Отслеживание F1 score для каждого класса
- Построение confusion matrix
- Использование weighted loss для борьбы с дисбалансом классов

**Технологии:** Python, PyTorch Lightning, PyTorch, scikit-learn, NumPy, matplotlib

---

### 🏠 Homework 3: Semi-Supervised Learning с Multi-Branch MLP

**Цель:** Достижение F1-score (macro) ≥ 32% через использование semi-supervised методов и продвинутых архитектур для работы с частично размеченными данными

**Multi-Branch MLP модель:**
- Реализована улучшенная архитектура с тремя параллельными ветками:
  - **Bottleneck Branch** - сужение размерности (dim → dim//4 → dim) с BatchNorm на каждом слое
  - **Inverted Bottleneck Branch** - расширение размерности (dim → dim×4 → dim) с BatchNorm на каждом слое
  - **Regular Branch** - обычный residual блок (dim → hidden_dim → dim) с BatchNorm на каждом слое
- Модель принимает вход, проецирует в hidden_dim через Sequential с BatchNorm+GELU+Dropout, пропускает через параллельные ветки и объединяет результаты через конкатенацию или суммирование, затем применяет выходную проекцию с BatchNorm

**Semi-Supervised Learning методы:**
- **Pseudo-labeling** - автоматическая разметка неразмеченных данных с порогом уверенности (pseudo_label_threshold=0.95) и warmup периодами
- **Consistency Weighting** - прогрессивное взвешивание unsupervised loss (consistency_weight=0.3) с адаптивным увеличением веса
- **Curriculum Learning** - постепенное включение неразмеченных данных после warmup_epochs=10

**Оптимизация гиперпараметров:**
- Подбор глубины модели (num_blocks=4)
- Подбор ширины модели (hidden_dim)
- Выбор learning rate (1e-3) и CosineAnnealing scheduler
- Выбор оптимизатора (AdamW по умолчанию) с weight_decay=1e-4
- Настройка веса consistency loss (consistency_weight=0.3)
- Порог уверенности для pseudo-labeling (pseudo_label_threshold=0.95)
- Настройка dropout (0.1) для регуляризации

**Технологии:** Python, PyTorch Lightning, PyTorch, scikit-learn, NumPy, matplotlib.
