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
