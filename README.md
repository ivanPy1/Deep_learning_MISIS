# Deep learning Course - Homeworks

Репозиторий с домашними заданиями по курсу "Глубокое обучение" (Deep Learning). Содержит реализации различных нейросетевых архитектур и алгоритмов машинного обучения.

## 📋 Список домашних заданий

### 🏠 Homework 1: Pytorch lightning classification
**Цель:** Освоение фреймворка PyTorch Lightning для создания структурированных и эффективных конвейеров глубокого обучения на примере задачи многоклассовой классификации вин.


**Основные задачи:**
- Реализация класса для управления данными - WineDataModule
- Реализация NeuralNetLightning - модель на PyTorch Lightning с методами:
   - `__init__` - инициализация модели и параметров
   - `forward` - прямой проход
   - `training_step` - шаг обучения
   - `validation_step` - шаг валидации
   - `test_step` - шаг тестирования
   - `configure_optimizers` - настройка оптимизатора

**Технологии:** Python, PyTorch Lightning, PyTorch, scikit-learn, NumPy, matplotlib

---
