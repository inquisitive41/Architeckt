# Architeckt: Руководство по установке

Этот документ описывает системные требования и шаги настройки как для локального инференса, так и для распределенных кластеров обучения.

## 💻 Аппаратные Требования (Hardware)

### Для Инференса / Использования (Минимум)
- **GPU:** 1x видеокарта NVIDIA с минимум 24GB видеопамяти (например, RTX 3090, RTX 4090).
- **ОЗУ:** 32 ГБ системной памяти.
- **Диск:** 50 ГБ свободного места (строго рекомендуется SSD/NVMe).

> [!NOTE]
> Запуск 11-миллиардной модели Architeckt на 24GB VRAM **строго требует 8-битного квантования** (библиотека `bitsandbytes`). Для чистого fp16/bf16 инференса вам потребуется минимум 32 ГБ видеопамяти.

### Для Обучения (Рекомендуется)
- **GPU:** Кластер из 8x NVIDIA A100 (80GB) или H100.
- **Соединение:** NVLink или NVSwitch (для быстрой синхронизации FSDP).
- **Диск:** Сетевое хранилище (NAS) для тяжелых токенизированных датасетов.

---

## 🛠 Программные Требования (Software)

Architeckt написан на чистом PyTorch, но использует самые свежие функции CUDA.

- **ОС:** Linux (Ubuntu 22.04+) или Windows 11 (через WSL2).
- **Python:** `3.10` или новее.
- **PyTorch:** `2.1` или новее (с поддержкой CUDA 11.8 / 12.1+).

---

## 📦 Шаги установки

### Шаг 1: Скачайте репозиторий

```bash
git clone https://github.com/your-org/Architeckt.git
cd Architeckt
```

### Шаг 2: Создайте виртуальное окружение

С помощью `conda` (рекомендуется):
```bash
conda create -n architeckt python=3.10
conda activate architeckt
```

Или с помощью стандартного `venv`:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Шаг 3: Установите ядро зависимости

Сначала установите PyTorch с поддержкой нужной версии CUDA (здесь CUDA 12.1):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pyyaml tqdm
```

### Шаг 4: Установите утилиты

Чтобы включить поддержку 8-битного сжатия (для локальных ПК):
```bash
pip install bitsandbytes
```

Для запуска аналитических бенчмарков (задержки, потребление памяти):
```bash
pip install matplotlib pandas
```

> [!WARNING]
> Если библиотека `bitsandbytes` выдает ошибку CUDA на Windows, убедитесь, что вы работаете через подсистему Linux (WSL2), либо используете неофициальную сборку `bitsandbytes-windows`.

---

## 🚦 Верификация установки

После установки обязательно прогоните базовые "дымовые" тесты (smoke tests), чтобы убедиться, что всё компилируется.

```bash
export PYTHONPATH="src"
python src/attention/adaptive_heads.py
python src/attention/linear_attention.py
python src/activations/swiglu_t.py
```

Если скрипты пишут `test passed`, значит установка завершена успешно. Вы готовы к работе!
