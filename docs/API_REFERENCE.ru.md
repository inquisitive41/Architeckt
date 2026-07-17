# Документация API

Этот документ содержит высокоуровневый обзор программных интерфейсов и базовых классов Architeckt. Он предназначен для разработчиков, желающих интегрировать Architeckt в свои проекты или модифицировать архитектуру.

## Основные Модули (Core)

### `src.attention.linear_attention.MSLA`
**Multi-Scale Linearized Attention**
Главный механизм внимания, заменяющий квадратичный Softmax.

```python
class MSLA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, n_scales: int):
        ...
```
- **Вход:** Тензор `x` с формой `(batch, seq_len, d_model)`.
- **Выход:** Тензор `out` с формой `(batch, seq_len, d_model)`.
- **Описание:** Вычисляет линейное внимание через кумулятивные суммы с экспоненциальным затуханием (decay). Обладает константным $O(1)$ потреблением памяти для KV-кэша.

### `src.attention.adaptive_heads.AdaptiveHeadGating`
**Маршрутизатор голов (AHG)**
Аппаратно отключает головы внимания с низкой полезностью.

```python
class AdaptiveHeadGating(nn.Module):
    def __init__(self, d_model: int, n_heads: int, threshold_percentile: float = 0.3):
        ...
```
- **Вход:** Тензор `x` `(batch, seq_len, d_model)`.
- **Выход:** Тензор-маска `gate` `(batch, seq_len, n_heads, 1, 1)`.
- **Использование:** Умножьте выход голов внимания на `gate`. Если модель находится в режиме `eval()`, самые слабые 30% голов будут строго занулены.

### `src.activations.swiglu_t.SwiGLU_T`
**Пороговый SwiGLU**
Создает разреженность внутри Feed-Forward сетей (FFN).

```python
class SwiGLU_T(nn.Module):
    def __init__(self, dim: int):
        ...
```
- **Описание:** Применяет `SiLU`, умноженный на маску, зависящую от обучаемого порога. Сигналы ниже порога обнуляются.

## Утилиты Обучения

### `src.training.trainer.ArchitecktTrainer`
**Главный цикл обучения**
Обрабатывает цикл PyTorch, смешанную точность (bf16), аккумуляцию градиентов и распределенное обучение FSDP.

```python
class ArchitecktTrainer:
    def __init__(self, model: nn.Module, config: TrainingConfig, device: Optional[torch.device] = None):
        ...
```
- **Методы:**
  - `train()`: Запускает итерации обучения.
  - `save_checkpoint(name: str)`: Сохраняет чекпоинт на диск.
  - `load_checkpoint(...)`: Восстанавливает состояние.

### `src.training.trainer.TrainingConfig`
Dataclass конфигурация.
- **Поля:** `learning_rate`, `use_fsdp`, `activation_checkpointing`, `max_steps`, и др.

## Утилиты Инференса

### `src.inference.inference_utils.load_for_inference`
**Загрузчик модели для экономии памяти.**

```python
def load_for_inference(model: nn.Module, checkpoint_path: str, use_8bit: bool = True) -> nn.Module
```
- **Описание:** Парсит чекпоинт PyTorch и на лету заменяет стандартные слои `nn.Linear` на 8-битные `Linear8bitLt` из библиотеки `bitsandbytes`.
- **Внимание:** В случае отсутствия библиотеки `bitsandbytes` функция безопасно загрузит стандартные (16-битные) веса.
