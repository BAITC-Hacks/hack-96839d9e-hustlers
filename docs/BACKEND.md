# Backend участника №2

Оригинальные NOAA GFS → проверенные погодные точки → модель участника №1 →
проверка результата → сохранение версии → интерфейс. Управление доступно через
одного агента Responses API или явно обозначенный детерминированный режим.

## Что находится в репозитории

- `core.py`: времена, координаты, ошибки, атомарные файлы и контрольные суммы.
- `weather.py`: оригинальный индекс NOAA, ограниченные HTTP Range, ecCodes,
  извлечение обеих турбин, исторические проверки, возобновляемый кэш.
- `cli.py`: аудит SCADA, получение одного часа/выпуска, сбор диапазона, CSV для ML.
- `ml_bridge.py`: контракт модели участника №1; обучение не выполняется.
- `pipeline.py`: проверки, агрегаты, сохранение и сравнение версий.
- `agent.py`: Responses API, строгие схемы tools, максимум восемь шагов.
- `backend.py`: совместимые с UI `run_forecast`, `run_replay`, `check_updates`.

Без модели возвращается `MODEL_NOT_CONFIGURED`, без подстановки фиктивных чисел.
Живой OpenAI-вызов требует ключа команды. Тестовые модели используются только
в pytest временных каталогах, не создавая «готовые прогнозы» в `data/`.

## Установка

Из корня проекта (проверяется Python 3.12, Linux):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
export PYTHONPATH=src
```

Без uv: `python3.12 -m venv .venv`, затем
`.venv/bin/python -m pip install -r requirements-dev.txt`.
Windows: `.venv\Scripts\python.exe`, переменные через PowerShell
`$env:PYTHONPATH = "src"` и аналогичные команды.

## Локальные данные и сбор

Все предоставленные и скачанные данные находятся в **`data/`, исключённой из Git**:

```text
data/scada/turbine_1.csv
data/scada/turbine_2.csv
data/weather/gfs/YYYYMMDDTHH/f019.json
data/weather/bundles/gfs-<hash>.json
data/weather/weather_for_ml.csv
data/reports/scada_audit.json
data/reports/weather_coverage.json
data/reports/collection_<start>_<end>.json
data/forecasts/forecast-<hash>/
```

Копии SCADA побайтово совпадают с предоставленными файлами. Аудит показывает
пропущенные отметки времени и подтверждает отсутствие февральских наблюдений.

На текущем рабочем месте собраны **152 ежедневных выпуска с 30.09.2025
по 28.02.2026: 14 592 строки для двух турбин**, по 48 часов в каждом выпуске.
Архив повторно проверен без сети, пропусков и дублей ключа нет.
Итоговые контрольные суммы и ограничения:
`data/reports/backend_verification.json`.

```bash
.venv/bin/python -m windops.cli audit
.venv/bin/python -m windops.cli probe --run 2026-01-31T00:00:00Z --lead 19
.venv/bin/python -m windops.cli weather --origin 2026-01-31T23:00:00+05:00
.venv/bin/python -m windops.cli weather --origin 2026-01-31T23:00:00+05:00 --offline
```

Возобновляемый сбор и объединение CSV:

```bash
.venv/bin/python -m windops.cli collect --start 2025-09-30 --end 2025-12-30 --workers 8
.venv/bin/python -m windops.cli collect --start 2025-12-31 --end 2026-01-30 --workers 8
.venv/bin/python -m windops.cli collect --start 2026-01-31 --end 2026-02-28 --workers 8
.venv/bin/python -m windops.cli export-weather
```

Даты означают **местные даты выпуска в 23:00**. Выпуск 31 января обеспечивает
1–2 февраля, выпуск 28 февраля содержит 1–2 марта и сохраняется целиком.
Начальный обучающий архив ограничен осенью 2025 года: наличие исходной SCADA
с 2023 года не означает наличия GFS за весь этот период. Расширяйте диапазон
тем же сборщиком по необходимости.

Один объект содержит пять выбранных глобальных полей, суммарно около 5 МБ.
Они загружаются один раз для двух турбин. В кэше остаются только точки, индекс
и свидетельства; глобальные сетки не накапливаются. Сбор диапазона требует
десятки ГБ трафика. Начинайте с малого; по умолчанию восемь потоков, максимум 24.
Временные ошибки повторяются до трёх раз с задержкой. Уже проверенные объекты
используются из кэша. Частичный выпуск не попадает в таблицу для обучения.

Отчёт `collection_*.json` перечисляет завершённые выпуски и ошибки.
`export-weather` обновляет общую таблицу и `weather_coverage.json`.
Повреждённый кэш вызывает ошибку, а не незаметную подмену источника.

## Координаты, единицы и историческая доступность

Извлечены координаты из двух Google Maps-ссылок организаторов:

- `turbine_1`: 43.645150, 78.535604.
- `turbine_2`: 43.643198, 78.538828.

Проверенный ближайший узел GFS: 43.75, 78.5, примерно в 12 км от площадок.
Одинаковая погода двух турбин допустима; искусственный шум не добавляется.
Ветер на 100 м не объявляется ветром на высоте ступицы.

Все времена backend — UTC; интерфейс — `Asia/Almaty`. В 2025–2026 годах
23:00 Казахстана = 18:00 UTC. Целевой час равен origin + шаг 1…48 часов.
Ежедневные входы используют GFS 00 UTC того же дня; резерв — 00 UTC предыдущего.
Для выпуска 31 января 18:00 UTC нужны **f019…f066**, а не f001…f048.

`availability_upper_bound` берётся из Last-Modified оригинального объекта
публичного NOAA S3. Это верхняя граница появления **данной версии** в архиве,
а не точное первоначальное время публикации NOAA. Объект допускается только
при `run_initialized_at <= availability_upper_bound <= forecast_origin`.
Позднее перезалитые объекты отклоняются; их прежняя доступность не угадывается.
Эта политика явно записана в `availability_basis`. Фиктивное правило
«инициализация + 6 часов = verified» не используется.

Проверяется каждый час. Сервер должен вернуть 206 и точный Content-Range;
полный HTTP 200 отвергается до чтения тела. ETag первого фрагмента фиксируется,
остальные запрашиваются с If-Match. Сохраняется SHA-256 каждого поля и кэша.
`retrieved_at` — сегодняшнее время скачивания, оно не доказывает доступность
в прошлом и не влияет на идентификатор содержимого.

Семантика 10-минутной метки SCADA (начало/конец интервала) ещё не подтверждена.
Её фиксирует участник №1 при агрегации; также требует уточнения нормализация
времени до марта 2024 года. Начальный погодный архив этого перехода не затрагивает.

## Таблица для участника №1

Ключ: `(site_id, forecast_origin, target_time)`. Перекрывающиеся горизонты
сохраняются. В CSV есть:

- `site_id`, `forecast_origin`, `target_time`, `lead_hours` (горизонт мощности).
- `run_initialized_at`, `gfs_lead_hours` (шаг файла GFS).
- `availability_upper_bound`, `availability_basis`.
- `temperature_2m_c`, `wind_u_10m_ms`, `wind_v_10m_ms`,
  `wind_u_100m_ms`, `wind_v_100m_ms`.
- `wind_speed_10m_ms`, `wind_speed_100m_ms`,
  `wind_direction_10m_deg`, `wind_direction_100m_deg`.
- Запрошенные `latitude`, `longitude`; фактические `grid_latitude`,
  `grid_longitude`, `distance_km`.
- `provider`, `provider_model`, `weather_version`, `quality_flag`.

Температура переведена из K в °C, скорость вычислена через hypot(u, v).
Направление — метеорологическое «откуда», градусы от севера по часовой стрелке.
При штиле направление условно 0, скорость и компоненты остаются нулевыми.
Схема входов: `gfs-0p25-nearest-v1`.

Разделять train/validation/test нужно по target_time без пересечения целевых
часов. Январская проверочная модель не обучается на январе. Первый февральский
origin — 31 января 23:00 местного времени: нельзя включать измерения 23:10–23:50
этого дня в её обучение. Cutoff задаётся точным временем, а не только датой.

## Контракт ML-модуля

Участник №1 предоставляет импортируемый модуль:

```bash
export WINDOPS_ML_MODULE=имя_реального_модуля_участника_1
```

Функции с именованными аргументами:

```python
get_model_metadata(site_id, forecast_origin) -> dict
predict_power(site_id, weather_rows, model_version) -> list[dict]
```

Первая выбирает исторически допустимый артефакт. Пример формата метаданных:

```json
{
  "site_id": "turbine_1",
  "version": "реальная-версия-модели",
  "training_cutoff": "2025-12-31T00:00:00Z",
  "normalization": "0_1",
  "weather_schema": "gfs-0p25-nearest-v1",
  "provider_model": "gfs_pgrb2.0p25",
  "labels_available_by_cutoff": true
}
```

Дата — пример формата, не утверждение о существующей модели.
`labels_available_by_cutoff` подтверждает, что все целевые интервалы обучения
закончились и ответы были известны к cutoff. Backend проверяет даты, но не может
восстановить историю обучения из непрозрачного артефакта.

`weather_rows` содержит только выбранную турбину. Результат — ровно 24/48 записей
с `target_time` (тот же ISO UTC) и `prediction` (конечное число 0–1).
Необязательные `p10/p50/p90` покрывают весь горизонт и должны быть упорядочены.
Backend не делает clip и не заполняет пропуски. Преобразования признаков
принадлежат модулю №1; обучение внутри predict не выполняется.

## Запуск и агент

После подключения реального ML-модуля:

```bash
export WINDOPS_BACKEND_MODULE=windops.backend
export WINDOPS_EXECUTION_MODE=deterministic
.venv/bin/python -m windops.cli forecast --site turbine_1 --origin 2026-01-31T23:00:00+05:00
.venv/bin/python -m streamlit run app.py
```

В игнорируемом `config.local.json` подготовлены site_id, зона и соглашение времени.
На другом рабочем месте заполните их по `config.example.json`.
UI по умолчанию подключает `windops.backend`; переменная позволяет заменить мост.
Для запрета сети используйте `WINDOPS_OFFLINE=1`.

Для агента задайте в окружении `OPENAI_API_KEY`, `OPENAI_MODEL` (доступная вашему
API-проекту модель) и `WINDOPS_EXECUTION_MODE=agent`. `.env` автоматически
не читается. Режим `auto` при отсутствии ключа или сбое LLM явно помечает запуск
как `deterministic_fallback`. Детерминированный режим не объявляется LLM-агентом.

LLM получает параметры запроса, идентификаторы, статусы и агрегаты. Сырые CSV и
погодные ряды ему не передаются. Используется `store=false`; reasoning items
нужны для продолжения API-цикла, но не пишутся в журнал проекта. Инструменты:
list_weather_candidates, fetch_weather, validate_weather, predict_power,
analyse_forecast, publish_forecast. Схемы строгие, дополнительные поля запрещены,
аргументы повторно проверяет backend. LLM не может менять origin, турбину,
координаты, модель, файловые пути или правила. Проверки действуют внутри predict
и publish, даже если отдельный validate пропущен. Успех — сохранённый bundle.

## Replay, обновления и результаты

```bash
.venv/bin/python -m windops.cli replay --start 2026-01-31 --end 2026-02-28
.venv/bin/python -m windops.cli check-updates forecast-<реальный-id> --as-of 2026-02-01T06:00:00Z
```

Replay двигает виртуальное время. Будущие файлы на диске остаются недопустимыми
входами для более раннего origin. Replay мощности требует модели; погодный архив
собирается независимо через collect.

check_updates проверяет текущий шестичасовой цикл GFS и предыдущий. Новый
допустимый цикл автоматически запускает расчёт, сохраняет новую версию и
`revises_forecast_id`. Без as_of UI сдвигает виртуальное время на 12 часов;
системное «сегодня» не используется. Без нового цикла возвращается пустой список.
Это историческая проверка по запросу, а не постоянно работающий планировщик.
Сравниваются только общие часы. Одинаковые входы возвращают тот же forecast_id.

В каталоге выпуска: `bundle.json`, `bundle_checksum.json`, `forecast.csv`,
`forecast_manifest.json`, `weather_manifest.json`, `events.jsonl`,
`validation_report.json`. Старые версии сохраняются. Журнал содержит реальные
вызовы и статусы, без секретов и придуманных рассуждений.

## Проверки и ограничения

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src app.py
uv pip check --python .venv/bin/python
```

Тесты проверяют время, сетку, будущее, нулевой ветер, ошибки модели, tool calling,
fallback, идемпотентность, сохранение старых версий, Range 206, контрольные суммы
и стыковку с UI. Они не доказывают качество ML или живую работу OpenAI.
Реальные проверки: probe, полный weather, повтор weather --offline, аудит CSV.
Сквозной реальный прогноз ожидает модель №1; живой LLM-сценарий — API-ключ.

## Источники

- [NOAA GFS в публичном AWS](https://registry.opendata.aws/noaa-gfs-bdp-pds/).
- [Инвентарь GFS](https://www.nco.ncep.noaa.gov/pmb/products/gfs/).
- [ecCodes Python](https://confluence.ecmwf.int/display/ECC/Python+3+interface).
- [Responses function calling](https://developers.openai.com/api/docs/guides/function-calling).
- [Сохранение контекста reasoning без хранения Responses](https://developers.openai.com/api/docs/guides/reasoning).

Используется прямое чтение индекса и ecCodes: это позволяет контролировать
диапазоны байтов и происхождение каждого поля без дополнительного слоя Herbie.
Отдельная база данных, GPU и обучение LLM для backend не требуются.
