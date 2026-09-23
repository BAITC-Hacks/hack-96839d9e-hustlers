# WindOps AI

Почасовой прогноз нормализованной мощности ВЭС для диспетчера и аналитика.
Кейс HackAlem AI: две турбины, горизонт 24/48 часов, исторические выпуски
на февраль 2026 с погодой, доступной на момент выпуска.

**Работают интерфейс, погодный backend и обученные модели двух турбин.**
Исходные CSV находятся в `data/scada/`, реальные
архивы NOAA GFS сохраняются в `data/weather/`. Есть агент Responses API, проверки,
сохранение версий и replay. Реальный детерминированный путь проверен; живой
OpenAI-вызов требует отдельной проверки и API-ключа. Январский backtest:
`data/ml/january_backtest.json`, февральский CSV: `data/ml/february_hourly.csv`.

Установка backend, сбор погоды, таблица для обучения и контракт модели:
[docs/BACKEND.md](docs/BACKEND.md). Фактическое покрытие погоды записывается
в локальный `data/reports/weather_coverage.json`.
Обучение, версии моделей, январские метрики и воспроизведение:
[docs/ML.md](docs/ML.md). Подключение: `WINDOPS_ML_MODULE=windops.ml.plugin`.

| Возможность | Статус |
| --- | --- |
| Русский интерфейс, формы, история выпусков, графики и таблицы | Работает |
| Загрузка JSON-выпуска, паспорт и диагностический экспорт | Работает |
| Проверки времени, сетки, квантилей, нормировки и финального экспорта | Работает |
| Просмотр независимой январской проверки из JSON | Работает; сохранены 2928 реальных прогнозных пар |
| Явно включаемый синтетический fixture | Работает; итоговая выгрузка запрещена |
| NOAA GFS, исторические проверки и кэш | Реализованы; проверены реальные загрузки |
| События, replay и проверка обновлений | Проверен детерминированный backend с настоящей моделью |
| Обучение и реальные февральские прогнозы мощности | CatBoost двух турбин; 58 полных выпусков и 1344 часа февраля |

## Запуск

Backend проверен на Linux с Python 3.12; исходный интерфейс — на Windows
с Python 3.13.11. Нужен Python 3.11+. Команды для Linux и сборщика погоды
приведены в [docs/BACKEND.md](docs/BACKEND.md).
Команды для Windows выполняются из корня репозитория в PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Приложение доступно локально по `http://localhost:8501`. После активации
окружения также работает `streamlit run app.py`. Для одного запуска UI
достаточно `requirements.txt`; `requirements-dev.txt` дополнительно ставит
погодный backend, SDK OpenAI и pytest.
Для настоящих моделей дополнительно установите `requirements-ml.txt`.
Внешние шрифты, CDN, API-ключи и сеть для просмотра интерфейса не нужны.

Запуск с уже сохранёнными локальными моделями на Linux:

```bash
export PYTHONPATH=src
export WINDOPS_ML_MODULE=windops.ml.plugin
export WINDOPS_EXECUTION_MODE=deterministic
export WINDOPS_OFFLINE=1
.venv/bin/python -m streamlit run app.py
```

При первом запуске виден экран без данных и объяснение отсутствующих настроек.
В боковой панели включите **«Тест интерфейса · синтетика»**, выберите турбину
и горизонт, нажмите **«Загрузить тестовый выпуск»**. Это проверка UI,
а не работа модели. Demo не включается автоматически при ошибке backend.

## Реальные данные и интеграция

На этом рабочем месте `config.local.json` уже заполнен для двух турбин
и зоны `Asia/Almaty`. На другом рабочем месте скопируйте
`config.example.json` в `config.local.json` и заполните:

- `sites`: массив одной/двух записей `{"site_id": "идентификатор_из_backend"}`.
  UI сохраняет исходные идентификаторы; координаты задаются в backend.
- `timezone`: подтверждённая зона IANA исходных временных меток.
  Зона ноутбука не используется по умолчанию.
- `timestamp_convention`: `target_time = issued_at + horizon_step hours`.
  Первый шаг — час после момента выпуска. Иное соглашение требует адаптации.
- `normalization`: `0_1`, только если подтверждена нормировка источника.
- `default_origin`: проверенный момент выпуска в ISO 8601 со смещением или `null`.

Без настроенного момента форма предлагает 31.01.2026 23:00 как границу
первого февральского горизонта. Это параметр формы, не заявленный успешный выпуск.
Без подтверждённой зоны расчёт недоступен.

Переменные окружения (шаблон — `.env.example`):

| Переменная | Назначение |
| --- | --- |
| `WINDOPS_CONFIG` | Путь к JSON-конфигурации, по умолчанию `config.local.json` |
| `WINDOPS_BACKEND_MODULE` | Импортируемый Python-модуль адаптера команды |
| `WINDOPS_SMOKE_BUNDLE` | Путь к реальному сохранённому выпуску для отдельного smoke-test |
| `WINDOPS_ML_MODULE` | `windops.ml.plugin` для обученных моделей |
| `WINDOPS_MODEL_DIR` | Каталог артефактов, по умолчанию `models/` |

`.env` автоматически не загружается. Задавайте переменные средствами оболочки.
Произвольные импорты из загруженных файлов не выполняются. Все вызовы backend
сосредоточены в `src/windops/ui/adapter.py`; они происходят только по действию
пользователя. UI-контракт изолирован и поддержан модулем `windops.backend`;
контракт подключения участника №1 описан в `docs/BACKEND.md`.

Подробные схемы, правила времени и точки подключения:
[docs/INTEGRATION.md](docs/INTEGRATION.md). SCADA лежит в `data/scada/`, модели —
в `models/<site_id>/ml-<hash>/`; UI их напрямую не читает и не отправляет LLM.
Исходные данные и архив уже добавлены командой в Git. Каталоги `models/`
и `artifacts/` исключены из Git; результаты новых расчётов находятся в `data/ml/`
и `data/forecasts/`. Commit и push результатов автоматически не выполняются.

Источники результата разделены: новый расчёт на реальных архивах (`real`),
сохранённый выпуск (`cached`) и тестовый fixture (`demo`). Исполнитель исходного
расчёта показывается отдельно. Загруженный пользователем JSON не становится
доверенным только из-за поля `verified`: до проверки backend итоговый CSV
заблокирован. Доступны паспорт и явно маркированная диагностическая выгрузка.

Январский экран принимает `data/ml/january_backtest.json`. Команды подготовки,
обучения и проверки описаны в [docs/ML.md](docs/ML.md). Фактическая выработка
за февраль не предоставлена, поэтому ошибки на феврале не считаются.

## Устройство

```text
app.py                         Формы, вкладки и пользовательские действия
src/windops/ui/adapter.py       Единственная граница с backend, проверка bundle
src/windops/ui/state.py         Сессия, версии и состояния запуска
src/windops/ui/exports.py       Экспорт с повторной проверкой
src/windops/ui/charts.py        Графики без заполнения пропусков
src/windops/ui/quality.py       Независимая январская проверка
src/windops/ui/components.py    Компоненты интерфейса
src/windops/ui/theme.py         Общая тема
src/windops/ui/configuration.py Настройки площадок и времени
src/windops/ui/fixtures.py      Только явно выбранная синтетика
src/windops/ml/                 Подготовка, обучение, backtest и плагин моделей
.streamlit/config.toml         Локальная тема Streamlit
```

Стек: Streamlit 1.64.0, Plotly 6.9.0, pandas 2.3.3; проверка — pytest и
Streamlit AppTest. Погодный сервис и агент находятся в backend; обучение модели
не входит в UI. Загрузка вкладок и скачивание не повторяют расчёт.

## Проверка и демонстрация

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app.py src
.\.venv\Scripts\python.exe -m pip check
```

Результат проверки backend, ML и UI на Linux: **162 passed, без пропусков**
при `WINDOPS_RUN_REAL_ML_TEST=1` и `WINDOPS_SMOKE_BUNDLE` на настоящий выпуск;
точные переменные и команды — [docs/ML.md](docs/ML.md).
Проверены настоящий GFS, обученные модели, replay, обновление и прямой BackendAdapter.
Responses API проверен тестовыми подстановками; живой LLM-вызов не выполнялся.
Также выполнены `compileall` и `pip check`. Браузерная приёмка Chromium
относится к предыдущей версии UI; для текущего изменения использован AppTest.
Сценарий показа и ручная визуальная приёмка: [docs/DEMO.md](docs/DEMO.md).
Проверка перед сдачей: [docs/SUBMISSION_CHECKLIST.md](docs/SUBMISSION_CHECKLIST.md).

Дополнительная приёмка браузером Chromium (при уже запущенном приложении):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-browser.txt
$env:PLAYWRIGHT_BROWSERS_PATH = Join-Path (Get-Location) '.playwright'
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe scripts\browser_acceptance.py
```

Скриншоты и результаты сохраняются в исключённый из Git `test-results/browser/`;
используется только синтетика. Проверяются размеры 1366×768, 1920×1080 и 800×900,
наведение, масштабирование/сброс, раскрытия, экспорт и ошибочная загрузка JSON.
Часть стандартных служебных меню Streamlit (например, системная кнопка загрузки
файла) остаётся на английском; сценарии и подписи приложения — на русском.

Документация используемых библиотек:
[Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest),
[скачивание без перезапуска](https://docs.streamlit.io/develop/api-reference/widgets/st.download_button),
[Plotly: почасовые линии и пропуски](https://plotly.com/python/line-charts/).

Вклад этой версии: интерфейс участника 3 и backend участника 2 (GFS, кэш,
исторические проверки, агент, replay и версии). Модель участника 1 ожидается. Развёртывание,
публикация репозитория и отправка решения на платформу не выполнялись.
