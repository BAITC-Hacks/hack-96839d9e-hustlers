# Development workflow — HackAlem AI

Пятичасовой хакатон: сначала минимальный проверяемый end-to-end сценарий.
Стек и кейс ещё не определены, поэтому команд запуска приложения и тестов пока нет.

1. **Understand case:** изучить требования, данные, ограничения и существующий код.
2. **Define acceptance criteria:** определить MVP и проверяемый основной сценарий.
3. **Design minimal architecture:** один основной агент; минимальные компоненты и зависимости.
4. **Implement end-to-end path:** провести реальный запрос до результата через минимальный интерфейс.
5. **Add tools:** добавить только необходимые tools с валидацией и структурированными ошибками.
   Если tool необходим основному сценарию, реализовать его уже на шаге 4.
6. **Add retrieval only if required:** сначала оценить простой поиск, затем усложнять по необходимости.
7. **Test:** проверить основной сценарий и важные ошибки; выполнить доступные проверки стека.
8. **Integrate UI:** подключить пользовательский интерфейс к проверенному пути и повторить end-to-end проверку.
9. **Deploy:** только если требуется кейсом/сдачей; выбрать минимальный подход тогда, не заранее.
10. **Verify:** проверить итоговую среду, обработку ошибок и отсутствие утечек секретов.
11. **README:** описать фактический стек, настройку, переменные без значений, запуск, проверки и ограничения.
12. **Submit:** сверить критерии сдачи; commit/push выполнять только по явному запросу пользователя.

При выборе стека дополнить `.gitignore` фактическими путями dependencies,
virtual environments, build artifacts и framework-generated files. Сохранить
lockfiles и необходимые конфигурационные шаблоны в репозитории. Актуализировать
`.env.example` под реально используемые переменные; secrets передавать backend
через окружение, никогда не коммитить `.env`.

## Definition of Done

A feature is done when:

- The required behavior exists.
- The main scenario works end-to-end.
- Errors are handled reasonably.
- Relevant tests/checks pass.
- No secrets are exposed.
- No unnecessary dependencies were introduced.
- The implementation matches the project architecture.

Перед завершением проверить `git diff`, staged diff и `git status --short`.
В отчёте указать изменённые файлы, выполненные проверки и ограничения.
Если тестов ещё нет, явно сообщить об этом; не создавать бессмысленные тесты.
