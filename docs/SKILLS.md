# Skills

Skill — логический capability/domain workflow, который может использовать агент.
Это соглашение проекта, а не требование устанавливать framework или plugin.

```text
Skill
 ├── instructions
 ├── tools
 ├── domain logic
 └── validation
```

- Skill решает одну логическую задачу.
- Не зависит напрямую от UI и не хранит секреты.
- Использует tools через контролируемый интерфейс backend, не обходит executor.
- Детерминированные правила делегирует domain logic, а не помещает в prompt.
- Имеет проверяемые вход, выход и критерии ошибок; должен быть тестируемым.
- Не создавать skills заранее для неизвестного кейса.

## Абстрактный пример

```text
Skill: <domain capability>

Input:
  ...

Process:
  ...

Tools:
  ...

Output:
  ...

Failure cases:
  ...
```

Каталог `skills/` пока пуст. Наполнять его только после определения требований.
