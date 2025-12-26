# Homework 6: Documentation

## Что Сделано

### MkDocs

- Material theme
- 12 страниц документации (сокращено с 40+)
- Автогенерация API docs
- Search, dark mode, code highlighting

### Структура

```
docs/
├── getting-started/
├── guides/
├── api/
├── deployment/
└── about/
```

### GitHub Actions

```yaml
on: push [main]
jobs: mkdocs gh-deploy --force
```

Автодеплой при push в main.

### Report Generator

`scripts/generate_experiment_report.py` - генерирует отчеты из MLflow с визуализациями.

## Команды

```bash
make docs-serve       # Preview
make generate-report  # Generate experiment report
```

## Результат

**Site:** https://vladimirskvortsov.github.io/itmo-ml-best-practices

12 страниц вместо 40+, только практическая информация без лишних объяснений.
