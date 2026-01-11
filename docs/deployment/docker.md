# Docker

## Dockerfile

```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git git-lfs
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "src/models/train_model.py"]
```

## Docker Compose

```yaml
services:
  postgres:
    image: postgres:15
    ports: ["5432:5432"]
  
  mlflow-server:
    image: python:3.10
    ports: ["3000:3000"]
    command: mlflow server --port 3000
```

## Commands

```bash
make docker-up
make docker-down
```
