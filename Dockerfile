FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl

RUN pip install --no-cache-dir poetry==1.6.1

RUN poetry config virtualenvs.create false

RUN poetry install --no-interaction --no-ansi --no-root

COPY . .

ENV PYTHONPATH=/app/src:$PYTHONPATH

CMD ["python", "src/main.py"]
