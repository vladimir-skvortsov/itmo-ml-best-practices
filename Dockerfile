FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install -y git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir "poetry>=2.0.0"

RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-interaction --no-ansi --no-root

COPY . .

ENV PYTHONPATH=/app/src:$PYTHONPATH

RUN mkdir -p data/raw data/processed data/interim data/external models mlruns reports/figures

EXPOSE 5000

# TODO: Replace with a real script when added
CMD ["/bin/bash"]
