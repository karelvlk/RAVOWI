# Base Image
FROM python:3.11.9

RUN set -eux; \
  apt-get update && apt-get install -y --no-install-recommends \
  curl \
  libgl1-mesa-glx \
  libglib2.0-0 \
  procps \
  nginx; \
  apt-get clean; \
  rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.8.2
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH "/root/.local/bin:$PATH"

# Disable poetry virtualenvs
RUN poetry config virtualenvs.create false

# Copy the project files
WORKDIR /meteval

COPY poetry.lock pyproject.toml ./
RUN poetry lock && poetry install --no-root --no-cache --without dev --no-interaction --no-ansi
COPY . .

COPY ./nginx/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 3000

RUN chmod +x /meteval/docker_startup.sh
# hadolint ignore=DL3025

CMD ["sh", "/meteval/docker_startup.sh"]
