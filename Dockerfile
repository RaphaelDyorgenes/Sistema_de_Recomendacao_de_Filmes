# ── Stage 1: Builder ────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Instalar Poetry para exportar requirements
RUN pip install --no-cache-dir poetry==1.8.5

# Copiar apenas arquivos de dependência para cachear a camada
COPY pyproject.toml poetry.lock ./

# Exportar requirements.txt (sem dependências de dev)
RUN poetry export -f requirements.txt --without dev -o requirements.txt

# Instalar dependências em um venv isolado
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copiar venv do builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copiar código-fonte e configurações
COPY src/ src/
COPY configs/ configs/
COPY dvc.yaml ./
COPY .env.example .env
COPY pyproject.toml ./

# Criar diretórios de dados e modelos
RUN mkdir -p data/raw data/processed models

# Variáveis de ambiente padrão
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m"]
CMD ["recsys.training.train"]
