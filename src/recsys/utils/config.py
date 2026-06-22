"""Configurações centralizadas do projeto, carregadas via variáveis de ambiente."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_project_root() -> Path:
    """Sobe a árvore de diretórios até encontrar ``pyproject.toml``.

    Returns:
        Caminho absoluto da raiz do projeto.

    Raises:
        FileNotFoundError: Se ``pyproject.toml`` não for encontrado.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    msg = "pyproject.toml não encontrado na árvore de diretórios."
    raise FileNotFoundError(msg)


PROJECT_ROOT = _find_project_root()


class Settings(BaseSettings):
    """Parâmetros globais do projeto, lidos do ``.env`` na raiz.

    Atributos com valor padrão funcionam mesmo sem o arquivo ``.env``,
    garantindo que ``pytest`` rode sem configuração extra.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Ambiente ────────────────────────────────────────────
    env: str = "dev"

    # ── MLflow ──────────────────────────────────────────────
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "recsys-movielens"

    # ── Dados ───────────────────────────────────────────────
    data_sample_size: int = 100_000
    random_seed: int = 42

    # ── Caminhos derivados (não vêm do .env) ────────────────
    @property
    def data_raw_dir(self) -> Path:
        """Diretório de dados brutos."""
        return PROJECT_ROOT / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        """Diretório de dados processados."""
        return PROJECT_ROOT / "data" / "processed"

    @property
    def models_dir(self) -> Path:
        """Diretório de artefatos de modelos."""
        return PROJECT_ROOT / "models"

    @property
    def configs_dir(self) -> Path:
        """Diretório de configurações externas."""
        return PROJECT_ROOT / "configs"


settings = Settings()
