"""Script de validação do ambiente de desenvolvimento.

Verifica dependências essenciais, variáveis de ambiente e estrutura de pastas.
Retorna exit-code 0 se tudo estiver correto, 1 caso contrário.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Garante saída UTF-8 no Windows (evita erros com emojis no cp1252).
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

# ── Constantes ──────────────────────────────────────────────────────────────

REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("torch", "PyTorch"),
    ("sklearn", "Scikit-Learn"),
    ("mlflow", "MLflow"),
    ("pydantic_settings", "Pydantic Settings"),
    ("pandas", "Pandas"),
    ("numpy", "NumPy"),
    ("kagglehub", "KaggleHub"),
]

REQUIRED_DIRS: list[str] = [
    "src/recsys",
    "tests",
    "data/raw",
    "data/processed",
    "models",
    "configs",
]


def _find_project_root() -> Path:
    """Localiza a raiz do projeto a partir do ``pyproject.toml``."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: dois níveis acima de scripts/
    return Path(__file__).resolve().parent.parent


def _check_packages() -> list[str]:
    """Verifica se os pacotes obrigatórios estão instalados."""
    errors: list[str] = []
    for module_name, display_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module_name)
            print(f"  ✅ {display_name}")
        except ImportError:
            errors.append(display_name)
            print(f"  ❌ {display_name} — não encontrado")
    return errors


def _check_env_file(root: Path) -> list[str]:
    """Verifica se o arquivo ``.env`` existe e contém as chaves mínimas."""
    errors: list[str] = []
    env_path = root / ".env"

    if not env_path.exists():
        errors.append(".env não encontrado")
        print("  ❌ Arquivo .env não encontrado na raiz do projeto")
        return errors

    print("  ✅ Arquivo .env encontrado")

    content = env_path.read_text(encoding="utf-8")
    required_keys = [
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "DATA_SAMPLE_SIZE",
        "RANDOM_SEED",
    ]
    for key in required_keys:
        if key in content:
            print(f"  ✅ {key} definido")
        else:
            errors.append(f"{key} ausente no .env")
            print(f"  ❌ {key} ausente no .env")
    return errors


def _check_directories(root: Path) -> list[str]:
    """Verifica se a estrutura de diretórios esperada existe."""
    errors: list[str] = []
    for rel in REQUIRED_DIRS:
        dirpath = root / rel
        if dirpath.is_dir():
            print(f"  ✅ {rel}/")
        else:
            errors.append(f"Diretório {rel}/ ausente")
            print(f"  ❌ {rel}/ — diretório ausente")
    return errors


def main() -> int:
    """Executa todas as validações e retorna o exit-code."""
    root = _find_project_root()
    all_errors: list[str] = []

    print(f"\n🔍 Validando ambiente em: {root}\n")

    print("── Pacotes Python ──")
    all_errors.extend(_check_packages())

    print("\n── Variáveis de Ambiente (.env) ──")
    all_errors.extend(_check_env_file(root))

    print("\n── Estrutura de Diretórios ──")
    all_errors.extend(_check_directories(root))

    if all_errors:
        print(f"\n⚠️  {len(all_errors)} problema(s) encontrado(s):")
        for err in all_errors:
            print(f"   • {err}")
        print()
        return 1

    print("\n✅ Ambiente validado com sucesso!\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
