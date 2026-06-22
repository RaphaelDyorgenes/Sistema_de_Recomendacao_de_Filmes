# Tech Challenge — Fase 02 · Sistema de Recomendação

Recomendação de produtos a partir do comportamento de navegação: rede neural
(PyTorch), pipeline reprodutível (DVC), tracking (MLflow) e clean code.

> 🚧 README completo será finalizado na Etapa 4.

## Estrutura
- `src/recsys/` — código-fonte (pacote instalável, src-layout)
  - `models/` — definições de modelos + Factory
  - `preprocessing/` — preprocessors (Strategy)
  - `data/`, `training/`, `evaluation/`, `utils/`
- `tests/` — testes
- `data/`, `models/`, `configs/` — dados, artefatos e configs

## 🚀 Preparando o Ambiente e Instalando Dependências

Este projeto utiliza o **Poetry** para o gerenciamento de dependências. Siga o passo a passo abaixo para rodar localmente:

### 1. Instale o Poetry
instale-o via pip:
```bash
pip install poetry
```

### 2. Instale as dependências do projeto
Certificar-se de estar dentro da pasta raiz do repositório (`Tech-Challenge-02`) e rode:
```bash
poetry install
```
Isso criará o ambiente virtual isolado `.venv` e instalará pacotes essenciais como PyTorch, Scikit-Learn e MLflow.

### 3. Validação do ambiente
Após a instalação, execute o script de validação para ter certeza de que as dependências e arquivos críticos estão configurados corretamente:
```bash
poetry run python scripts/validate_env.py
```
*Se a mensagem for de sucesso, o seu ambiente está 100% pronto para uso!*
