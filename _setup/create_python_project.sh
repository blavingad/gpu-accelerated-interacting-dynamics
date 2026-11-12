#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: ./create_python_project.sh PROJECT_NAME [PARENT_DIRECTORY] [EXTRAS]

Examples:
  ./create_python_project.sh market-research-lab
  ./create_python_project.sh physical-ai-lab ~/run/applied_research
  ./create_python_project.sh quant-lab ~/run/applied_research dev,finance,ml

EXTRAS: comma-separated names from dev,ml,finance (default: dev; dev always included)
Requires Python 3.12, Git and an internet connection for dependency installation.
Does not create or push a remote GitHub repository.
EOF
}

if [[ $# -lt 1 || $# -gt 3 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && exit 0
  exit 2
fi

project="$1"
parent="${2:-$PWD}"
extras="${3:-dev}"

if [[ ! "$project" =~ ^[a-z][a-z0-9]*(-[a-z0-9]+)*$ ]]; then
  echo 'Error: project name must be lowercase kebab-case (e.g. physical-ai-lab).' >&2
  exit 2
fi

package="${project//-/_}"
IFS=',' read -r -a requested_extras <<< "$extras"
for extra in "${requested_extras[@]}"; do
  case "$extra" in
    dev|ml|finance) ;;
    *) echo "Error: unknown extra '$extra' (choose dev,ml,finance)." >&2; exit 2 ;;
  esac
done

if [[ ",$extras," != *",dev,"* ]]; then
  extras="dev,$extras"
fi

command -v python3.12 >/dev/null || { echo 'Error: python3.12 not found.' >&2; exit 1; }
command -v git >/dev/null || { echo 'Error: git not found.' >&2; exit 1; }

mkdir -p "$parent"
parent="$(cd "$parent" && pwd -P)"
target="$parent/$project"
if [[ -e "$target" ]]; then
  echo "Error: '$target' already exists; refusing to overwrite." >&2
  exit 1
fi

mkdir -p "$target"
cd "$target"
mkdir -p "src/$package" tests notebooks scripts docs data/raw data/processed
: > "src/$package/__init__.py"
: > data/raw/.gitkeep
: > data/processed/.gitkeep

cat > pyproject.toml <<EOF
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "$project"
version = "0.1.0"
description = "Scientific Python research project"
requires-python = ">=3.12,<3.13"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
    "pandas>=2.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-cov>=5",
    "ruff>=0.6",
    "ipykernel>=6",
    "jupyterlab>=4",
]
ml = [
    "scikit-learn>=1.4",
    "jax>=0.4.30",
]
finance = [
    "yfinance>=0.2.40",
    "matplotlib>=3.8",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
line-length = 88
target-version = "py312"
EOF

cat > .gitignore <<'EOF'
.venv/
venv/
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/
.ipynb_checkpoints/
.env
.env.*
!.env.example
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep
checkpoints/
outputs/
*.pt
*.pth
*.ckpt
.vscode/
.idea/
.DS_Store
EOF

cat > README.md <<EOF
# $project

Scientific Python research project.

## Installation

\`python3.12 -m venv .venv\`
\`source .venv/bin/activate\`
\`python -m pip install -e ".[dev]"\`

## Tests

\`python -m pytest -v\`
EOF

cat > tests/test_smoke.py <<EOF
import $package


def test_package_import():
    assert $package is not None
EOF

git init -b main
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e ".[${extras}]"
.venv/bin/python -m pytest -v
.venv/bin/python -m ruff check .

# An initial commit is only possible if Git identity is configured.
git add .
if git var GIT_AUTHOR_IDENT >/dev/null 2>&1 && git var GIT_COMMITTER_IDENT >/dev/null 2>&1; then
  git commit -m "Initial project structure"
else
  echo 'Git identity not configured; files staged, but no initial commit created.'
  echo 'Configure git user.name and user.email, then run git commit.'
fi

cat <<EOF

Project created at: $target
Package import: $package

To begin:
  cd "$target"
  source .venv/bin/activate
  python -m pytest -v

Optional Jupyter kernel registration (if dev extra installed):
  python -m ipykernel install --user --name "$project" --display-name "Python 3.12 ($project)"

To publish on GitHub: create an empty GitHub repository, then:
  git remote add origin git@github.com:YOUR_USERNAME/$project.git
  git push -u origin main
EOF
