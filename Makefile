# ===========================================================================
# Mortality & Longevity Analysis — Makefile (DataOps / DevOps)
# ===========================================================================

.PHONY: help install data-download data-demo validate notebooks report quality clean

PYTHON      := python
NOTEBOOKS   := notebooks
REPORTS     := reports

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
install: ## Installe les dépendances + pre-commit
	pip install -r requirements.txt
	pip install pre-commit ruff black isort papermill pandera nbstripout
	pre-commit install
	@echo "✅ Environnement DataOps prêt"

# ---------------------------------------------------------------------------
# Open Data
# ---------------------------------------------------------------------------
data-download: ## Télécharge INSEE + WHO + HMD (open data)
	$(PYTHON) src/download_hmd_data.py

data-demo: ## Génère données démo (CI offline / test rapide)
	$(PYTHON) src/download_hmd_data.py --demo-only

# ---------------------------------------------------------------------------
# DataOps — Validation
# ---------------------------------------------------------------------------
validate: ## Valide toutes les tables de mortalité (Pandera + actuarial)
	@for f in data/processed/*.parquet; do \
		echo "🔍 Validation : $$f"; \
		$(PYTHON) src/validate_data.py --data-path "$$f" \
			--output "reports/validation/$$(basename $$f .parquet)_validation.json"; \
	done
	@echo "✅ Validation terminée"

# ---------------------------------------------------------------------------
# Notebooks (DataOps — Papermill)
# ---------------------------------------------------------------------------
notebooks-run: ## Exécute tous les notebooks avec Papermill
	mkdir -p $(REPORTS)/executed_notebooks
	@for nb in $(NOTEBOOKS)/0*.ipynb; do \
		echo "📓 $$nb"; \
		papermill "$$nb" "$(REPORTS)/executed_notebooks/$$(basename $$nb)" \
			-p CI_MODE false \
			-p DATA_PATH "data/processed/"; \
	done

notebooks-ci: ## Exécute les notebooks en mode CI (rapide)
	mkdir -p $(REPORTS)/executed_notebooks
	@for nb in $(NOTEBOOKS)/0*.ipynb; do \
		papermill "$$nb" "$(REPORTS)/executed_notebooks/$$(basename $$nb)" \
			-p CI_MODE true \
			--no-progress-bar \
			--kernel python3 || echo "⚠️ $$nb — skip"; \
	done

notebooks-strip: ## Supprime les outputs des notebooks (avant commit)
	nbstripout $(NOTEBOOKS)/*.ipynb
	@echo "🧹 Notebooks strippés"

# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------
report-html: ## Convertit notebooks exécutés en HTML
	mkdir -p $(REPORTS)/html
	@for nb in $(REPORTS)/executed_notebooks/0*.ipynb; do \
		jupyter nbconvert --to html "$$nb" --output-dir $(REPORTS)/html/; \
	done
	@echo "📄 Rapports HTML : $(REPORTS)/html/"

report-pdf: ## Convertit notebooks exécutés en PDF (requiert LaTeX)
	mkdir -p $(REPORTS)/pdf
	@for nb in $(REPORTS)/executed_notebooks/0*.ipynb; do \
		jupyter nbconvert --to pdf "$$nb" --output-dir $(REPORTS)/pdf/ || true; \
	done

# ---------------------------------------------------------------------------
# Jupyter Lab
# ---------------------------------------------------------------------------
lab: ## Lance Jupyter Lab
	jupyter lab --no-browser

# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------
pipeline: data-download validate notebooks-run report-html ## Pipeline DataOps complet
	@echo "✅ Pipeline mortalité terminé"

pipeline-ci: data-demo validate notebooks-ci ## Pipeline CI (sans téléchargement)

# ---------------------------------------------------------------------------
# DVC
# ---------------------------------------------------------------------------
dvc-repro: ## Reproduit le pipeline DVC complet
	dvc repro

dvc-dag: ## Affiche le DAG DVC
	dvc dag

dvc-metrics: ## Affiche les métriques (validation, etc.)
	dvc metrics show

# ---------------------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------------------
quality: ## Lint + format check + notebook strip check
	ruff check .
	black --check .
	isort --check .
	@echo "✅ Qualité code OK"

format: ## Formate le code
	black .
	isort .
	ruff check --fix .

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean: ## Nettoie les artefacts temporaires
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
	@echo "🧹 Nettoyage terminé"
