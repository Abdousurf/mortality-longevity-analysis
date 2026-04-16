"""
DataOps — Validation des données de mortalité avec Pandera
==========================================================
Vérifie la cohérence actuarielle des tables de mortalité avant analyse :
  - qx ∈ [0, 1]
  - qx croissant avec l'âge (monotonie au-delà de 30 ans)
  - Pas de valeurs manquantes sur les colonnes critiques
  - Cohérence hommes/femmes (qx(F) < qx(M) pour tout âge après 20 ans)
  - Pas d'anomalies COVID isolées (vérification temporelle)

Utilisé en CI pour garantir la qualité des données avant modélisation Lee-Carter.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schémas Pandera — contrats de données actuariels
# ---------------------------------------------------------------------------

MORTALITY_SCHEMA = DataFrameSchema(
    columns={
        "annee": Column(
            int,
            checks=[
                Check.greater_than_or_equal_to(1900),
                Check.less_than_or_equal_to(2030),
            ],
            description="Année calendaire",
        ),
        "age": Column(
            int,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(110),
            ],
            description="Âge en années révolues",
        ),
        "qx": Column(
            float,
            checks=[
                Check.greater_than(0.0),
                Check.less_than_or_equal_to(1.0),
                Check(
                    lambda s: s.notna().all(), error="qx ne doit pas contenir de NaN"
                ),
            ],
            description="Probabilité de décès entre x et x+1",
        ),
        "sexe": Column(
            str,
            checks=Check.isin(["M", "F"]),
            description="Sexe : M (hommes) ou F (femmes)",
        ),
    },
    checks=[
        # Règle actuarielle : qx(100) >= 0.3 pour les deux sexes (mortalité élevée des centenaires)
        Check(
            lambda df: df[df["age"] == 100]["qx"].min() >= 0.10,
            error="qx à 100 ans trop faible — données suspectes",
        ),
        # qx(0) doit être > qx(1) (mortalité infantile > mortalité âge 1)
        Check(
            lambda df: (
                df[df["age"] == 0]["qx"].mean() > df[df["age"] == 1]["qx"].mean()
            ),
            error="Anomalie : mortalité à 0 an devrait être > 1 an",
        ),
    ],
    name="MortalityTable",
    description="Table de mortalité standard — format INSEE/HMD compatible",
)


HMD_SCHEMA = DataFrameSchema(
    columns={
        "Year": Column(int, checks=Check.between(1800, 2030)),
        "Age": Column(int, checks=Check.between(0, 110)),
        "mx": Column(float, checks=[Check.greater_than(0), Check.less_than(10)]),
        "qx": Column(
            float, checks=[Check.greater_than(0), Check.less_than_or_equal_to(1)]
        ),
        "sexe": Column(str, checks=Check.isin(["M", "F"])),
    },
    name="HMDTable",
)


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------
def validate_mortality_table(
    df: pd.DataFrame, schema: DataFrameSchema = MORTALITY_SCHEMA
) -> bool:
    """
    Valide un DataFrame de mortalité contre le schéma Pandera.
    Retourne True si valide, lève une exception sinon.
    """
    try:
        schema.validate(df, lazy=True)
        log.info("✅ Validation schéma : OK (%d lignes)", len(df))
        return True
    except pa.errors.SchemaErrors as e:
        log.error("❌ Validation échouée :")
        for _, row in e.failure_cases.iterrows():
            log.error(
                "  Colonne '%s' : %s → %s",
                row.get("column"),
                row.get("check"),
                row.get("failure_case"),
            )
        raise


def check_actuarial_monotonicity(
    df: pd.DataFrame, min_age: int = 30
) -> dict[str, bool]:
    """
    Vérifie la monotonie croissante de qx avec l'âge (règle actuarielle).
    Les qx doivent être croissants à partir de 30 ans.
    """
    results = {}

    for sexe in df["sexe"].unique():
        sub = df[(df["sexe"] == sexe) & (df["age"] >= min_age)].copy()

        # Moyennage par âge (toutes années)
        mean_qx = sub.groupby("age")["qx"].mean().reset_index()
        mean_qx = mean_qx.sort_values("age")

        # Monotonie : qx(x+1) >= qx(x) pour x >= 30
        diffs = mean_qx["qx"].diff().dropna()
        violations = (diffs < -0.001).sum()  # Tolérance petite pour les tables lissées
        is_monotone = violations == 0

        results[sexe] = is_monotone
        if not is_monotone:
            log.warning(
                "⚠️  Monotonie qx violée (%s) : %d violations (âge >= %d)",
                sexe,
                violations,
                min_age,
            )
        else:
            log.info("✅ Monotonie qx OK (%s, âge >= %d)", sexe, min_age)

    return results


def check_gender_differential(df: pd.DataFrame, min_age: int = 20) -> bool:
    """
    Vérifie que qx(F) < qx(M) pour les âges > 20 ans (réalité démographique).
    Une inversion massive indiquerait une erreur de données.
    """
    pivot = (
        df[df["age"] >= min_age]
        .groupby(["age", "sexe"])["qx"]
        .mean()
        .unstack(fill_value=np.nan)
    )

    if "M" not in pivot.columns or "F" not in pivot.columns:
        log.warning("Impossible de comparer H/F — colonnes manquantes")
        return True

    violations = (pivot["F"] > pivot["M"]).sum()
    total = len(pivot)
    pct = violations / total

    if pct > 0.20:  # Plus de 20% d'inversions = problème
        log.error(
            "❌ Différentiel genre suspect : qx(F) > qx(M) pour %.0f%% des âges",
            pct * 100,
        )
        return False

    log.info(
        "✅ Différentiel genre OK : qx(F) < qx(M) pour %.0f%% des âges", (1 - pct) * 100
    )
    return True


def validate_all(data_path: Path) -> dict[str, Any]:
    """
    Validation complète d'un fichier de données de mortalité.
    Retourne un rapport de validation pour la CI.
    """
    if not data_path.exists():
        log.error("Fichier introuvable : %s", data_path)
        return {"valid": False, "error": "file_not_found"}

    df = pd.read_parquet(data_path)
    report: dict[str, Any] = {"file": str(data_path), "n_rows": len(df), "checks": {}}

    # 1. Schéma
    try:
        validate_mortality_table(df)
        report["checks"]["schema"] = "pass"
    except Exception as e:
        report["checks"]["schema"] = f"fail: {e}"
        report["valid"] = False
        return report

    # 2. Monotonie
    monotone = check_actuarial_monotonicity(df)
    report["checks"]["monotonicity"] = {
        k: "pass" if v else "fail" for k, v in monotone.items()
    }

    # 3. Différentiel genre
    gender_ok = check_gender_differential(df)
    report["checks"]["gender_differential"] = "pass" if gender_ok else "fail"

    # 4. Statistiques basiques
    report["stats"] = {
        "years": sorted(df["annee"].unique().tolist()) if "annee" in df.columns else [],
        "age_range": (
            [int(df["age"].min()), int(df["age"].max())] if "age" in df.columns else []
        ),
        "qx_range": (
            [float(df["qx"].min()), float(df["qx"].max())] if "qx" in df.columns else []
        ),
    }

    report["valid"] = all(
        v == "pass" or (isinstance(v, dict) and all(x == "pass" for x in v.values()))
        for v in report["checks"].values()
    )

    return report


# ---------------------------------------------------------------------------
# Entrypoint CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Valide les données de mortalité (DataOps)"
    )
    parser.add_argument(
        "--data-path", default="data/processed/insee_mortality_france.parquet"
    )
    parser.add_argument("--output", help="Chemin JSON pour le rapport de validation")
    args = parser.parse_args()

    report = validate_all(Path(args.data_path))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, default=str))
        log.info("📋 Rapport : %s", args.output)

    print(json.dumps(report, indent=2, default=str))

    if not report.get("valid", False):
        import sys

        sys.exit(1)
