# Documentation technique & fonctionnelle
## Mortality & Longevity Analysis

---

## Table des matières

1. [Vue d'ensemble fonctionnelle](#1-vue-densemble-fonctionnelle)
2. [Architecture de l'analyse](#2-architecture-de-lanalyse)
3. [Étape 1 — Construction des tables de mortalité](#3-étape-1--construction-des-tables-de-mortalité)
4. [Étape 2 — Modèle de Lee-Carter](#4-étape-2--modèle-de-lee-carter)
5. [Étape 3 — Projection stochastique](#5-étape-3--projection-stochastique)
6. [Étape 4 — Stress test Solvabilité II](#6-étape-4--stress-test-solvabilité-ii)
7. [Étape 5 — Visualisations](#7-étape-5--visualisations)
8. [Résultats clés](#8-résultats-clés)
9. [Glossaire actuariel mortalité](#9-glossaire-actuariel-mortalité)

---

## 1. Vue d'ensemble fonctionnelle

### Contexte métier

Le **risque de longévité** est l'un des risques systémiques majeurs pour les assureurs vie, les fonds de pension et les fournisseurs de rentes. Il se définit comme :

> "Le risque que les assurés vivent plus longtemps que prévu dans les tables de mortalité utilisées pour tarifier et réserver."

```
Exemple concret :
  Un portefeuille de rentes viagères de €100M est réservé sur des tables
  projetant e₆₅ (espérance de vie à 65 ans) = 20 ans.
  
  Si la réalité est e₆₅ = 22 ans (+2 ans)
  → les rentes durent 10% plus longtemps que prévu
  → réserve sous-estimée de ~€8–12M
  → risque de faillite si l'assureur n'a pas de capital suffisant (SCR)
```

### Ce que fait ce projet

1. **Construit** des tables de mortalité à partir de données brutes (méthode actuarielle standard)
2. **Calibre** le modèle de Lee-Carter (standard industrie) sur les données françaises 1968–2022
3. **Projette** la mortalité sur 25 ans avec intervalles de confiance (fan charts)
4. **Quantifie** l'impact d'un choc de longévité selon les normes Solvabilité II
5. **Visualise** les tendances pour le data storytelling actuariel

### Utilisateurs cibles

| Profil | Ce qu'il utilise |
|--------|-----------------|
| Actuaire vie/retraite | Tables qx, modèle Lee-Carter, stress tests SII |
| Analyste données (fonds de pension) | Projections e₀, e₆₅ |
| Risk Manager | SCR proxy, impact réserve |
| Data Scientist | Code Python reproductible, notebooks exécutables |

---

## 2. Architecture de l'analyse

```
Source : Human Mortality Database (HMD) — France
         INSEE — statistiques population
         FFSA — tables de référence (TF00-02)
               │
               ▼
┌──────────────────────────────┐
│  Étape 1 : Tables de mortalité│
│  raw_qx()                    │
│  whittaker_henderson()        │   ← lissage actuariel
│  build_life_table()           │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Étape 2 : Lee-Carter        │
│  LeeCarter.fit()             │   ← décomposition SVD
│  α_x, β_x, κ_t              │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Étape 3 : Projection        │
│  project_kappa()             │   ← marche aléatoire
│  project_qx()                │   ← 1000 simulations Monte Carlo
│  Fan charts 2023–2047        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Étape 4 : Stress Solv. II   │
│  longevity_shock_impact()    │   ← choc −20% mortalité
│  annuity_present_value()     │
│  SCR proxy, ΔRéserve         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Étape 5 : Visualisations    │
│  Diagrammes de Lexis         │
│  Fan charts                  │
│  Tables de résultats         │
└──────────────────────────────┘
```

**Stack :**

| Outil | Usage |
|-------|-------|
| NumPy + SciPy | Calculs matriciels, SVD, optimisation |
| Pandas | Manipulation des matrices mortalité (ages × années) |
| Statsmodels | ARIMA, régressions |
| Matplotlib + Seaborn | Graphiques publication-quality |
| Plotly | Charts interactifs |
| Jupyter | Notebooks exécutables |

---

## 3. Étape 1 — Construction des tables de mortalité

**Fichier :** `src/mortality_tables.py`

### Notation actuarielle

```
qₓ  = probabilité de décès entre l'âge x et x+1
pₓ  = 1 - qₓ = probabilité de survie
lₓ  = effectif de survivants à l'âge x (radix l₀ = 100 000)
dₓ  = lₓ - l(x+1) = nombre de décès entre x et x+1
Lₓ  = années-personnes vécues entre x et x+1
Tₓ  = Σ Lₓ (total années-personnes au-dessus de x)
eₓ  = Tₓ / lₓ = espérance de vie complète à l'âge x
μₓ  = force de mortalité (taux instantané)
```

### `raw_qx()` — Taux de mortalité bruts

```python
def raw_qx(deaths: np.ndarray, exposures: np.ndarray) -> np.ndarray:
    # Taux central de mortalité
    mx = deaths / exposures  # décès / exposition au risque (en années-personnes)

    # Conversion taux central → taux initial (hypothèse de Balducci)
    qx = mx / (1 + 0.5 * mx)
    return qx
```

**Hypothèse de Balducci :** suppose une distribution uniforme des décès dans l'intervalle [x, x+1]. C'est l'hypothèse standard des tables actuarielles réglementaires françaises.

### `whittaker_henderson_graduation()` — Lissage

Les taux bruts sont bruités (petits effectifs aux âges élevés). Le lissage Whittaker-Henderson est la méthode officielle de l'ISFA et des organismes actuariels pour produire des tables réglementaires.

**Problème d'optimisation :**

```
Minimiser : h × Σ(qₓ_lissé - qₓ_brut)² + (1-h) × Σ(Δ²qₓ_lissé)²
           ←── fidélité aux données ──→   ←──── régularité ────→
```

- `h = 0` → courbe parfaitement lisse (ignore les données)
- `h = 1` → courbe = données brutes (aucun lissage)
- `h = 0.1` → compromis standard (faible poids aux données, forte régularité)

```python
# Solution analytique : système linéaire
A = h * I + (1 - h) * D'D    # D = matrice de différences d'ordre 2
b = h * qx_raw
qx_smooth = solve(A, b)       # numpy.linalg.solve
```

### `build_life_table()` — Table de vie complète

À partir d'un vecteur qₓ, construit la table complète :

```python
# Initialisation : l₀ = 100 000
lx[0] = 100_000

# Récurrence : lx[i+1] = lx[i] × (1 - qx[i])
for i in range(n):
    lx[i+1] = lx[i] * (1 - qx[i])

# Décès : dx = lx - l(x+1)
# Années-personnes : Lx = 0.5 × (lx + l(x+1))  [trapèze]
# Total : Tx = Σ Lx (somme depuis la queue)
# Espérance de vie : ex = Tx / lx
```

**Exemple de sortie :**
```
Age |   qx    |   px    |   lx    |   dx   |   ex
 65 | 0.00891 | 0.99109 |  72,341 |   645  | 19.8
 70 | 0.01537 | 0.98463 |  67,528 | 1,038  | 15.3
 75 | 0.02943 | 0.97057 |  61,234 | 1,802  | 11.2
 80 | 0.05621 | 0.94379 |  51,108 | 2,873  |  7.7
```

---

## 4. Étape 2 — Modèle de Lee-Carter

**Fichier :** `src/mortality_tables.py` — Classe `LeeCarter`

### Formulation mathématique

```
ln(μₓ,ₜ) = αₓ + βₓ · κₜ + εₓ,ₜ

où :
  μₓ,ₜ  = taux central de mortalité à l'âge x, année t
  αₓ    = profil moyen de mortalité par âge (invariant dans le temps)
  βₓ    = sensibilité de l'âge x à la tendance temporelle
  κₜ    = indice temporel (driver des améliorations de mortalité)
  εₓ,ₜ  = terme d'erreur
```

### `LeeCarter.fit()` — Calibration par SVD

```python
def fit(self, mx_matrix: pd.DataFrame):
    # mx_matrix : DataFrame (ages × années), ex: (61 âges × 55 années)

    log_mx = np.log(mx_matrix.values)

    # Étape 1 : αₓ = moyenne en ligne (profil d'âge moyen)
    self.alpha = np.nanmean(log_mx, axis=1)

    # Étape 2 : Centrer la matrice
    Z = log_mx - self.alpha[:, np.newaxis]  # (ages × years)

    # Étape 3 : Décomposition SVD — extraire le 1er vecteur singulier
    U, s, Vt = svd(Z, full_matrices=False)
    # U[:, 0] = vecteur âge (direction principale)
    # Vt[0, :] = vecteur temps (direction principale)
    # s[0] = valeur singulière (magnitude)

    # Étape 4 : Normalisation (contrainte identifiabilité : Σβₓ = 1)
    b_raw = U[:, 0] * s[0]
    beta_sum = b_raw.sum()
    self.beta = b_raw / beta_sum      # βₓ normalisé
    self.kappa = Vt[0, :] * beta_sum  # κₜ ajusté en conséquence

    # Étape 5 : Modèle ARIMA(0,1,0) sur κₜ
    # κₜ = κₜ₋₁ + drift + σ·ε   (marche aléatoire avec dérive)
    kappa_diff = np.diff(self.kappa)
    self._kappa_drift = kappa_diff.mean()   # dérive annuelle moyenne
    self._kappa_sigma = kappa_diff.std()    # volatilité
```

**Interprétation des paramètres :**

```
αₓ  → Courbe en U : mortalité élevée à la naissance, faible entre 10-30 ans,
       croissante exponentiellement après 50 ans (loi de Gompertz)

βₓ  → Sensibilité aux améliorations : plus élevée aux âges 70-85
       (les progrès médicaux bénéficient plus aux âges élevés)

κₜ  → Tendance déclinante (−) = amélioration de la mortalité
       Pente annuelle ≈ −1.5 à −2.5 pour la France
```

---

## 5. Étape 3 — Projection stochastique

**Fichier :** `notebooks/03_lee_carter_model.py`

### `project_kappa()` — Marche aléatoire de κₜ

```python
def project_kappa(self, n_years: int, n_simulations: int = 1000):
    kappa_last = self.kappa[-1]  # dernière valeur observée

    for sim in range(n_simulations):
        # Chocs aléatoires N(drift, sigma)
        shocks = rng.normal(self._kappa_drift, self._kappa_sigma, n_years)
        # Trajectoire : somme cumulée des chocs
        simulations[sim] = kappa_last + np.cumsum(shocks)

    return simulations  # shape: (1000, 25)
```

1000 trajectoires de κₜ sur 25 ans → 1000 scénarios de mortalité future.

### `project_qx()` — Distribution des qₓ futurs

```python
# Pour chaque simulation et chaque année future :
log_mu = alpha + beta * kappa_sim[t]   # modèle LC appliqué
qx = exp(log_mu) / (1 + 0.5 * exp(log_mu))  # Balducci

# Central = moyenne des 1000 simulations
central_qx = exp(all_log_mu.mean(axis=0))

# Intervalles de confiance à 95%
lower_qx = exp(np.quantile(all_log_mu, 0.025, axis=0))
upper_qx = exp(np.quantile(all_log_mu, 0.975, axis=0))
```

**Sortie :** 3 matrices (central, lower, upper) de dimension (61 âges × 25 années)

### Fan chart — Lecture

```
q₇₀ (probabilité de décès à 70 ans)

  1.8% ─────────────────────────
  1.5%      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   ← 95% CI
  1.2%   ▓▓▓▓▓▓▓▓████████▓▓▓
  1.0%  ─────────██████████──── ← Estimation centrale
  0.8%         ▓▓▓▓▓▓▓▓▓▓▓▓▓
        2023   2030   2040  2047

  ██ = estimation centrale
  ▓▓ = intervalle de confiance 95%
  L'écartement croissant reflète l'incertitude grandissante sur le long terme.
```

---

## 6. Étape 4 — Stress test Solvabilité II

**Fichier :** `src/mortality_tables.py`

### Contexte réglementaire

La directive **Solvabilité II** (Article 142, Actes délégués) impose aux assureurs de calculer un **SCR (Solvency Capital Requirement)** pour le risque de longévité. Le stress standard est :

> "Appliquer une réduction permanente de **20%** des taux de mortalité qₓ."

### `annuity_present_value()` — Valeur actualisée de la rente

```python
def annuity_present_value(qx, interest_rate, start_age):
    # pₓ = 1 - qₓ
    px = 1 - qx
    n = len(px)

    # Probabilité de survie de t années : ₜpₓ = p₆₅ × p₆₆ × ... × p₆₄₊ₜ
    tpx = np.cumprod(np.concatenate([[1.0], px]))[:n]

    # Facteur d'actualisation : v^t = (1/(1+i))^t
    v = 1 / (1 + interest_rate)   # ex: i = 3%
    vt = v ** np.arange(n)        # [1, 0.971, 0.943, ...]

    # Valeur actualisée de la rente viagère : ä_x = Σ v^t × t_p_x
    return float(np.sum(vt * tpx))
```

**Interprétation :** `ä₆₅ = 14.2` signifie qu'un euro de rente annuelle pour un assuré de 65 ans vaut €14.2 aujourd'hui (en prenant en compte la probabilité de décès et l'actualisation à 3%).

### `longevity_shock_impact()` — Quantification du SCR

```python
def longevity_shock_impact(qx_base, shock_factor=0.80, interest_rate=0.03):

    # Scénario stressé : mortalité réduite de 20%
    qx_stressed = qx_base * shock_factor   # shock_factor = 0.80

    # Calculer les 2 valeurs de rente
    apv_base     = annuity_present_value(qx_base, interest_rate, 65)
    apv_stressed = annuity_present_value(qx_stressed, interest_rate, 65)

    # SCR proxy = surplus de réserve nécessaire
    scr_proxy = apv_stressed - apv_base

    # Impact en %
    impact_pct = (apv_stressed - apv_base) / apv_base
```

### Exemple de résultat

```
── Solvency II Longevity Stress (Males, age 65) ─────
  Annuity ä₆₅ (base):      14.2150
  Annuity ä₆₅ (stressed):  14.9230
  Reserve increase:         +4.98%
  SCR proxy per €1 annuity: 0.7080

  → Un portefeuille de rentes de €100M requiert
    ~€5M de capital additionnel sous le choc de longévité.
```

**Lecture :**
- Sans stress : €100M de primes reserves suffisent pour les €100M de rentes
- Avec stress SII (+20% survie) : il faut €105M → €5M de capital supplémentaire = SCR longévité

---

## 7. Étape 5 — Visualisations

**Notebook :** `notebooks/03_lee_carter_model.py`

### Graphique 1 — Paramètres Lee-Carter

3 sous-graphiques côte à côte :
- **αₓ** : profil de mortalité moyen par âge — courbe en J caractéristique
- **βₓ** : sensibilité temporelle par âge — pic aux âges 70–80
- **κₜ** : indice temporel 1968–2022 — déclin monotone = amélioration continue

Hommes vs Femmes superposés → visualise l'écart de mortalité entre sexes.

### Graphique 2 — Fan Chart

```python
# Remplissage de l'intervalle de confiance
ax.fill_between(central.index, lower, upper, alpha=0.25, color=color, label="95% CI")
# Courbe centrale
ax.plot(central.index, central, linewidth=2.5, label="Central estimate")
# Historique (dernières 10 années)
ax.plot(hist_qx.index, hist_qx, linestyle="--", color="gray", label="Historical")
```

Le fan chart est le standard de communication du risque de longévité dans les rapports ORSA et les présentations aux Conseils d'Administration.

### Graphique 3 — Espérance de vie à 65 ans

Projection e₆₅ centrale sur 25 ans pour hommes et femmes.
- Annotation automatique de la valeur terminale (2047)
- Grille légère pour faciliter la lecture

### Tableau récapitulatif — Amélioration de la mortalité

```
Age  | qx 1990  | qx 2022  | qx 2047 | Amél. 1990-2022 | Amél. 2022-2047
 65  | 0.01823  | 0.00891  | 0.00512 |     −51.1%      |    −42.5%
 70  | 0.03142  | 0.01537  | 0.00891 |     −51.1%      |    −42.0%
 75  | 0.05801  | 0.02943  | 0.01743 |     −49.3%      |    −40.8%
 80  | 0.10234  | 0.05621  | 0.03421 |     −45.1%      |    −39.2%
```

---

## 8. Résultats clés

### Tendances observées (France 1968–2022)

- Espérance de vie à 65 ans : **+4.2 ans** pour les hommes, **+3.8 ans** pour les femmes depuis 1990
- Taux d'amélioration annuel de la mortalité : **0.8–1.2%**
- Effets de cohorte marqués pour les générations 1940–1950 (bénéficiaires des soins post-WWII)

### Projections Lee-Carter (2023–2047)

| Indicateur | Hommes 2025 | Hommes 2047 | Variation |
|-----------|-------------|-------------|-----------|
| e₀ (naissance) | 80.2 | 84.1 | +3.9 ans |
| e₆₅ (à 65 ans) | 19.8 | 22.3 | +2.5 ans |
| q₇₀ | 1.54% | 0.89% | −42% |

### Impact Solvabilité II

- Choc standard −20% mortalité → **+4.5–5.0% de réserve** sur un portefeuille de rentes
- SCR longévité typique : **€4–6M pour €100M** de portefeuille de rentes viagères

### COVID-19

- Surmortalité 2020 : **+8.3%** par rapport à la tendance (données INSEE)
- Rattrapage partiel en 2021–2022 : l'effet COVID est transitoire dans le modèle Lee-Carter
- Ne remet pas en cause la tendance longue d'amélioration

---

## 9. Glossaire actuariel mortalité

| Terme | Définition |
|-------|-----------|
| **qₓ** | Probabilité de décès dans l'année pour une personne d'âge exact x |
| **eₓ** | Espérance de vie complète à l'âge x (en années) |
| **Table de mortalité** | Tableau donnant qₓ (ou lₓ) pour chaque âge. Base de tarification et de provisionnement |
| **Table périodique** | Table construite à partir d'une période d'observation (ex : 2019–2021) |
| **Table générationnelle** | Table tenant compte des améliorations futures (utilisée pour les rentes longues) |
| **Modèle de Lee-Carter** | Modèle stochastique de projection de mortalité (Lee & Carter, 1992). Standard mondial |
| **SVD** | Singular Value Decomposition — méthode de factorisation matricielle utilisée pour calibrer Lee-Carter |
| **Marche aléatoire** | Modèle ARIMA(0,1,0) : κₜ = κₜ₋₁ + drift + bruit → trajectoires divergentes dans le temps |
| **Fan chart** | Représentation graphique des projections avec intervalles de confiance (forme d'éventail) |
| **Solvabilité II** | Directive européenne réglementant le capital des assureurs. En vigueur depuis 2016 |
| **SCR** | Solvency Capital Requirement — capital minimum réglementaire (VaR 99.5% à 1 an) |
| **Choc de longévité SII** | Réduction permanente de 20% des qₓ → calcul du surplus de réserve nécessaire |
| **ä_x** | Valeur actualisée de la rente viagère — somme des paiements futurs pondérés par la probabilité de survie et actualisés |
| **ORSA** | Own Risk and Solvency Assessment — rapport d'auto-évaluation des risques (Solvabilité II, Art. 45) |
| **Risque de longévité** | Risque que les assurés vivent plus longtemps que prévu → sous-provisionnement |
| **Effet cohorte** | Différences de mortalité entre générations dues à des événements historiques (guerre, épidémie, baby-boom) |
| **Graduation** | Lissage statistique des taux de mortalité bruts pour éliminer le bruit stochastique |
| **Whittaker-Henderson** | Méthode standard de graduation (compromis fidélité/régularité) utilisée par les autorités actuarielles |
