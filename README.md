# GetAround – Delay Analysis, Pricing ML & API Deployment

## Contexte

GetAround est une plateforme de location de voitures entre particuliers (l’« Airbnb des voitures »).
Lorsqu’une location se termine en retard, le locataire suivant peut être bloqué.
Ce projet vise à aider l’équipe Produit à :

1. **Réduire les conflits entre locations** via un *buffer* (délai minimal entre deux locations)
2. **Optimiser les prix** avec un modèle de Machine Learning
3. **Déployer une API** permettant de prédire le prix d’un véhicule
4. **Fournir un dashboard Streamlit** facilitant la prise de décision

Projet réalisé dans le cadre du **bootcamp Data FullStack – Jedha** (bloc Deployment).

---

# Objectifs du projet

### 1. Delay Analysis (Analyse des retards)

* Étudier les retards au checkout
* Identifier les **conflits réels** (retard > temps prévu avant la prochaine location)
* Simuler plusieurs buffers (0h → 4h)
* Mesurer :

  * % de locations bloquées
  * % de conflits résolus
  * Impact business
* Servir ces résultats via un **dashboard Streamlit interactif**

### 2. Pricing ML (Modèle de prédiction du prix / jour)

* Entraîner un modèle supervisé (Random Forest)
* Prétraitement via `ColumnTransformer`
* Comparer LinearRegression vs RandomForestRegressor
* Sélection du **Random Forest** (meilleure performance)

### 3. Déploiement API FastAPI

* Endpoint `/predict`
* Input : caractéristiques du véhicule
* Output : prédiction du prix / jour
* API robuste, validations, gestion des erreurs
* Prête pour **Hugging Face Spaces** ou tout hébergement serverless

---

# Arborescence du projet

```
getaround_project/
│
├─ data/
│   ├─ raw/
│   │   ├─ get_around_delay_analysis.xlsx
│   │   └─ get_around_pricing_project.csv
│   └─ processed/
│       └─ buffer_scenarios.csv
│
├─ notebooks/
│   ├─ 01_delay_analysis_eda.ipynb
│   └─ 02_pricing_ml_model.ipynb
│
├─ app/
│   ├─ api/
│   │   ├─ main.py       # API FastAPI avec /predict
│   │   └─ model.joblib  # Pipeline ML sauvegardé
│   └─ dashboard/
│       └─ app_streamlit.py
│
├─ docs/
│   └─ api_documentation.md
│
├─ requirements.txt
└─ README.md
```

---

# 1 — Delay Analysis (Résumé)

## Insights clés (issus du notebook n°1)

### Retards

* ~44% des locations présentent un retard positif
* Retards allant d’environ **-200 min à +800 min**

### Définition métier du conflit

Un **conflit réel** survient lorsque :

```
delay_at_checkout_min > time_between_rentals_min
```

→ Ce sont les cas où le retard du locataire **empiète réellement** sur la prochaine location.

### Simulation de buffers (0–4 h)

Nous calculons, pour chaque buffer :

* % de locations bloquées
* % de conflits résolus
* Impact potentiel sur le revenu (si fourni)

### Exemple de logique observée :

* **Buffer 1h** → faible perte business, résout une partie significative des conflits
* **Buffer 2h** → résout encore plus de conflits mais bloque plus de locations
* Les véhicules **Connect** sont plus sensibles aux retards

---

# Dashboard Streamlit

Un tableau de bord aide le Product Manager à simuler l’impact du buffer :

* Filtre : buffer (0–4 h)
* Filtre : périmètre (toutes voitures / Connect seulement)
* Affiche :

  * Locations bloquées
  * Conflits résolus
  * Conflits observés
  * Bar chart de répartition

### Lancer le dashboard :

```bash
streamlit run app/dashboard/app_streamlit.py
```

---

# 2 — Modèle de Pricing ML (Résumé)

## Pipeline ML (notebook n°2)

### Prétraitement

* Imputation médiane (numérique)
* Imputation mode (catégorielle)
* OneHotEncoder pour les catégories
* ColumnTransformer + Pipeline

### Modèles testés

| Modèle            | MAE       | RMSE      | R²        |
| ----------------- | --------- | --------- | --------- |
| Linear Regression | ~13       | ~18.8     | ~0.66     |
| **Random Forest** | **~11.7** | **~17.8** | **~0.69** |

**Random Forest choisi** (meilleure MAE/RMSE & R²)

### Export

Le pipeline (`preprocessor + regressor`) est sauvegardé dans :

```
app/api/model.joblib
```

Prêt pour l’API.

---

# 3 — API FastAPI

## Lancer l’API en local

```bash
uvicorn app.api.main:app --reload
```

Endpoints :

* `/` → message de bienvenue
* `/health` → vérification du statut
* `/predict` → prédiction de prix

---

## Exemple de requête `/predict`

### Via *curl*

```bash
curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "input": [
    {
      "mileage": 150000,
      "engine_power": 90,
      "fuel": "diesel",
      "paint_color": "black",
      "car_type": "sedan",
      "private_parking_available": 1,
      "has_gps": 0,
      "has_air_conditioning": 1,
      "automatic_car": 0,
      "has_getaround_connect": 1,
      "has_speed_regulator": 1,
      "winter_tires": 0
    }
  ]
}'
```

### Via Python

```python
import requests

payload = {
    "input": [{
        "mileage": 150000,
        "engine_power": 90,
        "fuel": "diesel",
        "paint_color": "black",
        "car_type": "sedan",
        "private_parking_available": 1,
        "has_gps": 0,
        "has_air_conditioning": 1,
        "automatic_car": 0,
        "has_getaround_connect": 1,
        "has_speed_regulator": 1,
        "winter_tires": 0
    }]
}

res = requests.post("http://127.0.0.1:8000/predict", json=payload)
print(res.json())
```

---

# Déploiement HuggingFace (guide)

1. Créer un **Space** → choisir template **FastAPI**
2. Uploader :

   * `app/api/main.py`
   * `app/api/model.joblib`
   * `requirements.txt`
3. HuggingFace démarre automatiquement :

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 7860
   ```
4. Tester l’API depuis :

   ```
   https://<username>-<space>.hf.space/predict
   ```