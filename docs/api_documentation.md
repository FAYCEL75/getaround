# GetAround Pricing API – Documentation

API de prédiction de prix journalier pour les véhicules GetAround.  
Hébergée sur **Hugging Face Spaces**, consommable depuis n'importe quel client HTTP (front-end, back-end, notebook).

L'API expose un modèle de Machine Learning (pipeline `sklearn`) qui estime un **prix de location en €/jour**
à partir des caractéristiques d'un véhicule (marque, kilométrage, carburant, équipements, etc.).

---

## Endpoints disponibles

### `GET /` – Root

- **Description** : message de bienvenue et rappel d'utilisation de l'API.
- **Input** : aucun paramètre.
- **Output** :

```json
{
  "message": "Bienvenue sur la GetAround Pricing API (Hugging Face Space).",
  "usage": "Utilisez POST /predict pour obtenir une prédiction de prix."
}