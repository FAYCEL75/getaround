import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests
from pathlib import Path

# ======================================================
# ⚙️ Data loading & helpers
# ======================================================

@st.cache_data
def load_buffer_data() -> pd.DataFrame:
    """
    Load buffer_scenarios.csv from typical locations.
    """
    base_dir = Path(__file__).resolve().parents[2]

    candidate_paths = [
        base_dir / "data" / "processed" / "buffer_scenarios.csv",
        base_dir / "data" / "buffer_scenarios.csv",
        base_dir / "buffer_scenarios.csv",
    ]

    for p in candidate_paths:
        if p.exists():
            df_ = pd.read_csv(p)
            return df_

    raise FileNotFoundError(
        "Impossible de trouver buffer_scenarios.csv. "
        "Place-le dans data/processed/ ou adapte les chemins dans le code."
    )


@st.cache_data
def compute_recommendations(df_: pd.DataFrame):
    """
    Compute a recommended buffer per scope, with a simple rule:
      - conflicts_resolved >= 80%
      - blocked_ratio <= 7%
    If none satisfies it, pick argmax(conflicts_resolved - blocked).
    """
    recommendations = {}

    for scope in df_["scope"].unique():
        sub = df_[df_["scope"] == scope].copy()

        candidates = sub[
            (sub["conflicts_resolved_ratio"] >= 0.8)
            & (sub["blocked_ratio"] <= 0.07)
        ]
        if not candidates.empty:
            row = candidates.sort_values("buffer_hours").iloc[0]
        else:
            sub["score"] = sub["conflicts_resolved_ratio"] - sub["blocked_ratio"]
            row = sub.sort_values("score", ascending=False).iloc[0]

        recommendations[scope] = {
            "buffer_hours": int(row["buffer_hours"]),
            "blocked_ratio": float(row["blocked_ratio"]),
            "conflicts_resolved_ratio": float(row["conflicts_resolved_ratio"]),
        }

    return recommendations


def classify_scenario(blocked: float, resolved: float) -> str:
    """
    Classify scenario as 'optimal', 'acceptable' or 'risqué'.
    """
    if blocked <= 0.05 and resolved >= 0.8:
        return "optimal"
    if blocked <= 0.10 and resolved >= 0.6:
        return "acceptable"
    return "risqué"


# ======================================================
# 🌐 API Pricing HF
# ======================================================

API_URL = "https://faycel75-getaround-deployment.hf.space/predict"


def call_pricing_api(features: dict):
    """
    Appelle l'API FastAPI déployée sur Hugging Face.
    Retourne le prix prédit (float) ou None en cas d'erreur.
    """
    payload = {"input": [features]}
    try:
        resp = requests.post(API_URL, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "prediction" in data and len(data["prediction"]) > 0:
            return float(data["prediction"][0]), payload, data
        else:
            st.error("Réponse API inattendue : clé 'prediction' absente.")
            return None, payload, data
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API : {e}")
        return None, payload, None


# ======================================================
# 🎨 Page config & global style (FULL WIDTH, McKinsey-like)
# ======================================================

st.set_page_config(
    page_title="GetAround – Dashboard d'analyse & Pricing ML",
    page_icon="🚗",
    layout="wide",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background-color: #f3f4f6;
    }
    h1, h2, h3 {
        font-weight: 700 !important;
        color: #0f172a;
    }
    h1 {
        font-size: 2rem;
    }
    .kpi-card {
        background-color: #ffffff;
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(15,23,42,0.06);
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #6b7280;
    }
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 4px;
        color: #111827;
    }
    .kpi-status-optimal {
        color: #16a34a;
        font-weight: 600;
    }
    .kpi-status-acceptable {
        color: #ea580c;
        font-weight: 600;
    }
    .kpi-status-risky {
        color: #b91c1c;
        font-weight: 600;
    }
    .info-panel {
        background-color: #e5e7eb;
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px solid #d1d5db;
        font-size: 0.9rem;
        color: #111827;
    }
    .footer {
        font-size: 0.78rem;
        color: #9ca3af;
        margin-top: 40px;
        padding-top: 8px;
        border-top: 1px solid #d1d5db;
    }
    .executive-badge {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 999px;
        background-color: #0f172a;
        color: #e5e7eb;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    .price-card {
        background: linear-gradient(135deg, #0f172a, #1d4ed8);
        color: #f9fafb;
        padding: 20px 24px;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(15,23,42,0.5);
        margin-top: 10px;
    }
    .price-amount {
        font-size: 2.6rem;
        font-weight: 750;
        letter-spacing: 0.03em;
    }
    .price-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        background-color: rgba(15,23,42,0.85);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .price-sub {
        font-size: 0.9rem;
        color: #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# 📥 Load data & recommendations
# ======================================================

df = load_buffer_data()
recommendations = compute_recommendations(df)

buffer_values = sorted(df["buffer_hours"].unique())
scope_options = {
    "Toutes les voitures": "all",
    "Voitures Connect uniquement": "connect_only",
}

# ======================================================
# 🎛️ Sidebar – Mode + paramètres produit
# ======================================================

st.sidebar.header("⚙️ Navigation & Paramètres")

mode = st.sidebar.radio(
    "Vue",
    ["📊 Analyse du délai minimum", "💶 Pricing ML (API GetAround)"],
    index=0,
)

# Ces paramètres ne servent que pour la vue buffer, mais on peut les garder visibles
selected_scope_label = st.sidebar.radio(
    "Portée de la fonctionnalité",
    list(scope_options.keys()),
    index=0,
)

selected_scope = scope_options[selected_scope_label]

selected_buffer = st.sidebar.slider(
    "Seuil de délai minimum (heures)",
    min_value=int(min(buffer_values)),
    max_value=int(max(buffer_values)),
    value=int(2),
    step=1,
)

st.sidebar.info(
    "💡 Comparez 0h, 1h, 2h et 3h pour visualiser le compromis entre "
    "conflits résolus et locations bloquées."
)

st.sidebar.markdown(
    f"🔍 Scénario courant : **buffer {selected_buffer}h**",
)

# ======================================================
# 🧭 Header commun
# ======================================================

st.markdown(
    """
    <div class="executive-badge">Executive Analytics</div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    # 🚗 GetAround – Dashboard d'analyse du délai minimum & Pricing ML

    Ce tableau de bord permet :
    - d'ajuster le **délai minimum entre deux locations** pour limiter les conflits ;
    - d'estimer le **prix optimal** d'une location via un modèle de machine learning (API).
    """
)

st.markdown("---")

# ======================================================
# 🟦 MODE 1 : Analyse du délai minimum (ton dashboard actuel)
# ======================================================

if mode == "📊 Analyse du délai minimum":

    # === Current scenario ===
    scenario_mask = (df["buffer_hours"] == selected_buffer) & (
        df["scope"] == selected_scope
    )
    scenario = df.loc[scenario_mask].iloc[0]

    blocked_ratio = float(scenario["blocked_ratio"])
    revenue_blocked_ratio = (
        float(scenario["revenue_blocked_ratio"])
        if not np.isnan(scenario["revenue_blocked_ratio"])
        else np.nan
    )
    conflicts_resolved_ratio = float(scenario["conflicts_resolved_ratio"])
    conflict_ratio = float(scenario["conflict_ratio"])
    n_rentals = int(scenario["n_rentals"])

    status_label = classify_scenario(blocked_ratio, conflicts_resolved_ratio)
    status_text = {
        "optimal": "Configuration jugée OPTIMALE.",
        "acceptable": "Configuration ACCEPTABLE : compromis raisonnable entre conflits résolus et locations bloquées.",
        "risqué": "Configuration RISQUÉE : perte de CA importante ou réduction de conflits insuffisante.",
    }[status_label]

    status_css_class = {
        "optimal": "kpi-status-optimal",
        "acceptable": "kpi-status-acceptable",
        "risqué": "kpi-status-risky",
    }[status_label]

    # === Recommended scenario ===
    rec = recommendations[selected_scope]
    rec_buffer = rec["buffer_hours"]
    rec_blocked = rec["blocked_ratio"]
    rec_resolved = rec["conflicts_resolved_ratio"]
    rec_match = (selected_buffer == rec_buffer)

    exec_col1, exec_col2 = st.columns([2, 3])

    with exec_col1:
        st.subheader("🧭 Résumé exécutif")

        st.markdown(
            f"""
            - Scope sélectionné : **{selected_scope_label.lower()}**  
            - Buffer courant : **{selected_buffer} heure(s)**  
            - Conflits résolus : **{conflicts_resolved_ratio * 100:.1f} %**  
            - Locations bloquées : **{blocked_ratio * 100:.1f} %**  
            - Conflits sans buffer : **{conflict_ratio * 100:.1f} %**  
            """
        )
        st.markdown(
            f'<span class="{status_css_class}">➡ {status_text}</span>',
            unsafe_allow_html=True,
        )

    with exec_col2:
        st.subheader("⭐ Recommandation automatique")

        if rec_match:
            st.success(
                f"Vous êtes déjà sur le **buffer recommandé** pour le scope "
                f"**{selected_scope_label.lower()}** : **{rec_buffer} heure(s)**.\n\n"
                f"À ce seuil, environ **{rec_blocked * 100:.1f} %** des locations sont bloquées "
                f"et **{rec_resolved * 100:.1f} %** des conflits sont résolus."
            )
        else:
            st.warning(
                f"Pour le scope **{selected_scope_label.lower()}**, le buffer recommandé est "
                f"de **{rec_buffer} heure(s)**.\n\n"
                f"À ce seuil, environ **{rec_blocked * 100:.1f} %** des locations seraient bloquées "
                f"et **{rec_resolved * 100:.1f} %** des conflits seraient résolus."
            )

        st.markdown(
            """
            <div class="info-panel">
            <b>Logique utilisée :</b><br>
            • viser au moins <b>80&nbsp;%</b> de conflits résolus,<br>
            • tout en gardant moins de <b>7&nbsp;%</b> de locations bloquées.<br>
            Si aucun seuil ne satisfait ces conditions, on maximise <code>conflits_résolus − locations_bloquées</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ======================================================
    # 📌 KPI cards
    # ======================================================

    st.subheader("📌 Indicateurs pour le scénario sélectionné")

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Nombre total de locations (scope sélectionné)</div>
              <div class="kpi-value">{n_rentals:,}</div>
            </div>
            """.replace(",", " "),
            unsafe_allow_html=True,
        )

    with k2:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">% de locations bloquées</div>
              <div class="kpi-value">{blocked_ratio * 100:.1f} %</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k3:
        if np.isnan(revenue_blocked_ratio):
            ca_value = "Donnée non fournie"
            ca_caption = (
                "Les données sources ne contiennent pas le CA. "
                "Dans un contexte réel, il faudrait joindre une table revenue."
            )
        else:
            ca_value = f"{revenue_blocked_ratio * 100:.1f} %"
            ca_caption = "Part moyenne du chiffre d'affaires potentiellement bloqué."

        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">% du CA potentiellement impacté</div>
              <div class="kpi-value">{ca_value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(ca_caption)

    with k4:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">% de conflits résolus par le buffer</div>
              <div class="kpi-value">{conflicts_resolved_ratio * 100:.1f} %</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        f"Conflits observés **sans buffer** : environ **{conflict_ratio * 100:.1f} %** "
        "des locations créent un chevauchement problématique."
    )

    st.markdown("---")

    # ======================================================
    # 📊 Double-axis chart + vertical rule + star on recommended buffer
    # ======================================================

    st.subheader("📊 Impact du buffer sur les conflits et les locations bloquées")

    df_scope = df[df["scope"] == selected_scope].sort_values("buffer_hours").copy()
    df_scope["locations_bloquees_%"] = df_scope["blocked_ratio"] * 100
    df_scope["conflits_resolus_%"] = df_scope["conflicts_resolved_ratio"] * 100

    base = alt.Chart(df_scope).encode(
        x=alt.X("buffer_hours:O", title="Buffer (heures)")
    )

    line_resolved = base.mark_line(
        point=alt.OverlayMarkDef(filled=True, size=70),
        strokeWidth=3,
        color="#1d4ed8"
    ).encode(
        y=alt.Y(
            "conflits_resolus_%:Q",
            axis=alt.Axis(title="% de conflits résolus", titleColor="#1d4ed8"),
        ),
        tooltip=[
            alt.Tooltip("buffer_hours:O", title="Buffer (h)"),
            alt.Tooltip("conflits_resolus_%:Q", format=".1f", title="% conflits résolus"),
        ],
    )

    line_blocked = base.mark_line(
        point=alt.OverlayMarkDef(filled=True, size=70),
        strokeWidth=3,
        color="#f97316"
    ).encode(
        y=alt.Y(
            "locations_bloquees_%:Q",
            axis=alt.Axis(
                title="% de locations bloquées",
                titleColor="#f97316",
                orient="right",
            ),
        ),
        tooltip=[
            alt.Tooltip("buffer_hours:O", title="Buffer (h)"),
            alt.Tooltip("locations_bloquees_%:Q", format=".1f", title="% locations bloquées"),
        ],
    )

    selected_df = pd.DataFrame({"buffer_hours": [selected_buffer]})
    rule = alt.Chart(selected_df).mark_rule(
        color="#0f172a",
        strokeDash=[4, 4],
        strokeWidth=2,
    ).encode(
        x=alt.X("buffer_hours:O")
    )

    rec_df = pd.DataFrame({"buffer_hours": [rec_buffer], "y": [rec_resolved * 100]})
    star = alt.Chart(rec_df).mark_point(
        shape="star",
        size=200,
        color="#fbbf24",
    ).encode(
        x=alt.X("buffer_hours:O"),
        y=alt.Y("y:Q"),
        tooltip=[
            alt.Tooltip("buffer_hours:O", title="Buffer recommandé (h)"),
            alt.Tooltip("y:Q", format=".1f", title="% conflits résolus"),
        ],
    )

    double_axis_chart = (
        alt.layer(line_resolved, line_blocked, rule, star)
        .resolve_scale(y="independent")
        .properties(height=380)
    )

    st.altair_chart(double_axis_chart, use_container_width=True)

    # ======================================================
    # 🌡️ Heatmap de décision + tableau consulting
    # ======================================================

    st.subheader("🌡️ Carte de décision & tableau de synthèse")

    heat_df = df[df["scope"] == selected_scope].copy()
    heat_df["% conflits résolus"] = heat_df["conflicts_resolved_ratio"] * 100
    heat_df["% locations bloquées"] = heat_df["blocked_ratio"] * 100
    heat_df["score"] = heat_df["conflicts_resolved_ratio"] - heat_df["blocked_ratio"]

    heat_long = heat_df.melt(
        id_vars=["buffer_hours"],
        value_vars=["% conflits résolus", "% locations bloquées"],
        var_name="Métrique",
        value_name="Valeur",
    )

    heat_chart = (
        alt.Chart(heat_long)
        .mark_rect()
        .encode(
            x=alt.X("buffer_hours:O", title="Buffer (heures)"),
            y=alt.Y("Métrique:N", title=""),
            color=alt.Color(
                "Valeur:Q",
                scale=alt.Scale(scheme="blues"),
                title="Valeur (%)",
            ),
            tooltip=[
                alt.Tooltip("buffer_hours:O", title="Buffer (h)"),
                alt.Tooltip("Métrique:N"),
                alt.Tooltip("Valeur:Q", format=".1f", title="Valeur (%)"),
            ],
        )
        .properties(height=120)
    )

    c_heat, c_table = st.columns([2, 3])

    with c_heat:
        st.altair_chart(heat_chart, use_container_width=True)

    with c_table:
        st.markdown("**Tableau de synthèse (score consulting)**")
        summary_table = heat_df[["buffer_hours", "% locations bloquées", "% conflits résolus", "score"]].copy()
        summary_table = summary_table.rename(
            columns={
                "buffer_hours": "Buffer (h)",
                "% locations bloquées": "% loc. bloquées",
                "% conflits résolus": "% conflits résolus",
                "score": "Score (résolus − bloquées)",
            }
        )
        summary_table["Score (résolus − bloquées)"] = (summary_table["Score (résolus − bloquées)"] * 100).round(1)
        st.dataframe(
            summary_table.style.format(
                {
                    "% loc. bloquées": "{:.1f}",
                    "% conflits résolus": "{:.1f}",
                    "Score (résolus − bloquées)": "{:.1f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    # ======================================================
    # 🧾 Interprétation finale
    # ======================================================

    st.subheader("🧾 Interprétation et aide à la décision")

    st.markdown(
        f"""
    Pour le scope **{selected_scope_label.lower()}** et un buffer de **{selected_buffer} heure(s)** :

    - 🟦 **{blocked_ratio * 100:.1f} %** des locations seraient **bloquées**.  
    - 🟩 **{conflicts_resolved_ratio * 100:.1f} %** des conflits seraient **résolus**.  
    - 🟧 Les conflits représentent **{conflict_ratio * 100:.1f} %** des locations totales.

    En appliquant notre grille de lecture :

    - **Statut de la configuration actuelle** : `{status_label.upper()}`  
    - **Buffer recommandé pour ce scope** : **{rec_buffer} heure(s)**  
    - À ce seuil recommandé : **{rec_resolved * 100:.1f} %** de conflits résolus pour **{rec_blocked * 100:.1f} %** de locations bloquées.

    **Lecture Produit (exemple soutenance)** :

    > « Sur ce scope, un buffer de **{rec_buffer} heure(s)** permet de résoudre la grande majorité des conflits tout en maintenant un niveau de locations bloquées inférieur à **{rec_blocked * 100:.1f} %**.  
    > C'est le meilleur compromis entre expérience utilisateur et perte de chiffre d'affaires estimée. »
    """
    )

# ======================================================
# 🟩 MODE 2 : Onglet Pricing ML (API HF)
# ======================================================

else:
    st.subheader("💶 Pricing ML – API GetAround (modèle sklearn)")

    st.markdown(
        """
        Cet onglet interroge directement l'**API FastAPI déployée sur Hugging Face** pour
        estimer le **prix journalier recommandé** d'une location, en fonction des
        caractéristiques du véhicule et des options choisies.
        """
    )

    left, right = st.columns([2, 3])

    with left:
        st.markdown("### 🎛️ Paramètres véhicule")

        with st.form("pricing_form"):
            model_key = st.text_input("Model key (nom du modèle / gamme)", value="Volkswagen Golf")
            mileage = st.number_input("Kilométrage (km)", min_value=0, max_value=500_000, value=80_000, step=1)
            engine_power = st.number_input("Puissance moteur (ch)", min_value=40, max_value=600, value=110, step=1)

            fuel = st.selectbox(
                "Carburant",
                ["diesel", "petrol", "hybrid", "electric", "lpg", "other"],
                index=1,
            )

            paint_color = st.selectbox(
                "Couleur",
                ["black", "white", "grey", "silver", "blue", "red", "brown", "green", "other"],
                index=0,
            )

            car_type = st.selectbox(
                "Type de véhicule",
                ["sedan", "hatchback", "suv", "estate", "convertible", "van", "coupe", "other"],
                index=2,
            )

            st.markdown("### 🧩 Options & équipements")

            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                private_parking_available = st.checkbox("Parking privé disponible", value=1)
                has_gps = st.checkbox("GPS intégré", value=1)
                has_air_conditioning = st.checkbox("Climatisation", value=1)
            with col_opt2:
                automatic_car = st.checkbox("Boîte automatique", value=1)
                has_getaround_connect = st.checkbox("GetAround Connect", value=1)
                has_speed_regulator = st.checkbox("Régulateur de vitesse", value=1)

            winter_tires = st.checkbox("Pneus hiver montés", value=0)

            submitted = st.form_submit_button("🚀 Estimer le prix avec le modèle ML")

        if submitted:
            features = {
                "model_key": model_key,
                "mileage": float(mileage),
                "engine_power": float(engine_power),
                "fuel": fuel,
                "paint_color": paint_color,
                "car_type": car_type,
                "private_parking_available": int(private_parking_available),
                "has_gps": int(has_gps),
                "has_air_conditioning": int(has_air_conditioning),
                "automatic_car": int(automatic_car),
                "has_getaround_connect": int(has_getaround_connect),
                "has_speed_regulator": int(has_speed_regulator),
                "winter_tires": int(winter_tires),
            }

            price, payload_sent, raw_response = call_pricing_api(features)

            with right:
                if price is not None:
                    st.markdown(
                        f"""
                        <div class="price-card">
                          <div class="price-badge">Prix recommandé par le modèle</div>
                          <div class="price-amount">{price:.0f} € / jour</div>
                          <div class="price-sub">
                            Estimation basée sur le pipeline <b>sklearn</b> (OneHotEncoder + Régression)<br>
                            intégrant le type de véhicule, le kilométrage et les options sélectionnées.
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("#### 🔍 Détail de la requête envoyée à l'API")
                    st.code(payload_sent, language="json")

                    st.markdown("#### 📦 Réponse brute de l'API")
                    st.json(raw_response)

                else:
                    st.warning("Impossible de récupérer une prédiction. Vérifiez les logs ou la configuration de l'API.")

    if not submitted:
        with right:
            st.info(
                "Complète le formulaire à gauche puis clique sur "
                "**« Estimer le prix avec le modèle ML »** pour interroger l'API Hugging Face."
            )

# ======================================================
# 🧾 Footer
# ======================================================

st.markdown(
    """
    <div class="footer">
    Dashboard développé pour le cas GetAround – Projet de fin de formation Data Science / MLOps (Jedha).  
    Vue 1 : calibration du buffer entre locations • Vue 2 : estimation de prix par modèle ML déployé sur Hugging Face.
    </div>
    """,
    unsafe_allow_html=True,
)