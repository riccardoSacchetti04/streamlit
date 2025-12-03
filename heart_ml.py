"""
Heart Disease – ML Demo

Piccola applicazione Streamlit che:
- carica un dataset pubblico su malattia cardiaca
- allena un modello binario (malattia sì/no)
- mostra alcune metriche di performance
- permette di fare una previsione per un singolo paziente
"""
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# *PAGE CONFIG 
st.set_page_config(page_title="Heart Disease – ML Demo", layout="wide", page_icon="❤️")

# *VARS  -----------------------------------------------------------
# Data path
DATA_PATH = Path("data/health/heart.csv")

# Cols selezionate come feature
FEATURE_COLS = ["age", "trestbps", "chol", "thalch", "oldpeak"]

# Etichette leggibili in italiano per l'UI
FEATURE_LABELS = {
    "age": "Età (anni)",
    "trestbps": "Pressione a riposo (mm Hg)",
    "chol": "Colesterolo (mg/dl)",
    "thalch": "Freq. cardiaca max (bpm)",
    "oldpeak": "Depressione ST (oldpeak)",
}
TARGET_LABEL = "Presenza di malattia (target)"

# *UTILS -----------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Legge il csv e crea la colonna target binaria."""
    df = pd.read_csv(DATA_PATH)

    # num: 0 = sano, 1–4 = malattia
    df["target"] = (df["num"] > 0).astype(int)

    cols = FEATURE_COLS + ["target"]
    return df[cols]


@st.cache_resource
def train_model(df: pd.DataFrame, max_depth: int, min_samples_leaf: int):
    """
    Allena un RandomForest e calcola:
    - accuracy su train e test
    - baseline (classe più frequente)
    - importanza delle feature
    """
    X = df[FEATURE_COLS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    # baseline: predire sempre la classe più frequente
    majority_class = int(y_test.value_counts().idxmax())
    baseline_pred = [majority_class] * len(y_test)
    baseline_acc = accuracy_score(y_test, baseline_pred) # calcola acc se y_pred fosse sempre 1 = malato contro y_true

    metrics = {
        "acc_train": acc_train,
        "acc_test": acc_test,
        "baseline_acc": baseline_acc,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # importanza delle feature
    fi = pd.Series(
        model.feature_importances_,
        index=[FEATURE_LABELS[c] for c in FEATURE_COLS],
    ).sort_values(ascending=True)

    return model, metrics, fi


######################################################################
# ------------------------------ ST APP ---------------------------- #
######################################################################

df = load_data()

# *SIDEBAR PARAMETRI -------------------------------------------------
st.sidebar.header("⚙️ Parametri Modello")
st.sidebar.write("Modifica gli iperparametri del Random Forest per vedere come cambia l'accuracy.")

# Sliders per controllare complessità
p_max_depth = st.sidebar.slider("Max Depth (Profondità max)", 1, 20, 5)
p_min_samples_leaf = st.sidebar.slider("Min Samples Leaf (Min campioni per foglia)", 1, 20, 2)


# *MAIN PAGE ---------------------------------------------------------
st.title("❤️ Heart Disease – ML Demo")
st.caption(
    "Esempio didattico: modello binario (malattia sì/no) su 5 feature "
    "numeriche. Non è uno strumento medico reale."
)

# *PANORAMICA --------------------------------------------------------

st.subheader("🔍 Panoramica del dataset")

col_a, col_b, col_c = st.columns(3)

n_patients = len(df)
positive_rate = df["target"].mean()

col_a.metric("Numero pazienti", n_patients)
col_b.metric("Con malattia (%)", f"{positive_rate:.1%}")
col_c.metric(
    "Sani vs malati",
    f"{(1 - positive_rate):.1%} sani / {positive_rate:.1%} malati",
)

with st.expander("Mostra prime righe del dataset"):
    st.dataframe(df.head())

# --- NUOVA SEZIONE: ANALISI DISTRIBUZIONE ---
st.markdown("### 📊 Analisi della distribuzione")

# 1. Selectbox per scegliere la feature
selected_feature_name = st.selectbox(
    "Seleziona una variabile da analizzare:",
    options=FEATURE_COLS,
    format_func=lambda x: FEATURE_LABELS[x] # Mostra l'etichetta leggibile
)

# 2. Due plot uno accanto all'altro
col_plot1, col_plot2 = st.columns(2)

with col_plot1:
    st.markdown(f"**Distribuzione: {FEATURE_LABELS[selected_feature_name]}**")
    fig_hist, ax_hist = plt.subplots(figsize=(5, 4))
    sns.histplot(data=df, x=selected_feature_name, kde=True, ax=ax_hist, color="skyblue")
    ax_hist.set_xlabel(FEATURE_LABELS[selected_feature_name])
    ax_hist.set_ylabel("Conteggio")
    st.pyplot(fig_hist)

with col_plot2:
    st.markdown("**Confronto Sani vs Malati**")
    fig_vio, ax_vio = plt.subplots(figsize=(5, 4))
    # Creiamo un df temporaneo per etichette più chiare nel grafico
    df_viz = df.copy()
    df_viz["Stato"] = df_viz["target"].map({0: "Sano", 1: "Malato"})
    
    sns.violinplot(
        data=df_viz, 
        x="Stato", 
        y=selected_feature_name, 
        palette="Set2", 
        ax=ax_vio
    )
    ax_vio.set_xlabel("")
    ax_vio.set_ylabel(FEATURE_LABELS[selected_feature_name])
    st.pyplot(fig_vio)

# 3. Tabella statistiche e domande
st.markdown("#### Statistiche descrittive e Analisi")
col_stats, col_questions = st.columns([1, 1])

with col_stats:
    # Mostriamo il describe trasposto per leggibilità
    desc_stats = df[selected_feature_name].describe()
    st.dataframe(desc_stats, use_container_width=True)

with col_questions:
    st.info(
        """
        **Domande guida:**
        1. La variabile ha una distribuzione normale (a campana)?
        2. Ci sono outlier (valori estremi molto lontani dalla media)?
        """
    )


# * RF PERFORMANCE ---------------------------------------------------

# Passiamo i parametri della sidebar al training
model, metrics, feature_importances = train_model(df, p_max_depth, p_min_samples_leaf)

st.subheader("📏 Performance del modello")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy su train", f"{metrics['acc_train']:.2%}")
col2.metric("Accuracy su test", f"{metrics['acc_test']:.2%}")
col3.metric(
    "Baseline (classe più frequente)",
    f"{metrics['baseline_acc']:.2%}",
)

st.caption(
    "Se l'accuracy su test è simile a quella su train e migliore della "
    "baseline, il modello sta generalizzando in modo ragionevole."
)

# *CORR & FEATURE IMPORTANCE -----------------------------------------

st.subheader("📈 Correlazioni e importanza delle variabili")

col_corr, col_imp = st.columns(2)

with col_corr:
    st.markdown("**Correlazione tra variabili e target**")

    # Rename cols for corr matrix plot
    df_corr = df.copy()
    rename_map = {col: FEATURE_LABELS[col] for col in FEATURE_COLS}
    rename_map["target"] = TARGET_LABEL
    df_corr = df_corr.rename(columns=rename_map)

    # compute corr matrix
    corr = df_corr.corr()

    # corr matrix heatmap with sns
    fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax_corr,
    )
    ax_corr.set_title("Matrice di correlazione")
    st.pyplot(fig_corr)

with col_imp:
    st.markdown("**Importanza delle variabili (RandomForest)**")

    # Plot feature importances barchart with matplotlib
    fig_imp, ax_imp = plt.subplots(figsize=(5, 4))
    feature_importances.plot(kind="barh", ax=ax_imp)
    ax_imp.set_xlabel("Importanza (Gini)")
    ax_imp.set_ylabel("Variabile")
    ax_imp.set_title("Importanza delle variabili")
    plt.tight_layout()
    st.pyplot(fig_imp)

# --- NUOVA SEZIONE: TOP FEATURE FOCUS ---
st.write("---")
st.subheader("🏆 Focus sulla variabile più importante")

# Recuperiamo il label della variabile più importante (ultima della serie ordinata)
top_imp_label = feature_importances.index[-1]
# Recuperiamo il nome originale della colonna invertendo il dizionario
top_col_name = [k for k, v in FEATURE_LABELS.items() if v == top_imp_label][0]

st.markdown(f"La variabile che influenza maggiormente il modello è: **{top_imp_label}**")

fig_box, ax_box = plt.subplots(figsize=(6, 3))
df_viz_top = df.copy()
df_viz_top["Condizione"] = df_viz_top["target"].map({0: "Sano", 1: "Malato"})

sns.boxplot(data=df_viz_top, x="Condizione", y=top_col_name, palette="Pastel1", ax=ax_box)
ax_box.set_title(f"Distribuzione di {top_imp_label} vs Malattia")
ax_box.set_xlabel("")
ax_box.set_ylabel(top_imp_label)
st.pyplot(fig_box)


# *FORM PAZIENTE ----------------------------------------------------

st.subheader("🧪 Inserisci i dati del paziente")

cols = st.columns(3)
user_input: dict[str, float] = {}

for i, col_name in enumerate(FEATURE_COLS):
    serie = df[col_name]
    min_val = float(serie.min())
    max_val = float(serie.max())
    default = float(serie.median()) # settiamo il valore di deafault sulla mediana

    label = FEATURE_LABELS[col_name]

    with cols[i % 3]: # <---- watch out
        # prenndiamo lo user input con number input di streamlit
        user_input[col_name] = st.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=default,
        )

if st.button("Predici rischio"):
    input_df = pd.DataFrame([user_input])
    proba = model.predict_proba(input_df)[0]
    pred = int(proba[1] > 0.5)

    col_res1, col_res2 = st.columns(2)
    label_risk = "ALTO" if pred == 1 else "BASSO"
    col_res1.metric("Rischio stimato", label_risk)
    col_res2.metric("Probabilità di malattia", f"{proba[1]:.1%}")

    st.write("Valori inseriti:")
    pretty_input = {FEATURE_LABELS[k]: v for k, v in user_input.items()}
    st.json(pretty_input)

    st.info(
        "⚠️ Esempio didattico su un dataset pubblico. "
        "Non è uno strumento clinico e non va usato per decisioni reali."
    )