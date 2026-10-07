import os
import re
import glob
import warnings
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import xgboost as xgb
import shap

from econml.dml import CausalForestDML

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Magisterka - Płaca Minimalna",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 1. USTAWIENIA I TWARDE DANE
# ============================================================

SEED = 42

PRZECIETNE_WYNAGRODZENIE_PL = {
    1999: 1706.74, 2000: 1923.81, 2001: 2061.85, 2002: 2133.21,
    2003: 2201.47, 2004: 2289.57, 2005: 2380.29, 2006: 2477.23,
    2007: 2691.03, 2008: 2943.88, 2009: 3102.96, 2010: 3224.98,
    2011: 3399.52, 2012: 3521.67, 2013: 3650.06, 2014: 3783.46,
    2015: 3899.78, 2016: 4047.21, 2017: 4271.51, 2018: 4585.03,
    2019: 4918.17, 2020: 5167.47, 2021: 5662.53, 2022: 6346.15,
    2023: 7155.48, 2024: 8181.72, 2025: 8903.56,
}

MIN_WAGE_PL = {
    2010: 1317, 2011: 1386, 2012: 1500, 2013: 1600, 2014: 1680,
    2015: 1750, 2016: 1850, 2017: 2000, 2018: 2100, 2019: 2250,
    2020: 2600, 2021: 2800, 2022: 3010, 2023: 3490, 2024: 4242,
    2025: 4666,
}

WOJEWODZTWA_MAP = {
    "02": "Dolnośląskie", "04": "Kujawsko-Pomorskie",
    "06": "Lubelskie", "08": "Lubuskie", "10": "Łódzkie",
    "12": "Małopolskie", "14": "Mazowieckie", "16": "Opolskie",
    "18": "Podkarpackie", "20": "Podlaskie", "22": "Pomorskie",
    "24": "Śląskie", "26": "Świętokrzyskie",
    "28": "Warmińsko-Mazurskie", "30": "Wielkopolskie",
    "32": "Zachodniopomorskie",
}

COLUMN_ALIASES = {
    "Kod": ["Kod", "kod", "Kod teryt", "Kod_TERYT", "id"],
    "Rok": ["Rok", "rok", "Year", "TIME"],
    "Stopa_Zatrudnienia": [
        "Stopa_Zatrudnienia", "Wskaźnik zatrudnienia",
        "Wskaznik zatrudnienia", "Wskaźnik zatrudnienia w wieku 15-64",
        "Wskaźnik zatrudnienia (15-64)", "Employment rate",
    ],
    "Bezrobocie_Rejestrowane": [
        "Bezrobocie_Rejestrowane", "Stopa bezrobocia rejestrowanego",
        "Stopa bezrobocia", "Bezrobocie rejestrowane",
        "Registered unemployment rate",
    ],
    "PKB_Nominalny": [
        "PKB_Nominalny", "PKB", "Produkt krajowy brutto",
        "Produkt krajowy brutto (ceny bieżące)",
    ],
    "Realny_PKB_per_capita": [
        "Realny_PKB_per_capita", "PKB na 1 mieszkańca",
        "PKB na 1 mieszkańca (ceny bieżące)",
        "PKB per capita", "PKB na mieszkańca",
    ],
    "WDB_Przemysl": [
        "WDB_Przemysl", "Wartość dodana brutto - przemysł",
        "Wartość dodana brutto przemysł", "Przemysł", "WDB przemysł",
    ],
    "WDB_Ogolem": [
        "WDB_Ogolem", "Wartość dodana brutto ogółem",
        "Wartość dodana brutto", "WDB ogółem",
    ],
    "Zatrudnieni_Ogolem": [
        "Zatrudnieni_Ogolem", "Pracujący ogółem", "Zatrudnieni ogółem",
        "Pracujący", "Pracujący ogółem (tys.)",
    ],
    "Populacja": [
        "Populacja", "Ludność", "Ludność ogółem", "Stan ludności",
    ],
    "Wskaznik_CPI": [
        "Wskaznik_CPI", "Wskaźnik cen towarów i usług konsumpcyjnych",
        "Wskaźnik cen towarów i usług konsumpcyjnych (rok poprzedni=100)",
        "CPI", "Wskaźnik CPI",
    ],
}


# ============================================================
# 2. NARZĘDZIA DO WCZYTYWANIA I NORMALIZACJI DANYCH
# ============================================================

def normalize_col_name(value):
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    return value


def find_column(df, canonical_name, required=True):
    aliases = COLUMN_ALIASES.get(canonical_name, [canonical_name])

    normalized = {normalize_col_name(c): c for c in df.columns}

    for alias in aliases:
        alias_n = normalize_col_name(alias)
        if alias_n in normalized:
            return normalized[alias_n]

    lower_cols = {str(c).lower(): c for c in df.columns}
    for alias in aliases:
        a = str(alias).lower()
        for col_lower, col in lower_cols.items():
            if a in col_lower or col_lower in a:
                return col

    if required:
        raise ValueError(
            f"Nie znaleziono wymaganej kolumny '{canonical_name}'. "
            f"Dostępne kolumny: {list(df.columns)}"
        )
    return None


def rename_to_canonical(df, canonical_names):
    out = df.copy()
    rename_map = {}

    for canonical in canonical_names:
        col = find_column(out, canonical, required=False)
        if col is not None and col != canonical:
            rename_map[col] = canonical

    return out.rename(columns=rename_map)


def read_table(path):
    path = str(path)
    suffix = Path(path).suffix.lower()

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path)

    if suffix == ".csv":
        errors = []
        for encoding in ["utf-8-sig", "utf-8", "cp1250", "latin1"]:
            for sep in [";", ",", "\t"]:
                try:
                    df = pd.read_csv(
                        path,
                        encoding=encoding,
                        sep=sep,
                        low_memory=False,
                    )
                    if df.shape[1] > 1:
                        return df
                except Exception as exc:
                    errors.append(str(exc))

        raise ValueError(
            f"Nie udało się odczytać pliku CSV: {path}. "
            f"Ostatni błąd: {errors[-1] if errors else 'nieznany'}"
        )

    raise ValueError(f"Nieobsługiwany format pliku: {path}")


def detect_file(uploaded_files, prefix):
    matches = [
        f for f in uploaded_files
        if Path(f).name.upper().startswith(prefix.upper())
    ]

    if not matches:
        return None

    preferred = []
    if prefix.upper() == "RYNE":
        preferred = [f for f in matches if "4098" in Path(f).name]
    elif prefix.upper() == "RACH":
        preferred = [f for f in matches if "3510" in Path(f).name]
    elif prefix.upper() == "CENY":
        preferred = [f for f in matches if "2496" in Path(f).name]

    return sorted(preferred or matches)[-1]


def get_local_data_files():
    extensions = {".csv", ".xlsx", ".xls", ".xlsm"}
    return [
        str(Path(p))
        for p in glob.glob("*")
        if Path(p).suffix.lower() in extensions
    ]


def coerce_numeric(df, columns):
    out = df.copy()
    for col in columns:
        if col in out.columns:
            if out[col].dtype == "object":
                out[col] = (
                    out[col]
                    .astype(str)
                    .str.replace("\xa0", "", regex=False)
                    .str.replace(" ", "", regex=False)
                    .str.replace(",", ".", regex=False)
                )
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def extract_teryt(code):
    digits = re.sub(r"\D", "", str(code))
    return digits.zfill(7)[:2] if digits else np.nan


# ============================================================
# 3. PRZYGOTOWANIE PANELU
# ============================================================

@st.cache_data(show_spinner=False)
def load_and_clean_data(ryne_path, rach_path, ceny_path):
    ryne_df = read_table(ryne_path)
    rach_df = read_table(rach_path)
    ceny_df = read_table(ceny_path)

    ryne_df = rename_to_canonical(
        ryne_df,
        [
            "Kod", "Rok", "Stopa_Zatrudnienia",
            "Bezrobocie_Rejestrowane",
            "Zatrudnieni_Ogolem", "Populacja",
        ],
    )

    rach_df = rename_to_canonical(
        rach_df,
        [
            "Kod", "Rok", "PKB_Nominalny",
            "Realny_PKB_per_capita",
            "WDB_Przemysl", "WDB_Ogolem",
        ],
    )

    ceny_df = rename_to_canonical(
        ceny_df,
        ["Rok", "Wskaznik_CPI"],
    )

    required_ryne = [
        "Kod", "Rok", "Stopa_Zatrudnienia",
        "Bezrobocie_Rejestrowane",
    ]
    required_rach = [
        "Kod", "Rok", "PKB_Nominalny",
        "WDB_Przemysl", "WDB_Ogolem",
    ]
    required_ceny = ["Rok", "Wskaznik_CPI"]

    for c in required_ryne:
        find_column(ryne_df, c, required=True)
    for c in required_rach:
        find_column(rach_df, c, required=True)
    for c in required_ceny:
        find_column(ceny_df, c, required=True)

    if "Populacja" not in ryne_df.columns:
        ryne_df["Populacja"] = np.nan

    if "Zatrudnieni_Ogolem" not in ryne_df.columns:
        ryne_df["Zatrudnieni_Ogolem"] = np.nan

    if "Realny_PKB_per_capita" not in rach_df.columns:
        rach_df["Realny_PKB_per_capita"] = np.nan

    for frame in [ryne_df, rach_df]:
        frame["Rok"] = pd.to_numeric(
            frame["Rok"], errors="coerce"
        ).astype("Int64")
        frame["Kod"] = frame["Kod"].astype(str).str.strip()

    ceny_df["Rok"] = pd.to_numeric(
        ceny_df["Rok"], errors="coerce"
    ).astype("Int64")

    ryne_df = coerce_numeric(
        ryne_df,
        [
            "Stopa_Zatrudnienia",
            "Bezrobocie_Rejestrowane",
            "Zatrudnieni_Ogolem",
            "Populacja",
        ],
    )

    rach_df = coerce_numeric(
        rach_df,
        [
            "PKB_Nominalny",
            "Realny_PKB_per_capita",
            "WDB_Przemysl",
            "WDB_Ogolem",
        ],
    )

    ceny_df = coerce_numeric(
        ceny_df,
        ["Wskaznik_CPI"],
    )

    ryne_df = ryne_df.drop_duplicates(["Kod", "Rok"])
    rach_df = rach_df.drop_duplicates(["Kod", "Rok"])
    ceny_df = ceny_df.drop_duplicates(["Rok"])

    df = ryne_df.merge(
        rach_df,
        on=["Kod", "Rok"],
        how="inner",
        suffixes=("", "_rach"),
    )

    df = df.merge(
        ceny_df,
        on=["Rok"],
        how="inner",
    )

    df["Kod_Str"] = df["Kod"].map(extract_teryt)
    df["Wojewodztwo"] = df["Kod_Str"].map(WOJEWODZTWA_MAP)
    df = df[df["Wojewodztwo"].notna()].copy()

    df["Przecietne_Wynagrodzenie_Kraj"] = df["Rok"].map(
        PRZECIETNE_WYNAGRODZENIE_PL
    )
    df["Placa_Minimalna"] = df["Rok"].map(MIN_WAGE_PL)

    df["CPI"] = pd.to_numeric(
        df["Wskaznik_CPI"], errors="coerce"
    )
    df["CPI_factor"] = df["CPI"] / 100.0

    df["Realna_Placa_Minimalna"] = (
        df["Placa_Minimalna"] / df["CPI_factor"]
    )

    df["Realne_Wynagrodzenie"] = (
        df["Przecietne_Wynagrodzenie_Kraj"] /
        df["CPI_factor"]
    )

    df["Realny_PKB"] = (
        df["PKB_Nominalny"] / df["CPI_factor"]
    )

    # Treatment: realny Wskaźnik Kaitza.
    # Ponieważ obie płace defluujemy tym samym CPI, CPI algebraicznie
    # skraca się w ilorazie, zachowując standardowy Kaitz.
    df["Kaitz_Index"] = (
        df["Realna_Placa_Minimalna"] /
        df["Realne_Wynagrodzenie"]
    )

    df = df.sort_values(["Kod_Str", "Rok"])

    df["Wzrost_PKB"] = (
        df.groupby("Kod_Str")["Realny_PKB"]
        .pct_change(fill_method=None) * 100
    )

    df["Udzial_Przemyslu"] = np.where(
        df["WDB_Ogolem"] != 0,
        df["WDB_Przemysl"] / df["WDB_Ogolem"],
        np.nan,
    )

    df["Produktywnosc"] = np.where(
        df["Zatrudnieni_Ogolem"] != 0,
        df["Realny_PKB"] / df["Zatrudnieni_Ogolem"],
        np.nan,
    )

    # Jeżeli GUS podał nominalne PKB per capita, przeliczamy je
    # samodzielnie na realną wielkość, aby zmienna była spójna.
    if "Populacja" in df.columns:
        calculated_real_pc = np.where(
            df["Populacja"] != 0,
            df["Realny_PKB"] / df["Populacja"],
            np.nan,
        )
        df["Realny_PKB_per_capita"] = calculated_real_pc

    model_cols = [
        "Wojewodztwo",
        "Kod_Str",
        "Rok",
        "Kaitz_Index",
        "Stopa_Zatrudnienia",
        "Bezrobocie_Rejestrowane",
        "Wzrost_PKB",
        "Realny_PKB_per_capita",
        "Udzial_Przemyslu",
        "Produktywnosc",
        "Populacja",
        "CPI",
    ]

    df = coerce_numeric(
        df,
        [
            c for c in model_cols
            if c in df.columns
            and c not in ["Wojewodztwo", "Kod_Str"]
        ],
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=model_cols).copy()
    df = df.drop_duplicates(["Kod_Str", "Rok"])

    return df.reset_index(drop=True)


# ============================================================
# 4. BENCHMARK EKONOMETRYCZNY
# ============================================================

def run_panel_benchmark(df):
    work = df.copy()

    formula_base = (
        "Stopa_Zatrudnienia ~ Kaitz_Index + "
        "Bezrobocie_Rejestrowane + Wzrost_PKB + "
        "Udzial_Przemyslu"
    )

    pooled = smf.ols(
        formula_base,
        data=work,
    ).fit(
        cov_type="HC1"
    )

    # Two-Way Fixed Effects:
    # efekty stałe województw + efekty stałe lat.
    twfe_formula = (
        formula_base +
        " + C(Wojewodztwo) + C(Rok)"
    )

    twfe = smf.ols(
        twfe_formula,
        data=work,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": work["Wojewodztwo"]},
    )

    return {
        "pooled_ols": pooled,
        "twfe": twfe,
        "ols_ate": float(pooled.params["Kaitz_Index"]),
        "twfe_ate": float(twfe.params["Kaitz_Index"]),
    }


# ============================================================
# 5. DML + CAUSAL FOREST
# ============================================================

X_COLS = [
    "Bezrobocie_Rejestrowane",
    "Wzrost_PKB",
    "Realny_PKB_per_capita",
    "Udzial_Przemyslu",
    "Produktywnosc",
]

W_COLS = [
    "Populacja",
    "CPI",
]


def build_xgb():
    return xgb.XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=SEED,
        n_jobs=1,
    )


@st.cache_resource(show_spinner=False)
def run_models(df):
    work = df.copy()

    benchmark = run_panel_benchmark(work)

    Y = work["Stopa_Zatrudnienia"].to_numpy(dtype=float)
    T = work["Kaitz_Index"].to_numpy(dtype=float)

    X = work[X_COLS].copy()
    W = work[W_COLS].copy()

    causal_forest = CausalForestDML(
        model_y=build_xgb(),
        model_t=build_xgb(),
        discrete_treatment=False,
        n_estimators=1000,
        min_samples_leaf=5,
        max_depth=None,
        max_features="sqrt",
        inference=True,
        random_state=SEED,
        n_jobs=1,
    )

    causal_forest.fit(
        Y,
        T,
        X=X,
        W=W,
    )

    effects = causal_forest.effect(X)
    work["Estimated_Effect_CF"] = effects

    try:
        effect_lower, effect_upper = causal_forest.effect_interval(
            X,
            alpha=0.05,
        )
        work["Effect_Lower_95"] = effect_lower
        work["Effect_Upper_95"] = effect_upper
    except Exception:
        pass

    # Próba użycia SHAP bezpośrednio z EconML.
    shap_values = None
    try:
        shap_values = causal_forest.shap_values(X)
    except Exception:
        shap_values = None

    # Stabilny fallback: SHAP dla surogatowego XGBoost,
    # który aproksymuje funkcję tau-hat.
    shap_surrogate = None
    shap_surrogate_values = None

    try:
        surrogate = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=1,
        )

        surrogate.fit(X, effects)

        explainer = shap.TreeExplainer(surrogate)
        shap_surrogate_values = explainer.shap_values(X)
        shap_surrogate = surrogate
    except Exception:
        pass

    return {
        "benchmark": benchmark,
        "causal_forest": causal_forest,
        "ate_cf": float(np.mean(effects)),
        "effect_median": float(np.median(effects)),
        "effect_std": float(np.std(effects)),
        "shap_values": shap_values,
        "shap_surrogate": shap_surrogate,
        "shap_surrogate_values": shap_surrogate_values,
        "X_data": X,
        "df_final": work,
    }


# ============================================================
# 6. WIZUALIZACJE
# ============================================================

def make_shap_plot(res):
    values = res["shap_surrogate_values"]
    X = res["X_data"]

    if values is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "Nie udało się obliczyć wartości SHAP.",
            ha="center",
            va="center",
        )
        ax.axis("off")
        return fig

    display_names = {
        "Bezrobocie_Rejestrowane": "Bezrobocie rejestrowane",
        "Wzrost_PKB": "Wzrost PKB",
        "Realny_PKB_per_capita": "Realny PKB per capita",
        "Udzial_Przemyslu": "Udział przemysłu w WDB",
        "Produktywnosc": "Produktywność",
    }

    X_display = X.rename(columns=display_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    shap.summary_plot(
        values,
        X_display,
        show=False,
        plot_size=None,
    )

    ax.set_title(
        "SHAP — czynniki heterogeniczności efektu płacy minimalnej"
    )

    fig.tight_layout()
    return fig


def format_effect(value):
    if pd.isna(value):
        return "—"
    return f"{value:.4f}"


# ============================================================
# 7. STREAMLIT
# ============================================================

st.title("Heterogeniczny wpływ płacy minimalnej na zatrudnienie")

st.markdown(
    "Regionalny panel Polski — **benchmark ekonometryczny → "
    "Double Machine Learning → Causal Forest → SHAP**"
)

st.sidebar.header("Panel sterowania")

local_files = get_local_data_files()

uploaded_ryne = st.sidebar.file_uploader(
    "RYNE — rynek pracy",
    type=["csv", "xlsx", "xls"],
    key="ryne_upload",
)

uploaded_rach = st.sidebar.file_uploader(
    "RACH — rachunki regionalne",
    type=["csv", "xlsx", "xls"],
    key="rach_upload",
)

uploaded_ceny = st.sidebar.file_uploader(
    "CENY — CPI",
    type=["csv", "xlsx", "xls"],
    key="ceny_upload",
)


def save_uploaded_temp(uploaded_file, prefix):
    if uploaded_file is None:
        return None

    temp_dir = Path(".streamlit_uploads")
    temp_dir.mkdir(exist_ok=True)

    target = temp_dir / f"{prefix}_{uploaded_file.name}"
    target.write_bytes(uploaded_file.getbuffer())

    return str(target)


ryne_upload_path = save_uploaded_temp(
    uploaded_ryne,
    "RYNE",
)

rach_upload_path = save_uploaded_temp(
    uploaded_rach,
    "RACH",
)

ceny_upload_path = save_uploaded_temp(
    uploaded_ceny,
    "CENY",
)

auto_ryne = detect_file(
    local_files,
    "RYNE",
)

auto_rach = detect_file(
    local_files,
    "RACH",
)

auto_ceny = detect_file(
    local_files,
    "CENY",
)

ryne_path = ryne_upload_path or auto_ryne
rach_path = rach_upload_path or auto_rach
ceny_path = ceny_upload_path or auto_ceny

st.sidebar.markdown("---")
st.sidebar.caption("Automatycznie wykryte pliki")

st.sidebar.write(
    f"**RYNE:** "
    f"{Path(ryne_path).name if ryne_path else 'brak'}"
)

st.sidebar.write(
    f"**RACH:** "
    f"{Path(rach_path).name if rach_path else 'brak'}"
)

st.sidebar.write(
    f"**CENY:** "
    f"{Path(ceny_path).name if ceny_path else 'brak'}"
)

run_analysis = st.sidebar.button(
    "🚀 Uruchom pełną analizę",
    type="primary",
    use_container_width=True,
)

if run_analysis:

    if not all(
        [ryne_path, rach_path, ceny_path]
    ):
        st.error(
            "Brakuje co najmniej jednego źródła danych. "
            "Aplikacja potrzebuje plików RYNE, RACH i CENY."
        )
        st.stop()

    with st.spinner(
        "Wczytywanie danych, budowa panelu i estymacja modeli..."
    ):
        try:
            df = load_and_clean_data(
                ryne_path,
                rach_path,
                ceny_path,
            )

            if df.empty:
                st.error(
                    "Po oczyszczeniu nie pozostały żadne obserwacje. "
                    "Sprawdź zakres lat i nazwy kolumn."
                )
                st.stop()

            res = run_models(df)
            df_final = res["df_final"]

        except Exception as exc:
            st.error(
                f"Błąd analizy: {exc}"
            )
            st.exception(exc)
            st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Dane & EDA",
            "📈 Benchmark ekonometryczny",
            "🌲 Causal Machine Learning",
            "🔍 Heterogeniczność (SHAP)",
        ]
    )

    # ========================================================
    # TAB 1
    # ========================================================

    with tab1:

        st.subheader(
            "Oczyszczony panel wojewódzki"
        )

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Obserwacje",
            len(df_final),
        )

        m2.metric(
            "Województwa",
            df_final["Wojewodztwo"].nunique(),
        )

        m3.metric(
            "Lata",
            f"{int(df_final['Rok'].min())}–"
            f"{int(df_final['Rok'].max())}",
        )

        m4.metric(
            "Średni Kaitz",
            f"{df_final['Kaitz_Index'].mean():.3f}",
        )

        st.dataframe(
            df_final[
                [
                    "Wojewodztwo",
                    "Rok",
                    "Kaitz_Index",
                    "Stopa_Zatrudnienia",
                    "Bezrobocie_Rejestrowane",
                    "Wzrost_PKB",
                    "Udzial_Przemyslu",
                ]
            ].sort_values(
                ["Rok", "Wojewodztwo"]
            ),
            use_container_width=True,
            hide_index=True,
        )

        col1, col2 = st.columns(2)

        with col1:

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            sns.histplot(
                df_final["Kaitz_Index"],
                bins=20,
                kde=True,
                ax=ax,
            )

            ax.set_title(
                "Rozkład wskaźnika Kaitza"
            )

            ax.set_xlabel(
                "Kaitz Index"
            )

            ax.set_ylabel(
                "Liczba obserwacji"
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                clear_figure=True,
            )

        with col2:

            yearly = (
                df_final
                .groupby("Rok")["Kaitz_Index"]
                .mean()
                .reset_index()
            )

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            ax.plot(
                yearly["Rok"],
                yearly["Kaitz_Index"],
                marker="o",
            )

            ax.set_title(
                "Średni Kaitz w czasie"
            )

            ax.set_xlabel(
                "Rok"
            )

            ax.set_ylabel(
                "Kaitz Index"
            )

            ax.grid(
                alpha=0.25
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                clear_figure=True,
            )

        st.info(
            "Wskaźnik Kaitza jest skonstruowany jako realna płaca "
            "minimalna podzielona przez realne przeciętne "
            "wynagrodzenie. Ponieważ obie wielkości są deflowane "
            "tym samym CPI, CPI algebraicznie skraca się w ilorazie."
        )

    # ========================================================
    # TAB 2
    # ========================================================

    with tab2:

        st.subheader(
            "Benchmark ekonometryczny"
        )

        pooled = res["benchmark"]["pooled_ols"]
        twfe = res["benchmark"]["twfe"]

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Pooled OLS — Kaitz",
                format_effect(
                    res["benchmark"]["ols_ate"]
                ),
            )

            st.caption(
                "Model bez efektów stałych; "
                "prosty punkt odniesienia."
            )

        with c2:

            st.metric(
                "Two-Way FE — Kaitz",
                format_effect(
                    res["benchmark"]["twfe_ate"]
                ),
            )

            st.caption(
                "Efekty stałe województw + efekty stałe lat."
            )

        st.markdown(
            "### Two-Way Fixed Effects"
        )

        st.text(
            twfe.summary()
        )

    # ========================================================
    # TAB 3
    # ========================================================

    with tab3:

        st.subheader(
            "Causal Forest + Double Machine Learning"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Średni efekt Causal Forest",
            format_effect(
                res["ate_cf"]
            ),
        )

        c2.metric(
            "Mediana efektu",
            format_effect(
                res["effect_median"]
            ),
        )

        c3.metric(
            "Odchylenie efektów",
            format_effect(
                res["effect_std"]
            ),
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                "**Średni estymowany efekt "
                "według województwa**"
            )

            regional = (
                df_final
                .groupby("Wojewodztwo")[
                    "Estimated_Effect_CF"
                ]
                .mean()
                .sort_values()
            )

            fig, ax = plt.subplots(
                figsize=(7, 8)
            )

            sns.barplot(
                x=regional.values,
                y=regional.index,
                ax=ax,
            )

            ax.axvline(
                0,
                color="black",
                linestyle="--",
                linewidth=1,
            )

            ax.set_xlabel(
                "Średni estymowany efekt τ(x)"
            )

            ax.set_ylabel("")

            ax.set_title(
                "Heterogeniczność efektu regionalnego"
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                clear_figure=True,
            )

        with col2:

            st.write(
                "**Efekt a warunki na rynku pracy**"
            )

            fig, ax = plt.subplots(
                figsize=(7, 6)
            )

            scatter = ax.scatter(
                df_final[
                    "Bezrobocie_Rejestrowane"
                ],
                df_final[
                    "Estimated_Effect_CF"
                ],
                c=df_final[
                    "Kaitz_Index"
                ],
                alpha=0.75,
            )

            ax.axhline(
                0,
                color="black",
                linestyle="--",
                linewidth=1,
            )

            ax.set_xlabel(
                "Stopa bezrobocia rejestrowanego"
            )

            ax.set_ylabel(
                "Estymowany efekt τ(x)"
            )

            ax.set_title(
                "Efekt płacy minimalnej "
                "a rezerwy rynku pracy"
            )

            cbar = fig.colorbar(
                scatter,
                ax=ax,
            )

            cbar.set_label(
                "Kaitz Index"
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                clear_figure=True,
            )

        st.markdown(
            "### Wyniki dla poszczególnych obserwacji"
        )

        result_columns = [
            "Wojewodztwo",
            "Rok",
            "Kaitz_Index",
            "Bezrobocie_Rejestrowane",
            "Wzrost_PKB",
            "Udzial_Przemyslu",
            "Estimated_Effect_CF",
        ]

        if "Effect_Lower_95" in df_final.columns:

            result_columns += [
                "Effect_Lower_95",
                "Effect_Upper_95",
            ]

        st.dataframe(
            df_final[
                result_columns
            ].sort_values(
                "Estimated_Effect_CF"
            ),
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # TAB 4
    # ========================================================

    with tab4:

        st.subheader(
            "Interpretacja heterogeniczności — SHAP"
        )

        st.markdown(
            """
            SHAP pokazuje, które cechy odpowiadają za różnice
            w **estymowanym efekcie τ(x)** pomiędzy obserwacjami.
            Wartości SHAP nie są dodatkowymi efektami przyczynowymi.
            """
        )

        fig = make_shap_plot(
            res
        )

        st.pyplot(
            fig,
            clear_figure=True,
        )

        st.markdown(
            "### Średnia bezwzględna wartość SHAP"
        )

        shap_values = (
            res["shap_surrogate_values"]
        )

        if shap_values is not None:

            importance = pd.DataFrame(
                {
                    "Zmienna": X_COLS,
                    "Średnia |SHAP|": np.mean(
                        np.abs(shap_values),
                        axis=0,
                    ),
                }
            ).sort_values(
                "Średnia |SHAP|",
                ascending=False,
            )

            name_map = {
                "Bezrobocie_Rejestrowane":
                    "Bezrobocie rejestrowane",
                "Wzrost_PKB":
                    "Wzrost PKB",
                "Realny_PKB_per_capita":
                    "Realny PKB per capita",
                "Udzial_Przemyslu":
                    "Udział przemysłu w WDB",
                "Produktywnosc":
                    "Produktywność",
            }

            importance["Zmienna"] = (
                importance["Zmienna"]
                .map(name_map)
            )

            st.dataframe(
                importance,
                use_container_width=True,
                hide_index=True,
            )

else:

    st.info(
        "Wybierz pliki RYNE, RACH i CENY lub umieść je w folderze "
        "aplikacji, a następnie kliknij **Uruchom pełną analizę**."
    )

st.sidebar.markdown("---")

st.sidebar.caption(
    "Model: CausalForestDML + XGBoost | Seed: 42 | "
    "H5 (młodzież) wyłączona z analizy"
)
