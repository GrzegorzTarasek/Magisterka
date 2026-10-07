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
        "Bezrobocie_Rejestrowane", "Stopa bezrobocia (BAEL)",
        "Stopa bezrobocia", "Stopa bezrobocia (BAEL)",
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


REPOSITORY_FILES = {
    "RYNE_SECTOR": "RYNE_4098_CTAB_20261006234440.csv",
    "RYNE_UNEMPLOYMENT": "RYNE_4100_CTAB_20261006234212.csv",
    "RYNE_EMPLOYMENT": "RYNE_4112_CTAB_20261006234336.csv",
    "RYNE_NEET": "RYNE_4498_CTAB_20261006234519.csv",  # H5 - celowo niewykorzystywany
    "RACH_GDP_NOMINAL": "RACH_3498_XTAB_20261007010718.xlsx",
    "RACH_GDP_REAL": "RACH_3502_XTAB_20261007010836.xlsx",
    "RACH_GDP_REAL_PC": "RACH_3503_XTAB_20261007010812.xlsx",
    "CENY": "CENY_2496_XTAB_20261007223438.xlsx",
}


def detect_file(prefix):
    prefix = prefix.upper()
    exact_name = REPOSITORY_FILES.get(prefix)

    if exact_name:
        exact_path = APP_DIR / exact_name
        if exact_path.exists():
            return str(exact_path)

    matches = sorted(
        p for p in APP_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".csv", ".xlsx", ".xls", ".xlsm"}
        and p.name.upper().startswith(prefix)
    )

    return str(matches[-1]) if matches else None


def get_local_data_files():
    extensions = {".csv", ".xlsx", ".xls", ".xlsm"}
    return [
        str(p)
        for p in APP_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
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

def find_year_columns(df, year, include=None, exclude=None):
    """Znajduje kolumny GUS w szerokim układzie zawierające dany rok."""
    include = include or []
    exclude = exclude or []
    year = str(year)
    result = []

    for col in df.columns:
        name = normalize_col_name(col).lower()
        if year not in name:
            continue
        if include and not all(str(x).lower() in name for x in include):
            continue
        if exclude and any(str(x).lower() in name for x in exclude):
            continue
        result.append(col)

    return result


def select_gus_wide_column(df, year, include_groups, preferred_groups=None, exclude=None):
    """Wybiera najlepszą kolumnę wartości dla danego roku z tabeli GUS."""
    exclude = exclude or ["wskaźnik precyzji", "precyzji"]
    preferred_groups = preferred_groups or []

    candidates = find_year_columns(
        df,
        year,
        include=[],
        exclude=exclude,
    )

    def score(col):
        name = normalize_col_name(col).lower()
        score_value = 0
        for group in include_groups:
            group = str(group).lower()
            if group in name:
                score_value += 10
        for group in preferred_groups:
            group = str(group).lower()
            if group in name:
                score_value += 100
        if "wartość liczbowa" in name:
            score_value += 30
        if "ogółem" in name:
            score_value += 50
        return score_value

    if not candidates:
        return None

    candidates = sorted(candidates, key=score, reverse=True)
    return candidates[0]


def wide_gus_to_long(df, value_name, selector, years=None):
    """Konwertuje szeroką tabelę GUS Kod/Nazwa + kolumny roczne do panelu Kod/Rok."""
    years = years or range(2010, 2026)
    rows = []

    kod_col = find_column(df, "Kod", required=True)

    for year in years:
        col = selector(df, year)
        if col is None:
            continue

        part = df[[kod_col, col]].copy()
        part.columns = ["Kod", value_name]
        part["Rok"] = int(year)
        rows.append(part)

    if not rows:
        raise ValueError(
            f"Nie udało się znaleźć żadnych kolumn rocznych dla zmiennej '{value_name}'. "
            f"Dostępne kolumny: {list(df.columns)}"
        )

    out = pd.concat(rows, ignore_index=True)
    out["Kod"] = out["Kod"].astype(str).str.strip()
    out[value_name] = pd.to_numeric(
        out[value_name].astype(str)
        .str.replace("\\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )
    return out.drop_duplicates(["Kod", "Rok"])


def make_year_panel_from_ryne(ryne_sector, ryne_unemployment, ryne_employment):
    """Buduje wspólny panel z trzech tabel RYNE w formacie szerokim."""

    def sector_selector(df, year):
        # RYNE 4098: odsetek pracujących wg sektorów ekonomicznych i płci.
        # Bierzemy sektor przemysłowy, ogółem, wartość liczbowa.
        exact = [
            c for c in df.columns
            if normalize_col_name(c).lower()
            == f"sektor przemysłowy;ogółem;wartość liczbowa;{year};[%]".lower()
        ]
        if exact:
            return exact[0]

        return select_gus_wide_column(
            df,
            year,
            include_groups=["sektor przemysłowy", "wartość liczbowa"],
            preferred_groups=["ogółem"],
        )

    def employment_selector(df, year):
        # RYNE 4112: wskaźnik zatrudnienia wg wieku i płci.
        # Preferujemy ogółem i grupę 15-64, a następnie 15-89.
        candidates = find_year_columns(
            df,
            year,
            exclude=["wskaźnik precyzji", "precyzji"],
        )
        candidates = [
            c for c in candidates
            if "wartość liczbowa" in normalize_col_name(c).lower()
        ]

        preference = [
            ["15-64", "ogółem"],
            ["15-89", "ogółem"],
            ["18-59/64", "ogółem"],
            ["ogółem"],
        ]

        for pref in preference:
            matches = [
                c for c in candidates
                if all(x.lower() in normalize_col_name(c).lower() for x in pref)
            ]
            if matches:
                return matches[0]

        return candidates[0] if candidates else None

    def unemployment_selector(df, year):
        # RYNE 4100: stopa bezrobocia wg wieku.
        # Preferujemy ogółem dla grupy 15-74, następnie 15-89.
        candidates = find_year_columns(
            df,
            year,
            exclude=["wskaźnik precyzji", "precyzji"],
        )
        candidates = [
            c for c in candidates
            if "wartość liczbowa" in normalize_col_name(c).lower()
        ]

        preference = [
            ["15-74", "ogółem"],
            ["15-89", "ogółem"],
            ["ogółem"],
        ]

        for pref in preference:
            matches = [
                c for c in candidates
                if all(x.lower() in normalize_col_name(c).lower() for x in pref)
            ]
            if matches:
                return matches[0]

        return candidates[0] if candidates else None

    sector = wide_gus_to_long(
        ryne_sector,
        "Udzial_Przemyslu",
        sector_selector,
        years=range(2021, 2026),
    )

    employment = wide_gus_to_long(
        ryne_employment,
        "Stopa_Zatrudnienia",
        employment_selector,
        years=range(2010, 2026),
    )

    unemployment = wide_gus_to_long(
        ryne_unemployment,
        "Bezrobocie_Rejestrowane",
        unemployment_selector,
        years=range(2010, 2025),
    )

    out = employment.merge(
        unemployment,
        on=["Kod", "Rok"],
        how="inner",
    )

    out = out.merge(
        sector,
        on=["Kod", "Rok"],
        how="inner",
    )

    return out


def reshape_rach_series(df, value_name, selector, years=range(2010, 2025)):
    kod_col = find_column(df, "Kod", required=True)
    rows = []
    for year in years:
        col = selector(df, year)
        if col is None:
            continue
        part = df[[kod_col, col]].copy()
        part.columns = ["Kod", value_name]
        part["Rok"] = int(year)
        rows.append(part)

    if not rows:
        raise ValueError(
            f"Nie znaleziono danych RACH dla zmiennej '{value_name}'. "
            f"Dostępne kolumny: {list(df.columns)}"
        )

    out = pd.concat(rows, ignore_index=True)
    out["Kod"] = out["Kod"].astype(str).str.strip()
    out[value_name] = pd.to_numeric(
        out[value_name].astype(str)
        .str.replace("\\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )
    return out.drop_duplicates(["Kod", "Rok"])


def rach_selector(df, year, keywords, preferred=None, exclude=None):
    return select_gus_wide_column(
        df,
        year,
        include_groups=keywords,
        preferred_groups=preferred or [],
        exclude=exclude or ["wskaźnik precyzji", "precyzji"],
    )


def build_rach_panel(rach_nominal, rach_real, rach_real_pc):
    """Buduje panel RACH z właściwych tabel GUS:
    3498 = PKB nominalne ogółem,
    3502 = PKB realne ogółem,
    3503 = PKB realne na 1 mieszkańca.
    """

    def nominal_selector(df, year):
        candidates = find_year_columns(df, year, exclude=["dynamika", "wskaźnik"])
        preferred = [
            c for c in candidates
            if "produkt krajowy brutto" in normalize_col_name(c).lower()
            and "ogółem" in normalize_col_name(c).lower()
        ]
        return preferred[0] if preferred else (
            candidates[0] if candidates else None
        )

    def real_selector(df, year):
        candidates = find_year_columns(df, year, exclude=["dynamika", "wskaźnik"])
        preferred = [
            c for c in candidates
            if "produkt krajowy brutto" in normalize_col_name(c).lower()
            and "ogółem" in normalize_col_name(c).lower()
        ]
        return preferred[0] if preferred else (
            candidates[0] if candidates else None
        )

    def real_pc_selector(df, year):
        candidates = find_year_columns(df, year, exclude=["dynamika", "wskaźnik"])
        preferred = [
            c for c in candidates
            if "produkt krajowy brutto" in normalize_col_name(c).lower()
            and "1 mieszkańca" in normalize_col_name(c).lower()
        ]
        return preferred[0] if preferred else (
            candidates[0] if candidates else None
        )

    nominal = reshape_rach_series(
        rach_nominal,
        "PKB_Nominalny",
        nominal_selector,
        years=range(2010, 2025),
    )

    real = reshape_rach_series(
        rach_real,
        "Realny_PKB",
        real_selector,
        years=range(2010, 2025),
    )

    real_pc = reshape_rach_series(
        rach_real_pc,
        "Realny_PKB_per_capita",
        real_pc_selector,
        years=range(2010, 2025),
    )

    out = nominal.merge(real, on=["Kod", "Rok"], how="inner")
    out = out.merge(real_pc, on=["Kod", "Rok"], how="inner")
    return out


def reshape_ceny_file(ceny_df):
    """Obsługuje CENY w układzie klasycznym lub szerokim."""
    rok_col = find_column(ceny_df, "Rok", required=False)
    cpi_col = find_column(ceny_df, "Wskaznik_CPI", required=False)

    if rok_col is not None and cpi_col is not None:
        out = ceny_df[[rok_col, cpi_col]].copy()
        out.columns = ["Rok", "Wskaznik_CPI"]
        return out

    # W szerokim eksporcie GUS rok może być elementem nazwy kolumny.
    candidates = []
    for col in ceny_df.columns:
        name = normalize_col_name(col).lower()
        if any(token in name for token in ["wskaźnik", "cpi", "ceny towarów"]):
            candidates.append(col)

    if not candidates:
        # Ostatnia próba: kolumny zawierające rok i 100 jako bazę.
        candidates = [
            c for c in ceny_df.columns
            if any(str(y) in normalize_col_name(c) for y in range(2010, 2026))
        ]

    rows = []
    for year in range(2010, 2026):
        year_cols = [c for c in candidates if str(year) in normalize_col_name(c)]
        if not year_cols:
            continue
        col = year_cols[0]
        part = pd.DataFrame({
            "Rok": [year] * len(ceny_df),
            "Wskaznik_CPI": ceny_df[col].values,
        })
        rows.append(part)

    if not rows:
        raise ValueError(
            "Nie udało się rozpoznać rocznych wartości CPI w pliku CENY. "
            f"Dostępne kolumny: {list(ceny_df.columns)}"
        )

    out = pd.concat(rows, ignore_index=True)
    out["Wskaznik_CPI"] = pd.to_numeric(
        out["Wskaznik_CPI"].astype(str)
        .str.replace("\\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )
    return out


@st.cache_data(show_spinner=False)
def load_and_clean_data(
    ryne_sector_path,
    ryne_unemployment_path,
    ryne_employment_path,
    rach_nominal_path,
    rach_real_path,
    rach_real_pc_path,
    ceny_path,
):
    ryne_sector_df = read_table(ryne_sector_path)
    ryne_unemployment_df = read_table(ryne_unemployment_path)
    ryne_employment_df = read_table(ryne_employment_path)
    rach_nominal_df = read_table(rach_nominal_path)
    rach_real_df = read_table(rach_real_path)
    rach_real_pc_df = read_table(rach_real_pc_path)
    ceny_df = read_table(ceny_path)

    ryne_df = make_year_panel_from_ryne(
        ryne_sector_df,
        ryne_unemployment_df,
        ryne_employment_df,
    )

    rach_df = build_rach_panel(
        rach_nominal_df,
        rach_real_df,
        rach_real_pc_df,
    )
    ceny_df = reshape_ceny_file(ceny_df)

    # Jeśli RACH ma tylko PKB i per capita, udział przemysłu pochodzi z RYNE.
    # Łączenie następuje po województwie/kodzie i roku.
    for frame in [ryne_df, rach_df]:
        frame["Kod"] = frame["Kod"].astype(str).str.strip()
        frame["Kod_Str"] = frame["Kod"].map(extract_teryt)

    rach_df = rach_df.drop_duplicates(["Kod", "Rok"])
    ryne_df = ryne_df.drop_duplicates(["Kod", "Rok"])

    df = ryne_df.merge(
        rach_df,
        on=["Kod", "Rok"],
        how="inner",
        suffixes=("", "_rach"),
    )

    ceny_df["Rok"] = pd.to_numeric(ceny_df["Rok"], errors="coerce")
    ceny_df["Wskaznik_CPI"] = pd.to_numeric(
        ceny_df["Wskaznik_CPI"], errors="coerce"
    )
    ceny_df = ceny_df.drop_duplicates(["Rok"])

    df = df.merge(ceny_df, on="Rok", how="inner")

    df["Wojewodztwo"] = df["Kod_Str"].map(WOJEWODZTWA_MAP)
    df = df[df["Wojewodztwo"].notna()].copy()

    df["Przecietne_Wynagrodzenie_Kraj"] = df["Rok"].map(
        PRZECIETNE_WYNAGRODZENIE_PL
    )
    df["Placa_Minimalna"] = df["Rok"].map(MIN_WAGE_PL)

    df["CPI"] = pd.to_numeric(df["Wskaznik_CPI"], errors="coerce")
    df["CPI_factor"] = df["CPI"] / 100.0

    df["Realna_Placa_Minimalna"] = (
        df["Placa_Minimalna"] / df["CPI_factor"]
    )
    df["Realne_Wynagrodzenie"] = (
        df["Przecietne_Wynagrodzenie_Kraj"] / df["CPI_factor"]
    )
    # Realny PKB pochodzi bezpośrednio z tabeli GUS RACH 3502
    # (PKB w cenach stałych), więc nie deflujemy go ponownie CPI.

    df["Kaitz_Index"] = (
        df["Realna_Placa_Minimalna"] /
        df["Realne_Wynagrodzenie"]
    )

    df = df.sort_values(["Kod_Str", "Rok"])

    df["Wzrost_PKB"] = (
        df.groupby("Kod_Str")["Realny_PKB"]
        .pct_change(fill_method=None) * 100
    )

    # Udział przemysłu z RYNE 4098 jest już procentem.
    # Przeliczamy na udział 0-1 dla modelu.
    if df["Udzial_Przemyslu"].dropna().abs().median() > 1.5:
        df["Udzial_Przemyslu"] = df["Udzial_Przemyslu"] / 100.0

    # Jeżeli mamy liczbę pracujących, możemy policzyć produktywność.
    # Gdy RACH jej nie zawiera, pozostawiamy NaN i usuwamy ją z X tylko później,
    # zamiast tworzyć sztuczną wartość.
    if "Zatrudnieni_Ogolem" in df.columns:
        df["Produktywnosc"] = np.where(
            pd.to_numeric(df["Zatrudnieni_Ogolem"], errors="coerce") != 0,
            df["Realny_PKB"] /
            pd.to_numeric(df["Zatrudnieni_Ogolem"], errors="coerce"),
            np.nan,
        )
    else:
        df["Produktywnosc"] = np.nan

    if "Populacja" not in df.columns:
        df["Populacja"] = np.nan

    # Nie rekonstruujemy automatycznie PKB per capita z niezweryfikowanych
    # jednostek ludności. Jeżeli RACH dostarcza tę zmienną, zachowujemy ją.
    if "Realny_PKB_per_capita" not in df.columns:
        df["Realny_PKB_per_capita"] = np.nan

    model_cols = [
        "Wojewodztwo", "Kod_Str", "Rok", "Kaitz_Index",
        "Stopa_Zatrudnienia", "Bezrobocie_Rejestrowane",
        "Wzrost_PKB", "Realny_PKB_per_capita",
        "Udzial_Przemyslu", "Produktywnosc", "Populacja", "CPI",
    ]

    numeric_cols = [
        c for c in model_cols
        if c not in ["Wojewodztwo", "Kod_Str"]
    ]
    df = coerce_numeric(df, numeric_cols)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Produktywność i PKB per capita mogą nie być dostępne w aktualnym
    # eksporcie RACH. Nie blokujemy przez to całej aplikacji.
    required_for_model = [
        "Wojewodztwo", "Kod_Str", "Rok", "Kaitz_Index",
        "Stopa_Zatrudnienia", "Bezrobocie_Rejestrowane",
        "Wzrost_PKB", "Udzial_Przemyslu", "CPI",
    ]
    df = df.dropna(subset=required_for_model).copy()

    # Uzupełnienie kontrolnych zmiennych, jeśli RACH ich nie podał.
    if df["Realny_PKB_per_capita"].isna().all():
        # Przy braku wiarygodnego PKB per capita nie używamy tej zmiennej w CF.
        df["Realny_PKB_per_capita"] = np.nan

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

X_COLS_BASE = [
    "Bezrobocie_Rejestrowane",
    "Wzrost_PKB",
    "Realny_PKB_per_capita",
    "Udzial_Przemyslu",
    "Produktywnosc",
]

W_COLS_BASE = [
    "Populacja",
    "CPI",
]

X_COLS = X_COLS_BASE
W_COLS = W_COLS_BASE


def get_available_model_columns(df):
    x_cols = [c for c in X_COLS_BASE if c in df.columns and not df[c].isna().all()]
    w_cols = [c for c in W_COLS_BASE if c in df.columns and not df[c].isna().all()]
    return x_cols, w_cols


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

    x_cols, w_cols = get_available_model_columns(work)

    if len(x_cols) < 2:
        raise ValueError(
            "Za mało dostępnych zmiennych heterogeniczności dla Causal Forest. "
            f"Dostępne: {x_cols}"
        )

    X = work[x_cols].copy()
    W = work[w_cols].copy() if w_cols else None

    model_frame = pd.concat([X, W] if W is not None else [X], axis=1)
    valid = model_frame.notna().all(axis=1)

    Y = Y[valid.to_numpy()]
    T = T[valid.to_numpy()]
    X = X.loc[valid].copy()
    if W is not None:
        W = W.loc[valid].copy()
    work = work.loc[valid].copy()

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
        "Bezrobocie_Rejestrowane": "Stopa bezrobocia (BAEL)",
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

# Dane są częścią repozytorium. Aplikacja nie wymaga żadnego
# ręcznego wgrywania plików przez użytkownika.
auto_ryne_sector = detect_file("RYNE_SECTOR")
auto_ryne_unemployment = detect_file("RYNE_UNEMPLOYMENT")
auto_ryne_employment = detect_file("RYNE_EMPLOYMENT")
auto_rach_nominal = detect_file("RACH_GDP_NOMINAL")
auto_rach_real = detect_file("RACH_GDP_REAL")
auto_rach_real_pc = detect_file("RACH_GDP_REAL_PC")
auto_ceny = detect_file("CENY")

ryne_sector_path = auto_ryne_sector
ryne_unemployment_path = auto_ryne_unemployment
ryne_employment_path = auto_ryne_employment
rach_nominal_path = auto_rach_nominal
rach_real_path = auto_rach_real
rach_real_pc_path = auto_rach_real_pc
ceny_path = auto_ceny

st.sidebar.markdown("---")
st.sidebar.caption("Automatycznie wykryte pliki")

st.sidebar.write(
    f"**RYNE 4098 — sektor:** "
    f"{Path(ryne_sector_path).name if ryne_sector_path else 'brak'}"
)

st.sidebar.write(
    f"**RYNE 4100 — bezrobocie:** "
    f"{Path(ryne_unemployment_path).name if ryne_unemployment_path else 'brak'}"
)

st.sidebar.write(
    f"**RYNE 4112 — zatrudnienie:** "
    f"{Path(ryne_employment_path).name if ryne_employment_path else 'brak'}"
)

st.sidebar.write(
    f"**RACH 3498 — PKB nominalne:** "
    f"{Path(rach_nominal_path).name if rach_nominal_path else 'brak'}"
)

st.sidebar.write(
    f"**RACH 3502 — PKB realne:** "
    f"{Path(rach_real_path).name if rach_real_path else 'brak'}"
)

st.sidebar.write(
    f"**RACH 3503 — PKB realne per capita:** "
    f"{Path(rach_real_pc_path).name if rach_real_pc_path else 'brak'}"
)

st.sidebar.write(
    f"**CENY:** "
    f"{Path(ceny_path).name if ceny_path else 'brak'}"
)

if all([
    ryne_sector_path,
    ryne_unemployment_path,
    ryne_employment_path,
    rach_nominal_path,
    rach_real_path,
    rach_real_pc_path,
    ceny_path,
]):
    st.sidebar.success("Dane z repozytorium są gotowe.")
else:
    st.sidebar.error("Nie znaleziono kompletu danych w repozytorium.")

run_analysis = st.sidebar.button(
    "🚀 Uruchom pełną analizę",
    type="primary",
    use_container_width=True,
)

if run_analysis:

    if not all(
        [
            ryne_sector_path,
            ryne_unemployment_path,
            ryne_employment_path,
            rach_nominal_path,
            rach_real_path,
            rach_real_pc_path,
            ceny_path,
        ]
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
                ryne_sector_path,
                ryne_unemployment_path,
                ryne_employment_path,
                rach_nominal_path,
                rach_real_path,
                rach_real_pc_path,
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
                "Stopa bezrobocia (BAEL)"
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
                    "Zmienna": res["X_data"].columns,
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
                    "Stopa bezrobocia (BAEL)",
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
        "Dane RYNE, RACH i CENY są automatycznie pobierane z repozytorium. "
        "Kliknij **Uruchom pełną analizę**."
    )

st.sidebar.markdown("---")

st.sidebar.caption(
    "Model: CausalForestDML + XGBoost | Seed: 42 | "
    "H5 (młodzież) wyłączona z analizy"
)
