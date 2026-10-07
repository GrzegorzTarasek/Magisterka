import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import statsmodels.formula.api as smf
import xgboost as xgb
import shap

try:
    from econml.dml import CausalForestDML
except Exception as exc:
    CausalForestDML = None
    ECONML_IMPORT_ERROR = exc
else:
    ECONML_IMPORT_ERROR = None

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Magisterka – płaca minimalna i zatrudnienie",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEED = 42
APP_DIR = Path(__file__).resolve().parent

# ============================================================
# 1. PARAMETRY DANYCH
# ============================================================

MIN_WAGE_PL = {
    2010: 1317, 2011: 1386, 2012: 1500, 2013: 1600, 2014: 1680,
    2015: 1750, 2016: 1850, 2017: 2000, 2018: 2100, 2019: 2250,
    2020: 2600, 2021: 2800, 2022: 3010, 2023: 3490, 2024: 4242,
    2025: 4666,
}

WOJEWODZTWA_MAP = {
    "02": "Dolnośląskie",
    "04": "Kujawsko-Pomorskie",
    "06": "Lubelskie",
    "08": "Lubuskie",
    "10": "Łódzkie",
    "12": "Małopolskie",
    "14": "Mazowieckie",
    "16": "Opolskie",
    "18": "Podkarpackie",
    "20": "Podlaskie",
    "22": "Pomorskie",
    "24": "Śląskie",
    "26": "Świętokrzyskie",
    "28": "Warmińsko-Mazurskie",
    "30": "Wielkopolskie",
    "32": "Zachodniopomorskie",
}

# Tylko dane faktycznie potrzebne do głównej analizy.
# Pozostałe pliki z repo mogą istnieć, ale nie są wymagane.
REPOSITORY_FILES = {
    "RYNE_UNEMPLOYMENT": "RYNE_4100_CTAB_20261006234212.csv",
    "RYNE_EMPLOYMENT": "RYNE_4112_CTAB_20261006234336.csv",
    "WYNA_REGIONAL": "WYNA_2504_XTAB_20261007010213.xlsx",
    "RACH_GDP_GROWTH": "RACH_3502_XTAB_20261007010836.xlsx",
    "RACH_GVA_SECTOR": "RACH_3505_XTAB_20261007011009.xlsx",
    "RACH_PRODUCTIVITY": "RACH_3510_XTAB_20261007010935.xlsx",
}

CORE_SOURCE_LABELS = {
    "RYNE_UNEMPLOYMENT": "RYNE 4100 – stopa bezrobocia BAEL",
    "RYNE_EMPLOYMENT": "RYNE 4112 – wskaźnik zatrudnienia BAEL",
    "WYNA_REGIONAL": "WYNA 2504 – przeciętne wynagrodzenie regionalne",
    "RACH_GDP_GROWTH": "RACH 3502 – dynamika realnego PKB",
    "RACH_GVA_SECTOR": "RACH 3505 – WDB wg sektorów",
    "RACH_PRODUCTIVITY": "RACH 3510 – WDB na 1 pracującego",
}


# ============================================================
# 2. PLIKI Z REPOZYTORIUM – ZERO UPLOADU
# ============================================================


def detect_file(key):
    expected = REPOSITORY_FILES[key]
    exact = APP_DIR / expected
    if exact.exists():
        return exact

    # Fallback jest ograniczony do konkretnego identyfikatora tabeli,
    # np. RACH_3502_, a nie wszystkich plików RACH_.
    stem_match = "_".join(expected.split("_")[:2]).upper() + "_"
    candidates = sorted(
        p for p in APP_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".csv", ".xlsx", ".xls", ".xlsm"}
        and p.name.upper().startswith(stem_match)
    )
    return candidates[-1] if candidates else None


# ============================================================
# 3. NARZĘDZIA
# ============================================================


def normalize_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\ufeff", "").strip())


def normalize_low(value):
    return normalize_text(value).lower()


def numeric_series(series):
    if series.dtype == object:
        series = (
            series.astype(str)
            .str.replace("\xa0", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.replace("−", "-", regex=False)
            .str.replace("–", "-", regex=False)
        )
    return pd.to_numeric(series, errors="coerce")


def year_from_text(value):
    text = normalize_text(value)
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)
    return int(match.group(1)) if match else None


def get_years_from_columns(columns):
    return sorted({
        year_from_text(c)
        for c in columns
        if year_from_text(c) is not None and 1900 <= year_from_text(c) <= 2100
    })


REGION_NAME_TO_CODE = {normalize_low(name): code for code, name in WOJEWODZTWA_MAP.items()}

def extract_teryt(value):
    if pd.isna(value):
        return np.nan
    text = normalize_text(value)
    digits = re.sub(r"\D", "", text)
    if not digits:
        return np.nan
    # GUS/Excel can remove a leading zero from a 7-digit TERYT code.
    # Handle common representations explicitly.
    if len(digits) == 2 and digits in WOJEWODZTWA_MAP:
        return digits
    if len(digits) >= 7:
        candidate = digits[:2]
        if candidate in WOJEWODZTWA_MAP:
            return candidate
    if len(digits) == 6 and ("0" + digits[:1]) in WOJEWODZTWA_MAP:
        return "0" + digits[:1]
    # A six/seven digit numeric code with a lost leading zero is ambiguous;
    # resolve it later from the regional name whenever possible.
    if len(digits) == 6:
        candidate = digits[:2]
        if candidate in WOJEWODZTWA_MAP:
            return candidate
    return np.nan

def code_from_region_name(value):
    text = normalize_low(value)
    if not text:
        return np.nan
    text = re.sub(r"^województwo\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return REGION_NAME_TO_CODE.get(text, np.nan)

def is_region_code(value):
    return pd.notna(extract_teryt(value))


def normalize_region_name(value):
    text = normalize_low(value)
    if not text:
        return np.nan
    text = re.sub(r"^województwo\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    code = REGION_NAME_TO_CODE.get(text)
    return WOJEWODZTWA_MAP.get(code, np.nan)


def add_region_key(df):
    out = df.copy()
    if "Kod" not in out.columns:
        raise ValueError("Brak kolumny Kod w tabeli GUS.")

    code_from_name = (
        out["Nazwa"].map(code_from_region_name)
        if "Nazwa" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    code_from_code = out["Kod"].map(extract_teryt)

    # Nazwa jest używana jako pierwszy klucz, bo Excel potrafi usunąć
    # początkowe zero z 7-cyfrowego kodu TERYT.
    out["Kod"] = code_from_name.fillna(code_from_code)
    out["Kod"] = out["Kod"].astype("object")
    out["Wojewodztwo"] = out["Kod"].map(WOJEWODZTWA_MAP)

    if "Nazwa" in out.columns:
        fallback_name = out["Nazwa"].map(normalize_region_name)
        out["Wojewodztwo"] = out["Wojewodztwo"].fillna(fallback_name)

    return out


def col_has_year(col, year):
    return bool(re.search(rf"(?<!\d){int(year)}(?!\d)", normalize_text(col)))


def clean_key_panel(df):
    out = add_region_key(df.copy())
    if "Rok" not in out.columns:
        raise ValueError("Brak kolumny Rok po odczytaniu tabeli GUS.")
    out["Rok"] = pd.to_numeric(out["Rok"], errors="coerce")
    out = out.dropna(subset=["Wojewodztwo", "Rok"]).copy()
    out["Rok"] = out["Rok"].astype(int)
    out["Kod"] = out["Wojewodztwo"].map({v: k for k, v in WOJEWODZTWA_MAP.items()})
    return out.drop_duplicates(["Wojewodztwo", "Rok"])


# ============================================================
# 4. ODCZYT GUS XLSX – WIELOWIERSZOWE NAGŁÓWKI
# ============================================================


def find_code_row(raw):
    for i in range(min(len(raw), 120)):
        values = [normalize_low(v) for v in raw.iloc[i].tolist()]
        has_code = any(v == "kod" or v.startswith("kod ") for v in values)
        has_name = any(v == "nazwa" or v.startswith("nazwa ") for v in values)
        if has_code and has_name:
            return i
    return None


def find_data_start(raw, code_row):
    for i in range(code_row + 1, min(len(raw), code_row + 50)):
        if raw.shape[1] < 2:
            continue
        if is_region_code(raw.iat[i, 0]) and normalize_text(raw.iat[i, 1]):
            return i
    return code_row + 1


def build_xlsx_headers(raw, code_row, data_start):
    # Bierzemy kilkanaście wierszy powyżej tabeli, ponieważ GUS ma różne
    # warianty nagłówków. W osobnym wierszu najczęściej zapisane są lata.
    start = max(0, code_row - 15)
    header = raw.iloc[start:data_start].copy()
    if header.empty:
        return [f"col_{j}" for j in range(raw.shape[1])]

    # Lata odczytujemy przed ffill, żeby nie zgubić początku kolejnych grup.
    year_map = [None] * raw.shape[1]
    for j in range(raw.shape[1]):
        found = []
        for i in range(header.shape[0]):
            y = year_from_text(header.iat[i, j])
            if y is not None and 1900 <= y <= 2100:
                found.append(y)
        if found:
            # Ostatni jawnie zapisany rok w danej kolumnie.
            year_map[j] = found[-1]

    # Opisy grup bywają zapisane tylko w pierwszej kolumnie grupy.
    labels = header.map(normalize_text).replace("", np.nan).ffill(axis=1).ffill(axis=0)

    columns = []
    used = set()
    for j in range(raw.shape[1]):
        direct = normalize_low(raw.iat[code_row, j])
        if direct == "kod" or direct.startswith("kod "):
            name = "Kod"
        elif direct == "nazwa" or direct.startswith("nazwa "):
            name = "Nazwa"
        else:
            parts = []
            for i in range(labels.shape[0]):
                val = normalize_text(labels.iat[i, j])
                if not val or val.lower() in {"kod", "nazwa"}:
                    continue
                if year_from_text(val) is not None:
                    continue
                if val not in parts:
                    parts.append(val)
            if year_map[j] is not None:
                parts.append(str(year_map[j]))
            name = " | ".join(parts)
            if not name:
                name = f"col_{j}"

        base = name
        n = 2
        while name in used:
            name = f"{base}__{n}"
            n += 1
        used.add(name)
        columns.append(name)

    return columns


@st.cache_data(show_spinner=False)
def read_excel_gus(path):
    xls = pd.ExcelFile(path, engine="openpyxl")
    candidates = []

    for sheet in xls.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet, header=None, engine="openpyxl")
        if raw.empty:
            continue

        code_row = find_code_row(raw)
        if code_row is None:
            continue
        data_start = find_data_start(raw, code_row)
        columns = build_xlsx_headers(raw, code_row, data_start)

        data = raw.iloc[data_start:].copy()
        data.columns = columns
        data = data.dropna(how="all")
        if data.empty:
            continue

        # Tylko niepuste kolumny.
        data = data.loc[:, ~data.isna().all(axis=0)].copy()
        data.columns = [normalize_text(c) for c in data.columns]

        score = 0
        if "Kod" in data.columns:
            score += 10000
        if "Nazwa" in data.columns:
            score += 1000
        score += 20 * len(get_years_from_columns(data.columns))
        score += min(data.shape[1], 200)
        candidates.append((score, sheet, data))

    if not candidates:
        raise ValueError(
            f"Nie znaleziono właściwej tabeli GUS w pliku {Path(path).name}. "
            "Nie znaleziono nagłówka Kod/Nazwa."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][2]


# ============================================================
# 5. ODCZYT GUS CSV
# ============================================================


@st.cache_data(show_spinner=False)
def read_csv_gus(path):
    best = None
    encodings = ["utf-8-sig", "utf-8", "cp1250", "windows-1250", "latin1"]
    separators = [";", ",", "\t"]

    for enc in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(
                    path,
                    encoding=enc,
                    sep=sep,
                    engine="python",
                    quotechar='"',
                )
                df.columns = [normalize_text(c) for c in df.columns]
                lows = [normalize_low(c) for c in df.columns]
                score = 0
                if "kod" in lows:
                    score += 10000
                if "nazwa" in lows:
                    score += 1000
                score += 20 * len(get_years_from_columns(df.columns))
                score += min(df.shape[1], 200)
                if best is None or score > best[0]:
                    best = (score, df)
            except Exception:
                continue

    if best is None:
        raise ValueError(f"Nie udało się odczytać CSV {Path(path).name}.")
    return best[1]


def read_table(path):
    suffix = Path(path).suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return read_excel_gus(str(path))
    if suffix == ".csv":
        return read_csv_gus(str(path))
    raise ValueError(f"Nieobsługiwany format pliku: {suffix}")


# ============================================================
# 6. SELEKCJA KOLUMN – HEURYSTYKA ODPORNA NA ZMIANY GUS
# ============================================================


def choose_year_column(
    df,
    year,
    include_all=(),
    include_any=(),
    exclude=(),
    prefer=(),
):
    candidates = []
    for col in df.columns:
        text = normalize_low(col)
        if not col_has_year(col, year):
            continue
        if any(term.lower() not in text for term in include_all):
            continue
        if include_any and not any(term.lower() in text for term in include_any):
            continue
        if any(term.lower() in text for term in exclude):
            continue

        score = 0
        for term in prefer:
            if term.lower() in text:
                score += 50
        if "wartość liczbowa" in text:
            score += 100
        if "wskaźnik precyzji" in text or "precyzji" in text:
            score -= 1000
        if "polska=100" in text or "polska = 100" in text:
            score -= 900
        if "%" in text and "wartość liczbowa" not in text:
            score -= 300
        candidates.append((score, col))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def panel_from_selector(df, value_name, selector, years):
    if "Kod" not in df.columns:
        raise ValueError(
            f"Tabela nie zawiera kolumny Kod. Dostępne: {list(df.columns)[:25]}"
        )

    rows = []
    used = []
    for year in years:
        col = selector(df, int(year))
        if col is None:
            continue
        keep_cols = ["Kod"]
        if "Nazwa" in df.columns:
            keep_cols.append("Nazwa")
        keep_cols.append(col)
        part = df[keep_cols].copy()
        rename_map = {"Kod": "Kod", col: value_name}
        if "Nazwa" in part.columns:
            rename_map["Nazwa"] = "Nazwa"
        part = part.rename(columns=rename_map)
        part["Rok"] = int(year)
        part[value_name] = numeric_series(part[value_name])
        rows.append(part)
        used.append((int(year), col))

    if not rows:
        raise ValueError(
            f"Nie znaleziono kolumn dla '{value_name}'. "
            f"Dostępne lata w tabeli: {get_years_from_columns(df.columns)}"
        )

    out = pd.concat(rows, ignore_index=True)
    out = clean_key_panel(out)
    return out, used


# ============================================================
# 7. RYNE – BAEL
# ============================================================


def build_ryne_panel(unemployment_df, employment_df):
    def unemployment_selector(df, year):
        return choose_year_column(
            df,
            year,
            include_all=("ogółem",),
            include_any=("wartość liczbowa",),
            exclude=("precyzji", "polska=100", "polska = 100", "15-24", "15-29", "15-64"),
            prefer=("wartość liczbowa", "ogółem"),
        )

    def employment_selector(df, year):
        return choose_year_column(
            df,
            year,
            include_all=("ogółem",),
            include_any=("wartość liczbowa",),
            exclude=("precyzji", "polska=100", "polska = 100", "15-24", "15-29", "15-64"),
            prefer=("wartość liczbowa", "ogółem"),
        )

    years_u = range(2000, 2026)
    years_e = range(2000, 2026)
    unemployment, _ = panel_from_selector(
        unemployment_df,
        "Stopa_Bezrobocia_BAEL",
        unemployment_selector,
        years_u,
    )
    employment, _ = panel_from_selector(
        employment_df,
        "Stopa_Zatrudnienia",
        employment_selector,
        years_e,
    )

    merged = employment.merge(
        unemployment[["Wojewodztwo", "Rok", "Stopa_Bezrobocia_BAEL"]],
        on=["Wojewodztwo", "Rok"],
        how="inner",
    )
    merged["Kod"] = merged["Wojewodztwo"].map({v: k for k, v in WOJEWODZTWA_MAP.items()})
    return merged, {
        "employment_years": sorted(employment["Rok"].unique()),
        "unemployment_years": sorted(unemployment["Rok"].unique()),
    }


# ============================================================
# 8. REGIONALNE WYNAGRODZENIE – KAITZ
# ============================================================


def build_regional_wages(wyna_df):
    def selector(df, year):
        candidates = []
        for col in df.columns:
            text = normalize_low(col)
            if not col_has_year(col, year):
                continue
            if "rok" not in text and "wynagrodzenia" not in text and "wynagrodzenie" not in text:
                continue
            if "wskaźnik precyzji" in text or "precyzji" in text:
                continue
            if "%" in text and "wartość liczbowa" not in text:
                continue

            score = 0
            if "wynagrodzenia ogółem" in text or "wynagrodzenie ogółem" in text:
                score += 300
            if "bez wypłat nagród rocznych" in text:
                score += 50
            if "wartość liczbowa" in text:
                score += 100
            candidates.append((score, col))

        if not candidates:
            return choose_year_column(
                df,
                year,
                include_all=("rok",),
                exclude=("precyzji", "%"),
                prefer=("wynagrodzenia ogółem", "wynagrodzenie ogółem", "wartość liczbowa"),
            )
        return sorted(candidates, reverse=True)[0][1]

    wages, used = panel_from_selector(
        wyna_df,
        "Przecietne_Wynagrodzenie_Regionalne",
        selector,
        range(2000, 2026),
    )
    wages = wages[wages["Przecietne_Wynagrodzenie_Regionalne"] > 0].copy()
    return wages, {"columns": used}


# ============================================================
# 9. RACH 3502 – DYNAMIKA REALNEGO PKB
# ============================================================


def build_gdp_growth(gdp_df):
    def selector(df, year):
        return choose_year_column(
            df,
            year,
            include_all=("dynamika",),
            include_any=(
                "produkt krajowy brutto",
                "produktu krajowego brutto",
            ),
            exclude=("precyzji", "polska=100", "polska = 100", "w odsetkach"),
            prefer=(
                "rok poprzedni=100",
                "rok poprzedni = 100",
                "wartość liczbowa",
            ),
        )

    index_df, used = panel_from_selector(
        gdp_df,
        "Dynamika_PKB_100",
        selector,
        range(2000, 2026),
    )
    index_df["Wzrost_PKB"] = index_df["Dynamika_PKB_100"] - 100.0
    return index_df[["Kod", "Wojewodztwo", "Rok", "Wzrost_PKB"]], {"columns": used}


# ============================================================
# 10. RACH 3505 – UDZIAŁ PRZEMYSŁU W WDB
# ============================================================


def _best_gva_column(df, year, mode):
    candidates = []
    for col in df.columns:
        text = normalize_low(col)
        if not col_has_year(col, year):
            continue
        if "wskaźnik precyzji" in text or "precyzji" in text:
            continue
        if "polska=100" in text or "polska = 100" in text:
            continue
        if "dynamika" in text:
            continue
        if "w odsetkach" in text:
            continue
        if "%" in text and "wartość liczbowa" not in text:
            continue

        score = 0
        if "wartość liczbowa" in text:
            score += 150

        if mode == "total":
            if "wartość dodana brutto" in text:
                score += 180
            if "ogółem" in text:
                score += 120
            if "przemysł" in text:
                score -= 50
            if "rolnictwo" in text or "budownictwo" in text or "usługi" in text:
                score -= 50
        else:
            if "przemysł" in text:
                score += 300
            if "b+c+d+e" in text:
                score += 250
            if "sekcja c" in text and "+" not in text:
                score += 170
            if "budownictwo" in text:
                score -= 80
            if "rolnictwo" in text:
                score -= 80
            if "usługi" in text:
                score -= 60

        candidates.append((score, col))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def build_gva_panel(gva_df):
    if "Kod" not in gva_df.columns:
        raise ValueError("RACH 3505 nie zawiera kolumny Kod.")

    rows = []
    used = []
    for year in range(2000, 2026):
        total_col = _best_gva_column(gva_df, year, "total")
        industry_col = _best_gva_column(gva_df, year, "industry")
        if total_col is None or industry_col is None:
            continue

        keep_cols = ["Kod"]
        if "Nazwa" in gva_df.columns:
            keep_cols.append("Nazwa")
        keep_cols += [total_col, industry_col]
        part = gva_df[keep_cols].copy()
        rename_map = {total_col: "WDB_Ogolem", industry_col: "WDB_Przemysl"}
        if "Nazwa" in part.columns:
            rename_map["Nazwa"] = "Nazwa"
        part = part.rename(columns=rename_map)
        part["Rok"] = int(year)
        part["WDB_Ogolem"] = numeric_series(part["WDB_Ogolem"])
        part["WDB_Przemysl"] = numeric_series(part["WDB_Przemysl"])
        part["Udzial_Przemyslu_GVA"] = np.where(
            part["WDB_Ogolem"] > 0,
            part["WDB_Przemysl"] / part["WDB_Ogolem"],
            np.nan,
        )
        rows.append(part[["Kod", "Rok", "WDB_Ogolem", "WDB_Przemysl", "Udzial_Przemyslu_GVA"]])
        used.append((year, total_col, industry_col))

    if not rows:
        raise ValueError(
            "Nie udało się wyodrębnić WDB ogółem i przemysłu z RACH 3505. "
            f"Dostępne kolumny: {list(gva_df.columns)[:60]}"
        )

    out = pd.concat(rows, ignore_index=True)
    out = clean_key_panel(out)
    return out, {"columns": used}


# ============================================================
# 11. RACH 3510 – PRODUKTYWNOŚĆ (ZMIENNA OPCJONALNA)
# ============================================================


def build_productivity_panel(productivity_df):
    def selector(df, year):
        return choose_year_column(
            df,
            year,
            include_all=("na 1 pracującego",),
            exclude=("precyzji", "dynamika", "polska=100", "polska = 100", "w odsetkach"),
            prefer=("wartość dodana brutto", "wartość liczbowa", "ogółem"),
        )

    productivity, used = panel_from_selector(
        productivity_df,
        "Produktywnosc",
        selector,
        range(2000, 2026),
    )
    productivity = productivity[productivity["Produktywnosc"] > 0].copy()
    return productivity, {"columns": used}


# ============================================================
# 12. FINALNY PANEL – AUTOMATYCZNE WYKRYCIE WSPÓLNEGO OKNA
# ============================================================


def merge_source(base, other, name):
    before = len(base)
    value_cols = [c for c in other.columns if c not in {"Kod", "Nazwa", "Wojewodztwo", "Rok"}]
    other_small = other[["Wojewodztwo", "Rok"] + value_cols].copy()
    out = base.merge(other_small, on=["Wojewodztwo", "Rok"], how="inner")
    out["Kod"] = out["Wojewodztwo"].map({v: k for k, v in WOJEWODZTWA_MAP.items()})
    return out, {"source": name, "before": before, "after": len(out)}


@st.cache_data(show_spinner=False)
def load_and_clean_data(
    unemployment_path,
    employment_path,
    wages_path,
    gdp_path,
    gva_path,
    productivity_path=None,
):
    raw_unemployment = read_table(unemployment_path)
    raw_employment = read_table(employment_path)
    raw_wages = read_table(wages_path)
    raw_gdp = read_table(gdp_path)
    raw_gva = read_table(gva_path)

    sources = {}

    ryne, ryne_info = build_ryne_panel(raw_unemployment, raw_employment)
    wages, wages_info = build_regional_wages(raw_wages)
    gdp, gdp_info = build_gdp_growth(raw_gdp)
    gva, gva_info = build_gva_panel(raw_gva)

    sources["RYNE"] = ryne_info
    sources["WYNA"] = wages_info
    sources["RACH_3502"] = gdp_info
    sources["RACH_3505"] = gva_info

    # Produktywność jest dodatkiem. Nie może wyzerować całego panelu tylko
    # dlatego, że GUS nie podał jej dla części lat.
    productivity = None
    if productivity_path is not None and Path(productivity_path).exists():
        try:
            raw_productivity = read_table(productivity_path)
            productivity, productivity_info = build_productivity_panel(raw_productivity)
            sources["RACH_3510"] = productivity_info
        except Exception as exc:
            sources["RACH_3510"] = {"error": str(exc)}

    # Wspólne lata tylko dla zmiennych rdzeniowych.
    year_sets = [
        set(ryne["Rok"].unique()),
        set(wages["Rok"].unique()),
        set(gdp["Rok"].unique()),
        set(gva["Rok"].unique()),
    ]
    common_years = sorted(set.intersection(*year_sets))

    # Równocześnie ograniczamy treatment do lat, dla których znamy płacę minimalną.
    common_years = sorted(set(common_years) & set(MIN_WAGE_PL.keys()))

    if len(common_years) < 3:
        raise ValueError(
            "Nie znaleziono co najmniej 3 wspólnych lat dla podstawowych źródeł.\n\n"
            f"RYNE: {sorted(ryne['Rok'].unique())}\n"
            f"WYNA: {sorted(wages['Rok'].unique())}\n"
            f"RACH 3502: {sorted(gdp['Rok'].unique())}\n"
            f"RACH 3505: {sorted(gva['Rok'].unique())}"
        )

    ryne = ryne[ryne["Rok"].isin(common_years)].copy()
    wages = wages[wages["Rok"].isin(common_years)].copy()
    gdp = gdp[gdp["Rok"].isin(common_years)].copy()
    gva = gva[gva["Rok"].isin(common_years)].copy()

    df = ryne
    merge_log = []
    for name, other in [
        ("WYNA 2504", wages),
        ("RACH 3502", gdp),
        ("RACH 3505", gva),
    ]:
        df, info = merge_source(df, other, name)
        merge_log.append(info)

    if productivity is not None:
        productivity = productivity[productivity["Rok"].isin(common_years)].copy()
        prod_cols = ["Wojewodztwo", "Rok"] + [c for c in productivity.columns if c not in {"Kod", "Nazwa", "Wojewodztwo", "Rok"}]
        df = df.merge(productivity[prod_cols], on=["Wojewodztwo", "Rok"], how="left")

    df["Wojewodztwo"] = df["Kod"].map(WOJEWODZTWA_MAP)
    df["Placa_Minimalna"] = df["Rok"].map(MIN_WAGE_PL)
    df["Kaitz_Index"] = (
        df["Placa_Minimalna"] / df["Przecietne_Wynagrodzenie_Regionalne"]
    )
    df["Kaitz_Procent"] = 100.0 * df["Kaitz_Index"]

    numeric_cols = [
        "Rok", "Stopa_Zatrudnienia", "Stopa_Bezrobocia_BAEL",
        "Przecietne_Wynagrodzenie_Regionalne", "Placa_Minimalna",
        "Kaitz_Index", "Kaitz_Procent", "Wzrost_PKB",
        "WDB_Ogolem", "WDB_Przemysl", "Udzial_Przemyslu_GVA",
        "Produktywnosc",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = numeric_series(df[col])

    df = df.replace([np.inf, -np.inf], np.nan)

    # W eksporcie RYNE 4100 dla części obserwacji 0,0% jest brakiem danych.
    df.loc[df["Stopa_Bezrobocia_BAEL"] <= 0, "Stopa_Bezrobocia_BAEL"] = np.nan
    df.loc[df["Przecietne_Wynagrodzenie_Regionalne"] <= 0, "Przecietne_Wynagrodzenie_Regionalne"] = np.nan
    df.loc[df["Kaitz_Index"] <= 0, "Kaitz_Index"] = np.nan

    core_required = [
        "Wojewodztwo", "Kod", "Rok", "Kaitz_Index",
        "Stopa_Zatrudnienia", "Stopa_Bezrobocia_BAEL",
        "Wzrost_PKB", "Udzial_Przemyslu_GVA",
    ]
    df = df.dropna(subset=core_required).copy()
    df = df.sort_values(["Wojewodztwo", "Rok"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "Po połączeniu źródeł nie pozostała żadna obserwacja.\n\n"
            f"Wspólne lata rdzenia: {common_years}\n"
            f"RYNE: {len(ryne)} wierszy, kody: {sorted(ryne['Kod'].dropna().unique())}\n"
            f"WYNA: {len(wages)} wierszy, kody: {sorted(wages['Kod'].dropna().unique())}\n"
            f"RACH 3502: {len(gdp)} wierszy, kody: {sorted(gdp['Kod'].dropna().unique())}\n"
            f"RACH 3505: {len(gva)} wierszy, kody: {sorted(gva['Kod'].dropna().unique())}\n"
            f"Stan po merge: {merge_log}\n"
        )

    n_regions = df["Wojewodztwo"].nunique()
    n_years = df["Rok"].nunique()
    if n_regions < 4:
        raise ValueError(
            f"Po połączeniu pozostały tylko {n_regions} województwa. "
            "To za mało do sensownej analizy panelowej."
        )
    if n_years < 3:
        raise ValueError(
            f"Po połączeniu pozostało tylko {n_years} lata: {sorted(df['Rok'].unique())}."
        )

    diagnostics = {
        "common_years": common_years,
        "source_years": {
            "RYNE 4112/4100": sorted(ryne["Rok"].unique()),
            "WYNA 2504": sorted(wages["Rok"].unique()),
            "RACH 3502": sorted(gdp["Rok"].unique()),
            "RACH 3505": sorted(gva["Rok"].unique()),
            "RACH 3510": sorted(productivity["Rok"].unique()) if productivity is not None else [],
        },
        "merge_log": merge_log,
        "n_regions": n_regions,
        "n_years": n_years,
    }

    return df, diagnostics, sources


# ============================================================
# 13. MODELE
# ============================================================


def build_formula(include_productivity=False, fixed_effects=False):
    terms = [
        "Kaitz_Index",
        "Stopa_Bezrobocia_BAEL",
        "Wzrost_PKB",
        "Udzial_Przemyslu_GVA",
    ]
    if include_productivity:
        terms.append("Produktywnosc")
    formula = "Stopa_Zatrudnienia ~ " + " + ".join(terms)
    if fixed_effects:
        formula += " + C(Wojewodztwo) + C(Rok)"
    return formula


def select_model_frame(df):
    work = df.copy()
    # Produktywność dodajemy tylko wtedy, gdy nie powoduje dużego spadku próby.
    include_productivity = "Produktywnosc" in work.columns
    if include_productivity:
        completeness = work["Produktywnosc"].notna().mean()
        include_productivity = completeness >= 0.85
    return work, include_productivity


def run_benchmark(df):
    work, include_productivity = select_model_frame(df)
    formula_base = build_formula(include_productivity, fixed_effects=False)
    formula_fe = build_formula(include_productivity, fixed_effects=True)

    required = [
        "Stopa_Zatrudnienia", "Kaitz_Index", "Stopa_Bezrobocia_BAEL",
        "Wzrost_PKB", "Udzial_Przemyslu_GVA",
    ]
    if include_productivity:
        required.append("Produktywnosc")
    work = work.dropna(subset=required).copy()

    pooled = smf.ols(formula_base, data=work).fit(cov_type="HC1")
    twfe = smf.ols(formula_fe, data=work).fit(
        cov_type="cluster",
        cov_kwds={"groups": work["Wojewodztwo"]},
    )
    return pooled, twfe, include_productivity, work


CF_BASE_X = [
    "Stopa_Bezrobocia_BAEL",
    "Wzrost_PKB",
    "Udzial_Przemyslu_GVA",
]


def build_xgb():
    return xgb.XGBRegressor(
        n_estimators=250,
        max_depth=2,
        learning_rate=0.04,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=SEED,
        n_jobs=1,
    )


def build_nuisance_controls(df):
    controls = df[["Wojewodztwo", "Rok"]].copy()
    return pd.get_dummies(
        controls,
        columns=["Wojewodztwo", "Rok"],
        drop_first=True,
        dtype=float,
    )


@st.cache_resource(show_spinner=False)
def run_causal_forest(df):
    if CausalForestDML is None:
        raise ImportError(
            "Nie udało się zaimportować econml. "
            f"Szczegóły: {ECONML_IMPORT_ERROR}"
        )

    x_cols = list(CF_BASE_X)
    if "Produktywnosc" in df.columns and df["Produktywnosc"].notna().mean() >= 0.85:
        x_cols.append("Produktywnosc")

    required = ["Stopa_Zatrudnienia", "Kaitz_Index"] + x_cols
    work = df.dropna(subset=required).copy()

    if len(work) < 40:
        raise ValueError(
            f"Do Causal Forest pozostało {len(work)} obserwacji. Potrzeba co najmniej 40."
        )
    if work["Kaitz_Index"].nunique() < 20:
        raise ValueError(
            "Treatment Kaitz Index ma zbyt małą zmienność dla Causal Forest."
        )

    Y = work["Stopa_Zatrudnienia"].to_numpy(float)
    T = work["Kaitz_Index"].to_numpy(float)
    X = work[x_cols].astype(float)
    W = build_nuisance_controls(work)

    forest = CausalForestDML(
        model_y=build_xgb(),
        model_t=build_xgb(),
        discrete_treatment=False,
        n_estimators=800,
        min_samples_leaf=5,
        max_features="sqrt",
        inference=True,
        cv=3,
        random_state=SEED,
        n_jobs=1,
    )
    forest.fit(Y, T, X=X, W=W)

    effects = forest.effect(X)
    work["Estimated_Effect_CF"] = effects

    try:
        low, high = forest.effect_interval(X, alpha=0.05)
        work["Effect_Lower_95"] = low
        work["Effect_Upper_95"] = high
    except Exception:
        pass

    surrogate = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=1,
    )
    surrogate.fit(X, effects)

    shap_values = None
    try:
        explainer = shap.TreeExplainer(surrogate)
        shap_values = explainer.shap_values(X)
    except Exception:
        pass

    return {
        "forest": forest,
        "df": work,
        "X": X,
        "x_cols": x_cols,
        "effects": effects,
        "ate": float(np.mean(effects)),
        "median": float(np.median(effects)),
        "std": float(np.std(effects)),
        "surrogate": surrogate,
        "shap_values": shap_values,
    }


# ============================================================
# 14. WIZUALIZACJE / NAZWY
# ============================================================

DISPLAY_NAMES = {
    "Kaitz_Index": "Kaitz Index",
    "Stopa_Zatrudnienia": "Wskaźnik zatrudnienia BAEL",
    "Stopa_Bezrobocia_BAEL": "Stopa bezrobocia BAEL",
    "Wzrost_PKB": "Wzrost realnego PKB",
    "Udzial_Przemyslu_GVA": "Udział przemysłu w WDB",
    "Produktywnosc": "WDB na 1 pracującego",
    "Przecietne_Wynagrodzenie_Regionalne": "Przeciętne wynagrodzenie regionalne",
    "Placa_Minimalna": "Płaca minimalna",
}


def nice_num(value, digits=4):
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{digits}f}"


def diagnostics_table(df):
    rows = []
    cols = [
        "Stopa_Zatrudnienia", "Stopa_Bezrobocia_BAEL", "Kaitz_Index",
        "Wzrost_PKB", "Udzial_Przemyslu_GVA", "Produktywnosc",
    ]
    for col in cols:
        if col not in df.columns:
            continue
        rows.append({
            "Zmienna": DISPLAY_NAMES.get(col, col),
            "N": int(df[col].notna().sum()),
            "Braki": int(df[col].isna().sum()),
            "Średnia": float(df[col].mean()),
            "Min": float(df[col].min()),
            "Max": float(df[col].max()),
        })
    return pd.DataFrame(rows)


def shap_figure(cf_result):
    values = cf_result["shap_values"]
    if values is None:
        return None
    X = cf_result["X"].rename(columns=DISPLAY_NAMES)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(values, X, show=False, plot_size=None)
    fig = plt.gcf()
    fig.suptitle("SHAP – zmienne związane z heterogenicznością τ(x)", y=1.02)
    fig.tight_layout()
    return fig


# ============================================================
# 15. START
# ============================================================

st.title("📊 Wpływ płacy minimalnej na zatrudnienie w województwach")
st.caption(
    "Panel regionalny Polski • BAEL • Rachunki regionalne GUS • "
    "regionalny Kaitz • Two-Way FE • DML + Causal Forest"
)

paths = {key: detect_file(key) for key in REPOSITORY_FILES}

st.sidebar.header("Źródła danych")
for key, path in paths.items():
    label = CORE_SOURCE_LABELS[key]
    if path is None:
        if key == "RACH_PRODUCTIVITY":
            st.sidebar.info(f"Opcjonalne: {label}")
        else:
            st.sidebar.error(f"Brak: {label}")
    else:
        st.sidebar.success(path.name)

required_keys = [
    "RYNE_UNEMPLOYMENT",
    "RYNE_EMPLOYMENT",
    "WYNA_REGIONAL",
    "RACH_GDP_GROWTH",
    "RACH_GVA_SECTOR",
]
missing = [k for k in required_keys if paths[k] is None]
if missing:
    st.error(
        "Nie znaleziono wymaganych plików w repozytorium:\n\n"
        + "\n".join(REPOSITORY_FILES[k] for k in missing)
    )
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption("Aplikacja korzysta bezpośrednio z plików znajdujących się w repozytorium. Nie ma ręcznego uploadu.")

try:
    with st.spinner("Wczytywanie tabel GUS i budowa panelu…"):
        df, data_diag, source_diag = load_and_clean_data(
            str(paths["RYNE_UNEMPLOYMENT"]),
            str(paths["RYNE_EMPLOYMENT"]),
            str(paths["WYNA_REGIONAL"]),
            str(paths["RACH_GDP_GROWTH"]),
            str(paths["RACH_GVA_SECTOR"]),
            str(paths["RACH_PRODUCTIVITY"]) if paths["RACH_PRODUCTIVITY"] else None,
        )
except Exception as exc:
    st.error("Błąd analizy danych")
    st.exception(exc)
    st.stop()

try:
    with st.spinner("Estymacja benchmarku i Causal Forest…"):
        pooled, twfe, include_productivity, benchmark_df = run_benchmark(df)
        cf = run_causal_forest(df)
except Exception as exc:
    st.error("Błąd modelowania")
    st.exception(exc)
    st.stop()

st.sidebar.metric("Obserwacje", len(df))
st.sidebar.metric("Województwa", df["Wojewodztwo"].nunique())
st.sidebar.metric("Lata", f"{int(df['Rok'].min())}–{int(df['Rok'].max())}")
st.sidebar.metric("Średni Kaitz", f"{df['Kaitz_Index'].mean():.3f}")
if df["Wojewodztwo"].nunique() < 12 or df["Rok"].nunique() < 5:
    st.sidebar.warning("Próba jest ograniczona przez dostępność danych GUS; interpretację wyników należy traktować ostrożnie.")

# ============================================================
# 16. TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Panel danych",
    "📈 Benchmark",
    "🌲 Causal Forest",
    "🔎 SHAP",
    "🔧 Diagnostyka",
])

with tab1:
    st.subheader("Finalny panel analityczny")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Obserwacje", len(df))
    c2.metric("Województwa", df["Wojewodztwo"].nunique())
    c3.metric("Zakres", f"{int(df['Rok'].min())}–{int(df['Rok'].max())}")
    c4.metric("Średni Kaitz", f"{df['Kaitz_Index'].mean():.3f}")

    shown = [
        "Wojewodztwo", "Rok", "Kaitz_Index",
        "Przecietne_Wynagrodzenie_Regionalne",
        "Stopa_Zatrudnienia", "Stopa_Bezrobocia_BAEL",
        "Wzrost_PKB", "Udzial_Przemyslu_GVA",
    ]
    if "Produktywnosc" in df.columns:
        shown.append("Produktywnosc")

    st.dataframe(
        df[shown]
        .rename(columns=DISPLAY_NAMES)
        .sort_values(["Rok", "Wojewodztwo"]),
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["Kaitz_Index"], bins=18, kde=True, ax=ax)
        ax.set_title("Rozkład regionalnego Kaitz Index")
        ax.set_xlabel("Płaca minimalna / przeciętne wynagrodzenie regionalne")
        ax.set_ylabel("Liczba obserwacji")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with right:
        yearly = df.groupby("Rok", as_index=False)["Kaitz_Index"].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(yearly["Rok"], yearly["Kaitz_Index"], marker="o")
        ax.set_title("Średni regionalny Kaitz w czasie")
        ax.set_xlabel("Rok")
        ax.set_ylabel("Kaitz Index")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    st.info(
        "W głównej próbie zakres lat jest wyznaczany automatycznie jako część wspólna "
        "lat dostępnych w RYNE, WYNA i RACH. Dzięki temu brak obserwacji w jednym źródle "
        "nie zeruje całego panelu tylko przez z góry narzucony zakres 2010–2025."
    )

with tab2:
    st.subheader("Benchmark ekonometryczny")
    a, b = st.columns(2)
    with a:
        st.metric("Pooled OLS – Kaitz", nice_num(pooled.params.get("Kaitz_Index")))
        st.caption("Benchmark bez efektów stałych.")
    with b:
        st.metric("Two-Way FE – Kaitz", nice_num(twfe.params.get("Kaitz_Index")))
        st.caption("Efekty stałe województw i lat; SE klastrowane po województwie.")

    st.caption(
        "W benchmarku zmienna Produktywnosc jest uwzględniana tylko wtedy, gdy ma "
        "co najmniej 85% kompletnych obserwacji."
    )
    st.text(twfe.summary())

with tab3:
    st.subheader("Double Machine Learning + Causal Forest")
    a, b, c = st.columns(3)
    a.metric("Średni efekt τ(x)", nice_num(cf["ate"]))
    b.metric("Mediana τ(x)", nice_num(cf["median"]))
    c.metric("SD τ(x)", nice_num(cf["std"]))

    left, right = st.columns(2)
    with left:
        regional = cf["df"].groupby("Wojewodztwo")["Estimated_Effect_CF"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(7, 8))
        ax.barh(regional.index, regional.values)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Średni estymowany efekt τ(x)")
        ax.set_title("Heterogeniczność efektu między województwami")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with right:
        plot_df = cf["df"].copy()
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(
            plot_df["Stopa_Bezrobocia_BAEL"],
            plot_df["Estimated_Effect_CF"],
            alpha=0.75,
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Stopa bezrobocia BAEL")
        ax.set_ylabel("Estymowany efekt τ(x)")
        ax.set_title("Heterogeniczność a warunki rynku pracy")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    st.info(
        "SHAP i wykresy heterogeniczności opisują zmienność estymowanego τ(x). "
        "Nie należy interpretować wartości SHAP jako dodatkowych efektów przyczynowych."
    )

with tab4:
    st.subheader("SHAP – heterogeniczność efektu")
    fig = shap_figure(cf)
    if fig is None:
        st.warning("Nie udało się policzyć SHAP dla modelu zastępczego.")
    else:
        st.pyplot(fig, clear_figure=True)
    st.caption(
        "Model zastępczy XGBoost aproksymuje przewidywany efekt τ(x) z Causal Forest. "
        "SHAP wskazuje, które cechy są związane z różnicami w τ(x)."
    )

with tab5:
    st.subheader("Diagnostyka danych")
    st.write("**Wspólne lata rdzenia:**", data_diag["common_years"])

    st.markdown("### Zakresy lat według źródła")
    year_rows = []
    for source, years in data_diag["source_years"].items():
        year_rows.append({
            "Źródło": source,
            "Pierwszy rok": min(years) if years else None,
            "Ostatni rok": max(years) if years else None,
            "Liczba lat": len(years),
            "Lata": ", ".join(map(str, years)),
        })
    st.dataframe(pd.DataFrame(year_rows), use_container_width=True, hide_index=True)

    st.markdown("### Panel po połączeniu")
    st.write(
        f"Województwa: **{data_diag['n_regions']}**, "
        f"lata: **{data_diag['n_years']}**, "
        f"obserwacje: **{len(df)}**."
    )

    st.markdown("### Kompletność zmiennych")
    st.dataframe(diagnostics_table(df), use_container_width=True, hide_index=True)

    st.markdown("### Kontrola zmian liczby obserwacji przy scalaniu")
    st.dataframe(pd.DataFrame(data_diag["merge_log"]), use_container_width=True, hide_index=True)

    st.markdown("### Status opcjonalnej produktywności")
    prod_status = source_diag.get("RACH_3510", {})
    if "error" in prod_status:
        st.warning("RACH 3510 nie został użyty jako zmienna opcjonalna: " + str(prod_status["error"]))
    elif prod_status:
        st.success("RACH 3510 odczytany; zmienna jest używana tylko przy wysokiej kompletności.")
    else:
        st.info("Plik RACH 3510 nie został znaleziony – nie jest wymagany do działania aplikacji.")
