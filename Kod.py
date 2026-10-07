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
# 1. SERIE STAŁE I MAPOWANIE TERYT
# ============================================================

MIN_WAGE_PL = {
    2010: 1317, 2011: 1386, 2012: 1500, 2013: 1600, 2014: 1680,
    2015: 1750, 2016: 1850, 2017: 2000, 2018: 2100, 2019: 2250,
    2020: 2600, 2021: 2800, 2022: 3010, 2023: 3490, 2024: 4242,
    2025: 4666,
}

# Zachowana jako seria referencyjna/awaryjna. Nie jest używana do regionalnego Kaitza.
PRZECIETNE_WYNAGRODZENIE_PL = {
    1999: 1706.74, 2000: 1923.81, 2001: 2061.85, 2002: 2133.21,
    2003: 2201.47, 2004: 2289.57, 2005: 2380.29, 2006: 2477.23,
    2007: 2691.03, 2008: 2943.88, 2009: 3102.96, 2010: 3224.98,
    2011: 3399.52, 2012: 3521.67, 2013: 3650.06, 2014: 3783.46,
    2015: 3899.78, 2016: 4047.21, 2017: 4271.51, 2018: 4585.03,
    2019: 4918.17, 2020: 5167.47, 2021: 5662.53, 2022: 6346.15,
    2023: 7155.48, 2024: 8181.72, 2025: 8903.56,
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

# ============================================================
# 2. PLIKI – WSZYSTKO Z REPO, ZERO UPLOADU
# ============================================================

REPOSITORY_FILES = {
    "RYNE_UNEMPLOYMENT": "RYNE_4100_CTAB_20261006234212.csv",
    "RYNE_EMPLOYMENT": "RYNE_4112_CTAB_20261006234336.csv",
    "WYNA_REGIONAL": "WYNA_2504_XTAB_20261007010213.xlsx",
    "RACH_GDP_GROWTH": "RACH_3502_XTAB_20261007010836.xlsx",
    "RACH_GVA_SECTOR": "RACH_3505_XTAB_20261007011009.xlsx",
    "RACH_PRODUCTIVITY": "RACH_3510_XTAB_20261007010935.xlsx",
}


def detect_file(key):
    expected = REPOSITORY_FILES[key]
    exact = APP_DIR / expected
    if exact.exists():
        return exact

    prefix = expected.split("_")[0].upper()
    ident = expected.split("_")[1].upper() if "_" in expected else ""

    try:
        candidates = sorted(
            p for p in APP_DIR.iterdir()
            if p.is_file() and p.name.upper().startswith(prefix)
        )
    except Exception:
        candidates = []

    for p in candidates:
        if ident and f"_{ident}_" in p.name.upper():
            return p
    return candidates[-1] if candidates else None


# ============================================================
# 3. NARZĘDZIA TEKSTOWE / TERYT / LICZBY
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


def looks_like_teryt(value):
    if pd.isna(value):
        return False
    digits = re.sub(r"\D", "", normalize_text(value))
    return len(digits) >= 7 and digits[:2] in WOJEWODZTWA_MAP


def extract_teryt(value):
    digits = re.sub(r"\D", "", normalize_text(value))
    if not digits:
        return np.nan
    digits = digits.zfill(7)
    prefix = digits[:2]
    return prefix if prefix in WOJEWODZTWA_MAP else np.nan


def year_from_text(value):
    text = normalize_text(value)
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)
    return int(match.group(1)) if match else None


def col_has_year(col, year):
    return year_from_text(col) == int(year) or bool(
        re.search(rf"(?<!\d){int(year)}(?!\d)", normalize_text(col))
    )


# ============================================================
# 4. ROBUSTNY ODCZYT GUS XLSX
# ============================================================


def find_code_name_row(raw, scan_limit=100):
    limit = min(len(raw), scan_limit)
    for i in range(limit):
        row = [normalize_low(v) for v in raw.iloc[i].tolist()]
        has_kod = any(v == "kod" or v.startswith("kod ") for v in row)
        has_nazwa = any(v == "nazwa" or v.startswith("nazwa ") for v in row)
        if has_kod and has_nazwa:
            return i
    return None


def find_data_start(raw, code_name_row, scan_limit=40):
    start = code_name_row + 1
    stop = min(len(raw), start + scan_limit)
    for i in range(start, stop):
        first = raw.iloc[i, 0] if raw.shape[1] else None
        second = raw.iloc[i, 1] if raw.shape[1] > 1 else None
        if looks_like_teryt(first) and normalize_text(second):
            return i
    return start


def _fill_merged_headers(header):
    h = header.copy()
    h = h.map(normalize_text)
    h = h.replace("", np.nan)
    # Najpierw poziomo – obsługa scalonych komórek grupy.
    h = h.ffill(axis=1)
    # Potem pionowo – obsługa opisów trwających przez kilka wierszy.
    h = h.ffill(axis=0)
    return h


def detect_year_row(raw, start_row, end_row):
    """Znajduje w obszarze nagłówka wiersz, w którym zapisano lata."""
    best = None
    for i in range(max(0, start_row), min(end_row, len(raw))):
        years = []
        for j in range(raw.shape[1]):
            y = year_from_text(raw.iat[i, j])
            if y is not None and 1900 <= y <= 2100:
                years.append((j, y))
        distinct = len(set(y for _, y in years))
        if len(years) >= 3 and distinct >= 3:
            score = len(years) + 2 * distinct
            if best is None or score > best[0]:
                best = (score, i, years)
    return best[1:] if best else (None, [])


def add_years_from_row(year_row_values, ncols):
    mapping = [None] * ncols
    current = None
    for j in range(ncols):
        val = year_row_values[j] if j < len(year_row_values) else None
        y = year_from_text(val)
        if y is not None:
            current = y
        mapping[j] = current
    return mapping


def build_flat_gus_columns(raw, code_name_row, data_start):
    """Spłaszcza nietypowe, wielowierszowe nagłówki GUS.

    Kluczowa poprawka względem poprzedniej wersji: lata mogą znajdować się
    w osobnym wierszu lub nawet przed wierszem Kod/Nazwa. Są wykrywane
    niezależnie i dopisywane do każdej kolumny.
    """
    header_start = max(0, code_name_row - 12)
    header_end = data_start
    header = raw.iloc[header_start:header_end].copy()

    if header.empty:
        return [f"col_{j}" for j in range(raw.shape[1])]

    # Wersja wypełniona do rozpoznawania opisów grup.
    filled = _fill_merged_headers(header)

    # Szukamy osobnego wiersza z latami w całym obszarze nagłówka.
    year_row_idx, year_pairs = detect_year_row(raw, header_start, header_end)
    year_map = [None] * raw.shape[1]
    if year_row_idx is not None:
        raw_year_row = [raw.iat[year_row_idx, j] for j in range(raw.shape[1])]
        year_map = add_years_from_row(raw_year_row, raw.shape[1])

    columns = []
    used = set()

    for j in range(raw.shape[1]):
        parts = []

        # Wszystkie sensowne wartości z nagłówka dla kolumny.
        for i in range(filled.shape[0]):
            value = filled.iat[i, j]
            if pd.isna(value):
                continue
            value = normalize_text(value)
            if not value:
                continue
            if value in {"Kod", "Nazwa"}:
                continue
            if value not in parts:
                parts.append(value)

        # Jeżeli osobny wiersz roku nie trafił do joinu, dodajemy go jawnie.
        if year_map[j] is not None:
            y_text = str(year_map[j])
            if y_text not in parts:
                parts.append(y_text)

        if any(normalize_low(p) == "kod" for p in [raw.iat[code_name_row, j]]):
            name = "Kod"
        elif any(normalize_low(p) == "nazwa" for p in [raw.iat[code_name_row, j]]):
            name = "Nazwa"
        else:
            name = " | ".join(parts).strip()
            if not name:
                name = f"col_{j}"

        # Kod/Nazwa mogą zostać rozpoznane jako kolumny bezpośrednio z wiersza.
        direct = normalize_low(raw.iat[code_name_row, j])
        if direct == "kod" or direct.startswith("kod "):
            name = "Kod"
        elif direct == "nazwa" or direct.startswith("nazwa "):
            name = "Nazwa"

        base_name = name
        k = 1
        while name in used:
            k += 1
            name = f"{base_name}__{k}"
        used.add(name)
        columns.append(name)

    return columns


def read_excel_gus(path):
    xls = pd.ExcelFile(path, engine="openpyxl")
    candidates = []

    for sheet in xls.sheet_names:
        raw = pd.read_excel(
            path,
            sheet_name=sheet,
            header=None,
            engine="openpyxl",
        )
        if raw.empty:
            continue

        code_name_row = find_code_name_row(raw)
        if code_name_row is None:
            continue

        data_start = find_data_start(raw, code_name_row)
        columns = build_flat_gus_columns(raw, code_name_row, data_start)

        data = raw.iloc[data_start:].copy()
        data.columns = columns[:data.shape[1]]
        data = data.dropna(how="all")
        if data.empty:
            continue

        # Usuwamy całkowicie puste kolumny.
        data = data.loc[:, ~data.isna().all(axis=0)].copy()
        data.columns = [normalize_text(c) for c in data.columns]

        score = 0
        if "Kod" in data.columns:
            score += 10000
        if "Nazwa" in data.columns:
            score += 1000
        score += 10 * sum(year_from_text(c) is not None for c in data.columns)
        score += min(data.shape[1], 100)

        candidates.append((score, sheet, data))

    if not candidates:
        raise ValueError(
            f"Nie udało się odczytać tabeli GUS z pliku '{Path(path).name}'. "
            "Nie znaleziono nagłówka z Kod/Nazwa."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][2]


# ============================================================
# 5. ROBUSTNY ODCZYT GUS CSV
# ============================================================


def read_csv_gus(path):
    best = None
    errors = []

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

                score = 0
                lows = [normalize_low(c) for c in df.columns]
                if "kod" in lows:
                    score += 10000
                if "nazwa" in lows:
                    score += 1000
                score += 10 * sum(year_from_text(c) is not None for c in df.columns)
                score += min(df.shape[1], 100)

                if best is None or score > best[0]:
                    best = (score, df)
            except Exception as exc:
                errors.append(str(exc))

    if best is None:
        raise ValueError(
            f"Nie udało się odczytać CSV '{Path(path).name}'. "
            f"Ostatni błąd: {errors[-1] if errors else 'nieznany błąd'}"
        )
    return best[1]


def read_table(path):
    suffix = Path(path).suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return read_excel_gus(path)
    if suffix == ".csv":
        return read_csv_gus(path)
    raise ValueError(f"Nieobsługiwany format pliku: {suffix}")


# ============================================================
# 6. WYBÓR KOLUMN ROCZNYCH
# ============================================================


def column_matches(col, required=None, any_of=None, exclude=None):
    text = normalize_low(col)
    required = required or []
    any_of = any_of or []
    exclude = exclude or []

    if any(req.lower() not in text for req in required):
        return False
    if any_of and not any(item.lower() in text for item in any_of):
        return False
    if any(item.lower() in text for item in exclude):
        return False
    return True


def find_matching_year_column(
    df,
    year,
    required=None,
    any_of=None,
    exclude=None,
    allow_fallback=False,
):
    candidates = [
        c for c in df.columns
        if col_has_year(c, year)
        and column_matches(c, required=required, any_of=any_of, exclude=exclude)
    ]

    if not candidates and allow_fallback:
        candidates = [c for c in df.columns if col_has_year(c, year)]

    if not candidates:
        return None

    def score(col):
        text = normalize_low(col)
        s = 0
        if "wartość liczbowa" in text:
            s += 100
        if "ogółem" in text:
            s += 40
        if "wartość" in text:
            s += 20
        if "wskaźnik precyzji" in text or "precyzji" in text:
            s -= 1000
        if "polska = 100" in text or "polska=100" in text:
            s -= 800
        return s

    return sorted(candidates, key=score, reverse=True)[0]


def wide_to_long(df, value_name, selector, years):
    if "Kod" not in df.columns:
        raise ValueError(
            f"Nie znaleziono wymaganej kolumny 'Kod'. Dostępne kolumny: {list(df.columns)}"
        )

    rows = []
    for year in years:
        col = selector(df, int(year))
        if col is None:
            continue
        part = df[["Kod", col]].copy()
        part.columns = ["Kod", value_name]
        part["Rok"] = int(year)
        part[value_name] = numeric_series(part[value_name])
        rows.append(part)

    if not rows:
        raise ValueError(
            f"Nie znaleziono danych dla zmiennej '{value_name}'. "
            f"Dostępne kolumny: {list(df.columns)}"
        )

    out = pd.concat(rows, ignore_index=True)
    out["Kod"] = out["Kod"].map(extract_teryt)
    out = out.dropna(subset=["Kod"])
    out = out.drop_duplicates(["Kod", "Rok"])
    return out


# ============================================================
# 7. RYNE – BAEL
# ============================================================


def build_ryne_panel(unemployment_df, employment_df):
    def unemployment_selector(df, year):
        return find_matching_year_column(
            df,
            year,
            required=["ogółem", "wartość liczbowa"],
            exclude=["wskaźnik precyzji", "precyzji", "polska = 100", "polska=100"],
            allow_fallback=False,
        )

    def employment_selector(df, year):
        return find_matching_year_column(
            df,
            year,
            required=["ogółem", "wartość liczbowa"],
            exclude=["wskaźnik precyzji", "precyzji", "polska = 100", "polska=100"],
            allow_fallback=False,
        )

    unemployment = wide_to_long(
        unemployment_df,
        "Stopa_Bezrobocia_BAEL",
        unemployment_selector,
        range(2010, 2026),
    )

    employment = wide_to_long(
        employment_df,
        "Stopa_Zatrudnienia",
        employment_selector,
        range(2010, 2026),
    )

    return employment.merge(
        unemployment,
        on=["Kod", "Rok"],
        how="inner",
    )


# ============================================================
# 8. WYNAGRODZENIA – REGIONALNY KAITZ
# ============================================================


def build_regional_wages(wyna_df):
    def wage_selector(df, year):
        candidates = [
            c for c in df.columns
            if col_has_year(c, year)
            and re.search(r"\brok\b", normalize_low(c))
            and "wartość liczbowa" in normalize_low(c)
            and "wskaźnik precyzji" not in normalize_low(c)
        ]
        if not candidates:
            candidates = [
                c for c in df.columns
                if col_has_year(c, year)
                and re.search(r"\brok\b", normalize_low(c))
                and "wskaźnik precyzji" not in normalize_low(c)
            ]
        return candidates[0] if candidates else None

    return wide_to_long(
        wyna_df,
        "Przecietne_Wynagrodzenie_Regionalne",
        wage_selector,
        range(2010, 2026),
    )


# ============================================================
# 9. RACH – PKB NOMINALNY + DYNAMIKA REALNEGO PKB
# ============================================================


def build_gdp_panel(gdp_growth_df):
    def growth_selector(df, year):
        # RACH 3502 zawiera dynamikę produktu krajowego brutto
        # w cenach stałych, rok poprzedni=100.
        return find_matching_year_column(
            df,
            year,
            required=["dynamika produktu krajowego brutto ogółem"],
            any_of=["rok poprzedni=100", "rok poprzedni = 100"],
            exclude=["precyzji"],
            allow_fallback=False,
        )

    growth_index = wide_to_long(
        gdp_growth_df,
        "Dynamika_PKB_100",
        growth_selector,
        range(2010, 2025),
    )
    growth_index["Wzrost_PKB"] = growth_index["Dynamika_PKB_100"] - 100.0
    return growth_index[["Kod", "Rok", "Wzrost_PKB"]]


# ============================================================
# 10. RACH 3505 – UDZIAŁ PRZEMYSŁU W WDB
# ============================================================


def _section_exact(col, section):
    text = normalize_low(col)
    section_low = section.lower()
    # Odróżnia „Sekcja B” od „Sekcja B+C+D+E”.
    if section_low == "sekcja b":
        return bool(re.search(r"sekcja b(?!\+)", text))
    if section_low == "sekcja c":
        return bool(re.search(r"sekcja c(?!\+)", text))
    if section_low == "sekcja d":
        return bool(re.search(r"sekcja d(?!\+)", text))
    if section_low == "sekcja e":
        return bool(re.search(r"sekcja e(?!\+)", text))
    return section_low in text


def build_gva_panel(gva_df):
    if "Kod" not in gva_df.columns:
        raise ValueError(
            f"RACH 3505 nie zawiera kolumny 'Kod'. Dostępne: {list(gva_df.columns)}"
        )

    def select_total(df, year):
        candidates = [
            c for c in df.columns
            if col_has_year(c, year)
            and "ogółem" in normalize_low(c)
            and "wartość liczbowa" in normalize_low(c)
            and "wskaźnik precyzji" not in normalize_low(c)
            and "%" not in normalize_text(c)
        ]
        if not candidates:
            candidates = [
                c for c in df.columns
                if col_has_year(c, year)
                and "ogółem" in normalize_low(c)
                and "wskaźnik precyzji" not in normalize_low(c)
            ]
        return candidates[0] if candidates else None

    def select_section(df, year, section):
        candidates = [
            c for c in df.columns
            if col_has_year(c, year)
            and _section_exact(c, section)
            and "wartość liczbowa" in normalize_low(c)
            and "wskaźnik precyzji" not in normalize_low(c)
            and "%" not in normalize_text(c)
        ]
        if not candidates:
            candidates = [
                c for c in df.columns
                if col_has_year(c, year)
                and _section_exact(c, section)
                and "wskaźnik precyzji" not in normalize_low(c)
                and "%" not in normalize_text(c)
                and "polska=100" not in normalize_low(c)
                and "polska = 100" not in normalize_low(c)
            ]
        return candidates[0] if candidates else None

    rows = []
    for year in range(2010, 2025):
        total_col = select_total(gva_df, year)
        if total_col is None:
            continue

        section_cols = {
            sec: select_section(gva_df, year, sec)
            for sec in ["Sekcja B", "Sekcja C", "Sekcja D", "Sekcja E"]
        }

        if all(section_cols.values()):
            part = gva_df[[
                "Kod",
                total_col,
                section_cols["Sekcja B"],
                section_cols["Sekcja C"],
                section_cols["Sekcja D"],
                section_cols["Sekcja E"],
            ]].copy()
            part.columns = [
                "Kod", "WDB_Ogolem", "WDB_B", "WDB_C", "WDB_D", "WDB_E"
            ]
            for c in ["WDB_Ogolem", "WDB_B", "WDB_C", "WDB_D", "WDB_E"]:
                part[c] = numeric_series(part[c])
            part["WDB_Przemysl"] = part[["WDB_B", "WDB_C", "WDB_D", "WDB_E"]].sum(axis=1, min_count=4)
        else:
            # Awaryjnie: tabela może zawierać gotową agregację B+C+D+E.
            industry = [
                c for c in gva_df.columns
                if col_has_year(c, year)
                and "b+c+d+e" in normalize_low(c)
                and "wartość liczbowa" in normalize_low(c)
                and "wskaźnik precyzji" not in normalize_low(c)
            ]
            if not industry:
                continue
            part = gva_df[["Kod", total_col, industry[0]]].copy()
            part.columns = ["Kod", "WDB_Ogolem", "WDB_Przemysl"]
            part["WDB_Ogolem"] = numeric_series(part["WDB_Ogolem"])
            part["WDB_Przemysl"] = numeric_series(part["WDB_Przemysl"])

        part["Rok"] = year
        rows.append(part[["Kod", "WDB_Ogolem", "WDB_Przemysl", "Rok"]])

    if not rows:
        raise ValueError(
            "Nie udało się zbudować panelu RACH 3505. "
            f"Dostępne kolumny: {list(gva_df.columns)}"
        )

    out = pd.concat(rows, ignore_index=True)
    out["Kod"] = out["Kod"].map(extract_teryt)
    out["Udzial_Przemyslu_GVA"] = np.where(
        out["WDB_Ogolem"] > 0,
        out["WDB_Przemysl"] / out["WDB_Ogolem"],
        np.nan,
    )
    return out


# ============================================================
# 11. RACH 3510 – PRODUKTYWNOŚĆ
# ============================================================


def build_productivity_panel(productivity_df):
    def productivity_selector(df, year):
        candidates = [
            c for c in df.columns
            if col_has_year(c, year)
            and "wartość dodana brutto na 1 pracującego" in normalize_low(c)
            and "dynamika" not in normalize_low(c)
            and "polska=100" not in normalize_low(c)
            and "polska = 100" not in normalize_low(c)
            and "wskaźnik precyzji" not in normalize_low(c)
        ]
        if not candidates:
            candidates = [
                c for c in df.columns
                if col_has_year(c, year)
                and "na 1 pracującego" in normalize_low(c)
                and "dynamika" not in normalize_low(c)
                and "wskaźnik precyzji" not in normalize_low(c)
            ]
        return candidates[0] if candidates else None

    return wide_to_long(
        productivity_df,
        "Produktywnosc",
        productivity_selector,
        range(2010, 2025),
    )


# ============================================================
# 12. BUDOWA FINALNEGO PANELU
# ============================================================

@st.cache_data(show_spinner=False)
def load_and_clean_data(
    ryne_unemployment_path,
    ryne_employment_path,
    wyna_regional_path,
    rach_growth_path,
    rach_gva_path,
    rach_productivity_path,
):
    sources = {
        "RYNE 4100": ryne_unemployment_path,
        "RYNE 4112": ryne_employment_path,
        "WYNA 2504": wyna_regional_path,
        "RACH 3502": rach_growth_path,
        "RACH 3505": rach_gva_path,
        "RACH 3510": rach_productivity_path,
    }

    loaded = {}
    for label, path in sources.items():
        try:
            loaded[label] = read_table(path)
        except Exception as exc:
            raise ValueError(f"Błąd odczytu {label} ({Path(path).name}): {exc}") from exc

    ryne = build_ryne_panel(loaded["RYNE 4100"], loaded["RYNE 4112"])
    wages = build_regional_wages(loaded["WYNA 2504"])
    gdp = build_gdp_panel(loaded["RACH 3502"])
    gva = build_gva_panel(loaded["RACH 3505"])
    productivity = build_productivity_panel(loaded["RACH 3510"])

    df = ryne.merge(wages, on=["Kod", "Rok"], how="inner")
    df = df.merge(gdp, on=["Kod", "Rok"], how="inner")
    df = df.merge(gva, on=["Kod", "Rok"], how="inner")
    df = df.merge(productivity, on=["Kod", "Rok"], how="inner")

    df["Wojewodztwo"] = df["Kod"].map(WOJEWODZTWA_MAP)
    df["Placa_Minimalna"] = df["Rok"].map(MIN_WAGE_PL)

    # Regionalny Kaitz – treatment ma zmienność między regionami i w czasie.
    df["Kaitz_Index"] = (
        df["Placa_Minimalna"] /
        df["Przecietne_Wynagrodzenie_Regionalne"]
    )
    df["Kaitz_Procent"] = 100 * df["Kaitz_Index"]

    numeric_cols = [
        "Rok",
        "Stopa_Zatrudnienia",
        "Stopa_Bezrobocia_BAEL",
        "Przecietne_Wynagrodzenie_Regionalne",
        "Placa_Minimalna",
        "Kaitz_Index",
        "Kaitz_Procent",
        "Wzrost_PKB",
        "WDB_Ogolem",
        "WDB_Przemysl",
        "Udzial_Przemyslu_GVA",
        "Produktywnosc",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = numeric_series(df[col])

    df = df.replace([np.inf, -np.inf], np.nan)

    # W regionalnej stopie bezrobocia BAEL wartości 0,0% są w tym
    # eksporcie oznaczeniem braków danych dla części obserwacji, a nie
    # realistyczną stopą bezrobocia. Traktujemy je więc jako brak.
    df.loc[df["Stopa_Bezrobocia_BAEL"] <= 0, "Stopa_Bezrobocia_BAEL"] = np.nan

    # Podstawowa kontrola zakresów przed estymacją.
    df.loc[~df["Przecietne_Wynagrodzenie_Regionalne"].gt(0), "Przecietne_Wynagrodzenie_Regionalne"] = np.nan
    df.loc[~df["Kaitz_Index"].gt(0), "Kaitz_Index"] = np.nan

    required = [
        "Wojewodztwo",
        "Kod",
        "Rok",
        "Kaitz_Index",
        "Stopa_Zatrudnienia",
        "Stopa_Bezrobocia_BAEL",
        "Wzrost_PKB",
        "Udzial_Przemyslu_GVA",
        "Produktywnosc",
    ]
    df = df.dropna(subset=required).copy()
    df = df.sort_values(["Wojewodztwo", "Rok"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "Po połączeniu tabel nie pozostały kompletne obserwacje panelowe. "
            "Zakresy lat lub selekcja kolumn GUS nie pokrywają się."
        )

    n_regions = df["Wojewodztwo"].nunique()
    n_years = df["Rok"].nunique()
    if n_regions < 4:
        raise ValueError(
            f"Po czyszczeniu znaleziono tylko {n_regions} województwa. "
            "Sprawdź Kod/TERYT."
        )
    if n_years < 3:
        raise ValueError(
            f"Po czyszczeniu znaleziono tylko {n_years} lata."
        )

    return df


# ============================================================
# 13. BENCHMARK – POOLED OLS + TWO-WAY FE
# ============================================================


def run_benchmark(df):
    base = (
        "Stopa_Zatrudnienia ~ Kaitz_Index + "
        "Stopa_Bezrobocia_BAEL + Wzrost_PKB + "
        "Udzial_Przemyslu_GVA + Produktywnosc"
    )

    pooled = smf.ols(base, data=df).fit(cov_type="HC1")
    twfe = smf.ols(
        base + " + C(Wojewodztwo) + C(Rok)",
        data=df,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["Wojewodztwo"]},
    )
    return pooled, twfe


# ============================================================
# 14. DML + CAUSAL FOREST
# ============================================================

CF_X_COLS = [
    "Stopa_Bezrobocia_BAEL",
    "Wzrost_PKB",
    "Udzial_Przemyslu_GVA",
    "Produktywnosc",
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

    x_cols = [c for c in CF_X_COLS if c in df.columns]
    frame = df[["Stopa_Zatrudnienia", "Kaitz_Index"] + x_cols].copy()
    valid = frame.notna().all(axis=1)
    work = df.loc[valid].copy()

    if len(work) < 40:
        raise ValueError(
            f"Po czyszczeniu pozostało tylko {len(work)} obserwacji. "
            "To za mało dla stabilnej estymacji Causal Forest."
        )

    Y = work["Stopa_Zatrudnienia"].to_numpy(dtype=float)
    T = work["Kaitz_Index"].to_numpy(dtype=float)
    X = work[x_cols].astype(float)
    W = build_nuisance_controls(work)

    forest = CausalForestDML(
        model_y=build_xgb(),
        model_t=build_xgb(),
        discrete_treatment=False,
        n_estimators=1000,
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
        lower, upper = forest.effect_interval(X, alpha=0.05)
        work["Effect_Lower_95"] = lower
        work["Effect_Upper_95"] = upper
    except Exception:
        pass

    surrogate = None
    shap_values = None
    try:
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
# 15. WIZUALIZACJE / DIAGNOSTYKA
# ============================================================

DISPLAY_NAMES = {
    "Kaitz_Index": "Kaitz Index",
    "Stopa_Zatrudnienia": "Wskaźnik zatrudnienia BAEL",
    "Stopa_Bezrobocia_BAEL": "Stopa bezrobocia BAEL",
    "Wzrost_PKB": "Wzrost realnego PKB",
    "Udzial_Przemyslu_GVA": "Udział przemysłu w WDB",
    "Produktywnosc": "WDB na 1 pracującego",
    "Przecietne_Wynagrodzenie_Regionalne": "Regionalne przeciętne wynagrodzenie",
    "Placa_Minimalna": "Płaca minimalna",
}


def nice_num(value, digits=4):
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{digits}f}"


def plot_shap(cf_result):
    values = cf_result["shap_values"]
    if values is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "Nie udało się obliczyć SHAP dla modelu zastępczego.",
            ha="center",
            va="center",
        )
        ax.axis("off")
        return fig

    X = cf_result["X"].rename(columns=DISPLAY_NAMES)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(values, X, show=False, plot_size=None)
    ax.set_title("SHAP – zmienne związane z heterogenicznością τ(x)")
    fig.tight_layout()
    return fig


def diagnostics(df):
    rows = []
    for col in [
        "Stopa_Zatrudnienia",
        "Stopa_Bezrobocia_BAEL",
        "Kaitz_Index",
        "Wzrost_PKB",
        "Udzial_Przemyslu_GVA",
        "Produktywnosc",
    ]:
        rows.append({
            "Zmienna": DISPLAY_NAMES.get(col, col),
            "N": int(df[col].notna().sum()),
            "Średnia": float(df[col].mean()),
            "Min": float(df[col].min()),
            "Max": float(df[col].max()),
        })
    return pd.DataFrame(rows)


# ============================================================
# 16. START APLIKACJI
# ============================================================

st.title("📊 Wpływ płacy minimalnej na zatrudnienie w województwach")
st.caption(
    "Panel regionalny Polski • BAEL • Rachunki regionalne GUS • "
    "regionalny Kaitz • Pooled OLS / Two-Way FE / Causal Forest + DML"
)

paths = {key: detect_file(key) for key in REPOSITORY_FILES}

st.sidebar.header("Źródła danych")
for key, path in paths.items():
    if path is None:
        st.sidebar.error(f"Brak: {REPOSITORY_FILES[key]}")
    else:
        st.sidebar.success(path.name)

required_keys = list(REPOSITORY_FILES.keys())
missing = [k for k in required_keys if paths[k] is None]
if missing:
    st.error(
        "Nie znaleziono wymaganych plików w repozytorium: "
        + ", ".join(REPOSITORY_FILES[k] for k in missing)
    )
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dane są ładowane automatycznie z repozytorium. "
    "Aplikacja nie wymaga ręcznego wgrywania plików."
)

try:
    with st.spinner("Wczytywanie i łączenie tabel GUS…"):
        df = load_and_clean_data(
            str(paths["RYNE_UNEMPLOYMENT"]),
            str(paths["RYNE_EMPLOYMENT"]),
            str(paths["WYNA_REGIONAL"]),
            str(paths["RACH_GDP_GROWTH"]),
            str(paths["RACH_GVA_SECTOR"]),
            str(paths["RACH_PRODUCTIVITY"]),
        )
except Exception as exc:
    st.error(f"Błąd analizy danych: {exc}")
    st.exception(exc)
    st.stop()

# Szybka kontrola zmienności treatmentu.
year_treat_variation = df.groupby("Rok")["Kaitz_Index"].nunique()
if (year_treat_variation <= 1).mean() > 0.5:
    st.warning(
        "W ponad połowie lat Kaitz ma zbyt małą zmienność między województwami. "
        "Sprawdź regionalne wynagrodzenia WYNA 2504."
    )

try:
    with st.spinner("Estymacja OLS, Two-Way FE i Causal Forest + DML…"):
        pooled, twfe = run_benchmark(df)
        cf = run_causal_forest(df)
except Exception as exc:
    st.error(f"Błąd modelowania: {exc}")
    st.exception(exc)
    st.stop()

st.sidebar.markdown("---")
st.sidebar.metric("Obserwacje", len(df))
st.sidebar.metric("Województwa", df["Wojewodztwo"].nunique())
st.sidebar.metric("Zakres lat", f"{int(df['Rok'].min())}–{int(df['Rok'].max())}")

# ============================================================
# 17. TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Panel danych",
    "📈 Benchmark",
    "🌲 Causal Forest",
    "🔎 SHAP",
    "ℹ️ Metodologia",
])

with tab1:
    st.subheader("Finalny panel analityczny")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Obserwacje", len(df))
    c2.metric("Województwa", df["Wojewodztwo"].nunique())
    c3.metric("Lata", f"{int(df['Rok'].min())}–{int(df['Rok'].max())}")
    c4.metric("Średni Kaitz", f"{df['Kaitz_Index'].mean():.3f}")

    shown_cols = [
        "Wojewodztwo",
        "Rok",
        "Kaitz_Index",
        "Przecietne_Wynagrodzenie_Regionalne",
        "Stopa_Zatrudnienia",
        "Stopa_Bezrobocia_BAEL",
        "Wzrost_PKB",
        "Udzial_Przemyslu_GVA",
        "Produktywnosc",
    ]
    st.dataframe(
        df[shown_cols]
        .rename(columns=DISPLAY_NAMES)
        .sort_values(["Rok", "Wojewodztwo"]),
        use_container_width=True,
        hide_index=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["Kaitz_Index"], bins=18, kde=True, ax=ax)
        ax.set_title("Rozkład regionalnego Kaitz Index")
        ax.set_xlabel("Płaca minimalna / przeciętne wynagrodzenie regionalne")
        ax.set_ylabel("Liczba obserwacji")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with col2:
        yearly = df.groupby("Rok", as_index=False)["Kaitz_Index"].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(yearly["Rok"], yearly["Kaitz_Index"], marker="o")
        ax.set_title("Średni regionalny Kaitz w czasie")
        ax.set_xlabel("Rok")
        ax.set_ylabel("Kaitz Index")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    st.markdown("### Kontrola jakości")
    st.dataframe(diagnostics(df), use_container_width=True, hide_index=True)

    st.info(
        "Kaitz jest regionalny: krajowa płaca minimalna jest dzielona przez "
        "przeciętne wynagrodzenie brutto w danym województwie. Dzięki temu "
        "treatment ma zmienność między regionami i w czasie."
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

    st.markdown("### Wyniki Two-Way FE")
    st.text(twfe.summary())

with tab3:
    st.subheader("Double Machine Learning + Causal Forest")
    a, b, c = st.columns(3)
    a.metric("Średni efekt τ(x)", nice_num(cf["ate"]))
    b.metric("Mediana τ(x)", nice_num(cf["median"]))
    c.metric("Odchylenie τ(x)", nice_num(cf["std"]))

    col1, col2 = st.columns(2)
    with col1:
        regional = (
            cf["df"].groupby("Wojewodztwo")["Estimated_Effect_CF"]
            .mean()
            .sort_values()
        )
        fig, ax = plt.subplots(figsize=(7, 8))
        sns.barplot(x=regional.values, y=regional.index, ax=ax)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Średni estymowany efekt τ(x)")
        ax.set_ylabel("")
        ax.set_title("Heterogeniczność efektu między województwami")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with col2:
        plot_df = cf["df"].copy()
        fig, ax = plt.subplots(figsize=(7, 6))
        scatter = ax.scatter(
            plot_df["Stopa_Bezrobocia_BAEL"],
            plot_df["Estimated_Effect_CF"],
            c=plot_df["Kaitz_Index"],
            alpha=0.75,
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Stopa bezrobocia BAEL")
        ax.set_ylabel("Estymowany efekt τ(x)")
        ax.set_title("Heterogeniczność a warunki rynku pracy")
        cb = fig.colorbar(scatter, ax=ax)
        cb.set_label("Kaitz Index")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    result_cols = [
        "Wojewodztwo",
        "Rok",
        "Kaitz_Index",
        "Stopa_Bezrobocia_BAEL",
        "Wzrost_PKB",
        "Udzial_Przemyslu_GVA",
        "Produktywnosc",
        "Estimated_Effect_CF",
    ]
    if "Effect_Lower_95" in cf["df"].columns:
        result_cols += ["Effect_Lower_95", "Effect_Upper_95"]
    st.dataframe(
        cf["df"][result_cols].sort_values("Estimated_Effect_CF"),
        use_container_width=True,
        hide_index=True,
    )

with tab4:
    st.subheader("SHAP – interpretacja heterogeniczności")
    st.markdown(
        "SHAP pokazuje, które cechy są związane z różnicami w estymowanym "
        "efekcie τ(x). Nie należy go interpretować jako dodatkowego efektu przyczynowego."
    )
    fig = plot_shap(cf)
    st.pyplot(fig, clear_figure=True)

    if cf["shap_values"] is not None:
        values = np.asarray(cf["shap_values"])
        importance = pd.DataFrame({
            "Zmienna": cf["x_cols"],
            "Średnia |SHAP|": np.mean(np.abs(values), axis=0),
        }).sort_values("Średnia |SHAP|", ascending=False)
        importance["Zmienna"] = importance["Zmienna"].map(
            lambda x: DISPLAY_NAMES.get(x, x)
        )
        st.dataframe(importance, use_container_width=True, hide_index=True)

with tab5:
    st.subheader("Metodologia")
    st.markdown(
        """
**Treatment:** regionalny Kaitz Index = płaca minimalna / przeciętne regionalne wynagrodzenie brutto.

**Outcome:** wskaźnik zatrudnienia według BAEL.

**Heterogeniczność:** stopa bezrobocia BAEL, wzrost realnego PKB, udział przemysłu (sekcje B–E) w wartości dodanej brutto oraz WDB na 1 pracującego.

**Benchmark:** Pooled OLS oraz Two-Way Fixed Effects z efektami stałymi województw i lat.

**Causal ML:** Double Machine Learning z XGBoost jako modelami nuisance oraz Causal Forest do estymacji zróżnicowanego efektu τ(x).

**SHAP:** interpretacja tego, które zmienne są związane z heterogenicznością oszacowanego efektu.

**H5:** zatrudnienie młodzieży jest wyłączone z głównej analizy.
        """
    )
    st.info(
        "RACH 3498 (poziom nominalnego PKB) i CENY 2496 nie są wymagane do estymacji modelu głównego. "
        "RACH 3502 dostarcza bezpośrednio dynamikę realnego PKB, a Kaitz jest relacją dwóch płac, "
        "więc wspólny deflator CPI nie zmienia jego wartości. Dzięki temu aplikacja nie uzależnia "
        "wyniku od dodatkowego etapu deflacji. "
    )

    st.warning(
        "Wyniki wymagają założeń identyfikacyjnych typowych dla danych obserwacyjnych. "
        "Causal Forest nie eliminuje automatycznie obciążenia wynikającego z nieobserwowanych czynników. "
        "Przy 16 województwach należy też ostrożnie traktować wnioskowanie oparte na klastrowanych SE."
    )
    st.markdown("### Pliki używane przez aplikację")
    for key, path in paths.items():
        st.write(f"**{key}:** {path.name}")
