# -*- coding: utf-8 -*-
"""
Heterogeniczny wpływ płacy minimalnej na zatrudnienie – panel województw (Polska)
=================================================================================
Aplikacja prezentacyjna (Streamlit) do pracy magisterskiej.

Architektura analizy
    1. Dane GUS (BDL) -> panel 16 województw x lata (parser odporny na wielopoziomowe
       nagłówki, zepsute kodowanie i kody TERYT 7- oraz 12-cyfrowe).
    2. Treatment: regionalny wskaźnik Kaitza = płaca minimalna / przeciętne wynagrodzenie
       w województwie (oba nominalne, więc iloraz jest tożsamy z ilorazem realnym).
    3. Benchmark ekonometryczny: pooled OLS -> FE -> TWFE (efekty województw i lat,
       błędy klastrowane po województwach, t(G-1), dzikie bootstrapowanie klastrów),
       event study oraz model z interakcjami (klasyczny test heterogeniczności).
    4. Causal ML: Causal Forest (Double ML, EconML) z nuisance-modelami XGBoost,
       cross-fitting grupowany po województwach.
    5. Interpretacja heterogeniczności: SHAP na modelu zastępczym CATE.
    6. Testy odporności.

Zabezpieczenia (nie usuwać):
    * twarde słowniki: MIN_WAGE_PL, PRZECIETNE_WYNAGRODZENIE_PL, WOJEWODZTWA_MAP,
    * nazwy województw wyłącznie ze słownika po kodzie TERYT (omija zepsute polskie znaki),
    * auto-wykrywanie plików w folderze + panel Drag & Drop w bocznym menu.
"""
from __future__ import annotations

import glob
import io
import os
import re
import unicodedata
import warnings
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Biblioteki ciężkie – import z flagami, żeby część "danych + ekonometria" działała zawsze
try:
    import statsmodels.api as sm
    HAS_SM = True
except Exception:  # pragma: no cover
    sm, HAS_SM = None, False
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:  # pragma: no cover
    xgb, HAS_XGB = None, False
try:
    import shap
    HAS_SHAP = True
except Exception:  # pragma: no cover
    shap, HAS_SHAP = None, False
try:
    from econml.dml import CausalForestDML
    HAS_ECONML = True
except Exception:  # pragma: no cover
    CausalForestDML, HAS_ECONML = None, False

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict

# =============================================================================
# 1. TWARDE DANE ZASZYTE W KODZIE (nie usuwać)
# =============================================================================
PRZECIETNE_WYNAGRODZENIE_PL = {
    1999: 1706.74, 2000: 1923.81, 2001: 2061.85, 2002: 2133.21, 2003: 2201.47,
    2004: 2289.57, 2005: 2380.29, 2006: 2477.23, 2007: 2691.03, 2008: 2943.88,
    2009: 3102.96, 2010: 3224.98, 2011: 3399.52, 2012: 3521.67, 2013: 3650.06,
    2014: 3783.46, 2015: 3899.78, 2016: 4047.21, 2017: 4271.51, 2018: 4585.03,
    2019: 4918.17, 2020: 5167.47, 2021: 5662.53, 2022: 6346.15, 2023: 7155.48,
    2024: 8181.72, 2025: 8903.56
}

# Płaca minimalna – wartości obowiązujące od 1 stycznia danego roku (zł brutto / mc)
MIN_WAGE_PL = {
    2010: 1317, 2011: 1386, 2012: 1500, 2013: 1600, 2014: 1680,
    2015: 1750, 2016: 1850, 2017: 2000, 2018: 2100, 2019: 2250,
    2020: 2600, 2021: 2800, 2022: 3010, 2023: 3490, 2024: 4242, 2025: 4666
}

# Kody TERYT (2 pierwsze cyfry kodu jednostki) -> poprawne nazwy województw
WOJEWODZTWA_MAP = {
    '02': 'Dolnośląskie', '04': 'Kujawsko-Pomorskie', '06': 'Lubelskie',
    '08': 'Lubuskie', '10': 'Łódzkie', '12': 'Małopolskie',
    '14': 'Mazowieckie', '16': 'Opolskie', '18': 'Podkarpackie',
    '20': 'Podlaskie', '22': 'Pomorskie', '24': 'Śląskie',
    '26': 'Świętokrzyskie', '28': 'Warmińsko-Mazurskie',
    '30': 'Wielkopolskie', '32': 'Zachodniopomorskie'
}

YEAR_MIN = min(MIN_WAGE_PL)          # 2010 – pierwszy rok z danymi o płacy minimalnej
YEAR_MAX = 2025
PRESAMPLE = (2005, 2009)             # okres sprzed próby – do wyznaczenia wynagrodzenia bazowego (ekspozycji)

# Specyfikacja plików GUS: prefiks -> (opis, status)
#   "wymagany" – bez niego panel nie powstanie,
#   "zalecany" – brak powoduje przejście na wariant zastępczy (z ostrzeżeniem),
#   "opcjonalny" – używany tylko do kontroli / robustness.
FILE_SPECS = {
    "RYNE_4100": ("Stopa bezrobocia wg BAEL, % (2010–2025)", "wymagany"),
    "RPR_3970":  ("Pracujący wg BAEL, kwartalnie, tys. osób (zmienna objaśniana)", "wymagany"),
    "RACH_3498": ("PKB ogółem, mln zł", "wymagany"),
    "RACH_3499": ("PKB na 1 mieszkańca, zł", "wymagany"),
    "RACH_3502": ("Dynamika PKB (r/r = 100), wolumen", "wymagany"),
    "RACH_3503": ("Dynamika PKB na 1 mieszkańca (r/r = 100)", "wymagany"),
    "RACH_3505": ("WDB wg sekcji PKD, mln zł (struktura sektorowa)", "wymagany"),
    "RACH_3510": ("WDB na 1 pracującego, zł (produktywność)", "wymagany"),
    "CENY_2496": ("CPI regionalny, kwartalnie (analogiczny okres r. poprz. = 100)", "wymagany"),
    "WYNA_2797": ("Przeciętne wynagrodzenie brutto wg województw i sekcji PKD, zł", "zalecany"),
    "LUDN_2137": ("Ludność wg wieku (kontrola; plik zawiera tylko 12 z 16 województw)", "opcjonalny"),
    "RYNE_4112": ("Wskaźnik zatrudnienia BAEL (dostępny dopiero od 2019)", "opcjonalny"),
    "RYNE_4098": ("Struktura pracujących wg sektorów BAEL (2021–2025)", "opcjonalny"),
    "RYNE_4498": ("Bezrobocie młodzieży (hipoteza H5 – wyłączona z analizy)", "opcjonalny"),
    "WYNA_2504": ("Wynagrodzenia kwartalne wg regionów (niewykorzystywany)", "opcjonalny"),
    "WYNA_4610": ("Wynagrodzenia miesięczne 2024–2026 (niewykorzystywany)", "opcjonalny"),
}
USED_FILES = ["RYNE_4100", "RPR_3970", "RACH_3498", "RACH_3499", "RACH_3502", "RACH_3503",
              "RACH_3505", "RACH_3510", "CENY_2496", "WYNA_2797", "LUDN_2137"]
REQUIRED_FILES = [k for k, v in FILE_SPECS.items() if v[1] == "wymagany"]

# Zmienne moderujące (heterogeniczność) – trzy wymiary keynesowskie + 2 kontrolne
MOD_COLS = ["Bezrobocie_BAEL", "Wzrost_PKB", "Udzial_Przemyslu",
            "Realny_PKB_per_capita", "Produktywnosc"]
KEYNES_COLS = ["Wzrost_PKB", "Bezrobocie_BAEL", "Udzial_Przemyslu"]   # popyt / rezerwy / struktura
MOD_LABELS = {
    "Bezrobocie_BAEL": "Bezrobocie (rezerwy rynku pracy, %)",
    "Wzrost_PKB": "Wzrost PKB (popyt zagregowany, % r/r)",
    "Udzial_Przemyslu": "Udział przemysłu w WDB (struktura)",
    "Realny_PKB_per_capita": "Realny PKB per capita (zł, ceny 2010)",
    "Produktywnosc": "Produktywność (realna WDB / pracującego, zł)",
}


class DataError(Exception):
    """Błąd danych wejściowych, który ma zostać pokazany użytkownikowi."""


# =============================================================================
# 2. NORMALIZACJA TEKSTU I KODOWANIA
# =============================================================================
_PL_FROM = "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"
_PL_TO = "acelnoszzACELNOSZZ"
_PL_TABLE = str.maketrans(_PL_FROM, _PL_TO)


def fix_encoding(s):
    """Naprawia typowe 'chińskie znaczki' (UTF-8 odczytane jako cp1250 i zapisane ponownie)."""
    if not isinstance(s, str):
        return s
    if any(ch in s for ch in ("Ă", "Ĺ", "Ä", "Â")):
        try:
            return s.encode("cp1250").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s
    return s


def norm(s) -> str:
    """Tekst -> małe litery, bez polskich znaków, pojedyncze spacje (do dopasowywania nagłówków)."""
    s = fix_encoding(str(s)).translate(_PL_TABLE)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.strip().lower())


def kod_to_woj_id(kod):
    """Kod jednostki GUS -> dwucyfrowy kod województwa ('02'...'32'), 'PL' dla Polski, None dla reszty.

    Obsługuje kody 7-cyfrowe (np. 0200000) i 12-cyfrowe z plików BAEL (np. 011200000000,
    gdzie województwo siedzi na pozycjach 3-4). Odrzuca powiaty, makroregiony i podregiony.
    """
    s = re.sub(r"\D", "", str(kod).split(".")[0])
    if not s:
        return None
    if set(s) == {"0"}:
        return "PL"
    if len(s) > 7:
        s = s.zfill(12)
        wid, rest = s[2:4], s[4:]
    else:
        s = s.zfill(7)
        wid, rest = s[:2], s[2:]
    if set(rest) != {"0"}:
        return None
    return wid if wid in WOJEWODZTWA_MAP else None


# =============================================================================
# 3. PARSER TABLIC GUS (xlsx z wielopoziomowym nagłówkiem + csv z nagłówkiem "a;b;c;rok;[jedn]")
# =============================================================================
def _num(x) -> float:
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace("\xa0", "").replace(" ", "").replace(",", ".")
    m = re.match(r"^-?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else np.nan


def _as_year(x):
    try:
        v = int(float(str(x).strip()))
    except (ValueError, TypeError):
        return None
    return v if 1900 <= v <= 2100 else None


def _map_df(df: pd.DataFrame, f) -> pd.DataFrame:
    return df.map(f) if hasattr(df, "map") else df.applymap(f)


@dataclass
class GusTable:
    """Tablica GUS w formie szerokiej: indeks = kod jednostki, kolumny = (etykieta, rok)."""
    name: str
    data: pd.DataFrame
    units: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def labels(self):
        return list(dict.fromkeys(self.data.columns.get_level_values(0)))

    def pick(self, pattern: str) -> pd.DataFrame:
        rx = re.compile(pattern)
        hits = [l for l in self.labels() if rx.search(l)]
        if len(hits) != 1:
            raise DataError(
                f"[{self.name}] wzorzec nagłówka '{pattern}' dopasował {len(hits)} kolumn(y): "
                f"{hits[:4]}. Dostępne etykiety (początek): {self.labels()[:6]}")
        return self.data[hits[0]]

    def long(self, pattern: str, out_name: str, keep_poland: bool = False,
             zero_is_missing: bool = False) -> pd.DataFrame:
        """Tablica szeroka -> format długi (Kod_Str, Rok, wartość). zero_is_missing: dla szeregów, które nie
        mogą wynosić 0 (GUS bywa wpisuje 0 zamiast braku danych) – zera zamieniane na braki i zgłaszane."""
        sub = self.pick(pattern).copy()
        ids = [kod_to_woj_id(k) for k in sub.index]
        sub.index = pd.Index(ids, name="Kod_Str")
        sub = sub[~pd.isna(sub.index)]
        if not keep_poland:
            sub = sub[sub.index != "PL"]
        sub.columns = pd.Index([int(c) for c in sub.columns], name="Rok")
        out = sub.reset_index().melt(id_vars="Kod_Str", var_name="Rok", value_name=out_name)
        out["Rok"] = out["Rok"].astype(int)
        if zero_is_missing:
            z = out[out[out_name] == 0]
            if not keep_poland:
                z = z[z["Kod_Str"] != "PL"]
            if len(z):
                spans = z.groupby("Kod_Str")["Rok"].agg(["min", "max"])
                yrs = {(int(a), int(b)) for a, b in zip(spans["min"], spans["max"])}
                span_txt = "/".join(f"{a}" if a == b else f"{a}–{b}" for a, b in sorted(yrs))
                if len(spans) == 16:
                    where = f"wszystkie województwa, {span_txt}"
                else:
                    where = ", ".join(WOJEWODZTWA_MAP.get(k, k) for k in spans.index) + f", {span_txt}"
                self.notes.append((self.name, out_name, where))
            out.loc[out[out_name] == 0, out_name] = np.nan
        return out.dropna(subset=[out_name]).drop_duplicates(["Kod_Str", "Rok"])


def _is_code(x) -> bool:
    return bool(re.fullmatch(r"\d{6,12}", str(x).strip().replace(".0", "")))


def _build_table(name, codes, header_cols, values) -> GusTable:
    """header_cols: lista (label, year, unit). values: ndarray [n_codes x n_cols]."""
    cols = pd.MultiIndex.from_arrays(
        [[h[0] for h in header_cols], [h[1] for h in header_cols]], names=["label", "Rok"])
    df = pd.DataFrame(values, index=pd.Index(codes, name="Kod"), columns=cols)
    units = {h[0]: h[2] for h in header_cols}
    return GusTable(name=name, data=df, units=units)


def parse_xlsx(name: str, data: bytes) -> GusTable:
    xl = pd.ExcelFile(io.BytesIO(data))
    sheet = "TABLICA" if "TABLICA" in xl.sheet_names else xl.sheet_names[-1]
    raw = xl.parse(sheet, header=None, dtype=object)
    first = next((i for i in range(len(raw)) if _is_code(raw.iat[i, 0])), None)
    if first is None:
        raise DataError(f"[{name}] nie znaleziono wierszy z kodami jednostek (arkusz '{sheet}').")
    hdr = raw.iloc[:first, 2:]
    year_row = max(range(first), key=lambda r: sum(_as_year(v) is not None for v in hdr.iloc[r]))
    unit_row = max(range(first), key=lambda r: sum(str(v).strip().startswith("[") for v in hdr.iloc[r]))
    label_rows = [r for r in range(first) if r not in (year_row, unit_row)]
    lab = hdr.iloc[label_rows].T.ffill().T if label_rows else None
    header_cols = []
    for j in range(hdr.shape[1]):
        year = _as_year(hdr.iat[year_row, j])
        if year is None:
            continue
        parts = []
        if lab is not None:
            parts = [norm(lab.iat[i, j]) for i in range(lab.shape[0]) if pd.notna(lab.iat[i, j])]
        unit = str(hdr.iat[unit_row, j]).strip() if unit_row != year_row else ""
        header_cols.append((" | ".join(parts) or "wartosc", year, unit, j))
    body = raw.iloc[first:, :]
    body = body[[_is_code(x) for x in body.iloc[:, 0]]]
    codes = [re.sub(r"\.0$", "", str(x).strip()) for x in body.iloc[:, 0]]
    vals = _map_df(body.iloc[:, 2:], _num).to_numpy(dtype=float)
    vals = vals[:, [h[3] for h in header_cols]]
    return _build_table(name, codes, [h[:3] for h in header_cols], vals)


def parse_csv(name: str, data: bytes) -> GusTable:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("cp1250", errors="replace")
    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str, keep_default_na=False)
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    kod_col = next((c for c in df.columns if norm(c) == "kod"), df.columns[0])
    df = df[[_is_code(x) for x in df[kod_col]]]
    codes = [str(x).strip() for x in df[kod_col]]
    header_cols, keep = [], []
    for j, col in enumerate(df.columns):
        if j < 2:
            continue
        parts = [p.strip() for p in fix_encoding(str(col)).split(";")]
        year = next((int(p) for p in parts if re.fullmatch(r"(19|20)\d{2}", p)), None)
        if year is None:
            continue
        unit = next((p for p in parts if p.startswith("[")), "")
        label = " | ".join(norm(p) for p in parts
                           if p and not re.fullmatch(r"(19|20)\d{2}", p) and not p.startswith("[")) or "wartosc"
        header_cols.append((label, year, unit))
        keep.append(col)
    vals = _map_df(df[keep], _num).to_numpy(dtype=float)
    return _build_table(name, codes, header_cols, vals)


def parse_gus(name: str, data: bytes) -> GusTable:
    return parse_csv(name, data) if name.lower().endswith(".csv") else parse_xlsx(name, data)


# =============================================================================
# 4. WYKRYWANIE PLIKÓW (auto-wykrywanie w folderze; Drag & Drop obsługuje interfejs)
# =============================================================================
_PREFIX_RX = re.compile(r"^([A-Za-z]{3,4}_\d{3,4})")


def prefix_of(filename: str):
    m = _PREFIX_RX.match(os.path.basename(filename))
    return m.group(1).upper() if m else None


def search_dirs():
    dirs = []
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        dirs += [base, os.path.join(base, "data")]
    except NameError:  # pragma: no cover
        pass
    dirs += [os.getcwd(), os.path.join(os.getcwd(), "data")]
    out = []
    for d in dirs:
        if os.path.isdir(d) and d not in out:
            out.append(d)
    return out


def discover_files(dirs=None) -> dict:
    """Zwraca {prefiks: (nazwa_pliku, bajty)}. Przy duplikatach wygrywa plik z najnowszym znacznikiem w nazwie."""
    found = {}
    for d in (dirs or search_dirs()):
        for p in sorted(glob.glob(os.path.join(d, "*"))):
            if not p.lower().endswith((".xlsx", ".csv")):
                continue
            key = prefix_of(p)
            if key in FILE_SPECS:
                with open(p, "rb") as fh:
                    found[key] = (os.path.basename(p), fh.read())
    return found


# =============================================================================
# 5. BUDOWA PANELU
# =============================================================================
def _merge_all(frames):
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["Kod_Str", "Rok"], how="outer")
    return out


def _quarter_mean(tbl: GusTable, template: str, out_name: str, min_q: int = 4) -> pd.DataFrame:
    """Średnia roczna z kwartałów (wymagane komplet kwartałów, inaczej NaN)."""
    parts = []
    for q in (1, 2, 3, 4):
        f = tbl.long(template.format(q=q), f"q{q}", keep_poland=True)
        parts.append(f.set_index(["Kod_Str", "Rok"])[f"q{q}"])
    m = pd.concat(parts, axis=1)
    m[out_name] = np.where(m.notna().sum(axis=1) >= min_q, m.mean(axis=1), np.nan)
    return m[[out_name]].dropna().reset_index()


def _price_levels(tbl: GusTable) -> pd.DataFrame:
    """Regionalny CPI z kwartalnych indeksów łańcuchowych ('okres poprzedni = 100').
    Zwraca (Kod_Str, Rok): Poziom_Cen (średnia roczna z 4 kwartałów, skala dowolna) oraz CPI_Region
    (średnioroczny wskaźnik cen r/r, poprzedni rok = 100)."""
    parts = []
    for q in (1, 2, 3, 4):
        f = tbl.long(rf"^{q} kwartal \| ogolem - okres poprzedni = 100$", "idx", keep_poland=True)
        f["q"] = q
        parts.append(f)
    m = pd.concat(parts).sort_values(["Kod_Str", "Rok", "q"])
    m["lvl"] = m.groupby("Kod_Str")["idx"].transform(lambda v: (v / 100).cumprod())
    a = m.groupby(["Kod_Str", "Rok"]).agg(Poziom_Cen=("lvl", "mean"), nq=("lvl", "size")).reset_index()
    a.loc[a["nq"] < 4, "Poziom_Cen"] = np.nan
    a = a.sort_values(["Kod_Str", "Rok"])
    prev = a.groupby("Kod_Str")["Poziom_Cen"].shift(1)
    contiguous = a.groupby("Kod_Str")["Rok"].diff() == 1
    a["CPI_Region"] = np.where(contiguous, a["Poziom_Cen"] / prev * 100, np.nan)
    return a.drop(columns="nq")


def add_shift(df: pd.DataFrame, cols, k: int, suffix: str) -> pd.DataFrame:
    """Przesunięcie w czasie po (województwo, rok) – odporne na luki w latach."""
    lag = df[["Kod_Str", "Rok"] + list(cols)].copy()
    lag["Rok"] = lag["Rok"] + k
    lag = lag.rename(columns={c: f"{c}{suffix}" for c in cols})
    return df.merge(lag, on=["Kod_Str", "Rok"], how="left")


def build_panel(sources: dict):
    """sources: {prefiks: (nazwa_pliku, bajty)} -> (panel DataFrame, słownik diagnostyczny)."""
    diag = {"warnings": [], "notes": [], "checks": {}}
    missing = [k for k in REQUIRED_FILES if k not in sources]
    if missing:
        raise DataError("Brakuje wymaganych plików: " + ", ".join(
            f"{k} ({FILE_SPECS[k][0]})" for k in missing))
    T = {k: parse_gus(sources[k][0], sources[k][1]) for k in USED_FILES if k in sources}

    # --- rynek pracy ------------------------------------------------------------------
    unemp = T["RYNE_4100"].long(r"^ogolem \| wartosc liczbowa$", "Bezrobocie_BAEL", zero_is_missing=True)
    emp = _quarter_mean(T["RPR_3970"], r"^{q} kwartal \| ogolem \| ogolem \| ogolem \| wartosc liczbowa$",
                        "Zatrudnieni_tys")
    emp = emp[emp["Kod_Str"] != "PL"]

    # --- rachunki regionalne ----------------------------------------------------------
    pkb = T["RACH_3498"].long(r"^produkt krajowy brutto ogolem$", "PKB_mln")
    pkb_pc = T["RACH_3499"].long(r"^produkt krajowy brutto na 1 mieszkanca$", "PKB_pc")
    dyn = T["RACH_3502"].long(r"^dynamika produktu krajowego brutto ogolem, rok poprzedni=100$", "Dyn_PKB")
    dyn_pc = T["RACH_3503"].long(
        r"^dynamika produktu krajowego brutto na 1 mieszkanca, rok poprzedni=100$", "Dyn_PKB_pc")
    wdb_tot = T["RACH_3505"].long(r"^ogolem$", "WDB_Ogolem", zero_is_missing=True)
    wdb_sec = [T["RACH_3505"].long(rf"^sekcja {s}$", f"WDB_{s.upper()}", zero_is_missing=True)
               for s in "abcde"]
    wdb_pw = T["RACH_3510"].long(r"^wartosc dodana brutto na 1 pracujacego$", "WDB_na_pracujacego",
                                    zero_is_missing=True)

    # --- ceny -------------------------------------------------------------------------
    cpi_all = _price_levels(T["CENY_2496"])
    cpi_pl = cpi_all[cpi_all["Kod_Str"] == "PL"].set_index("Rok")
    cpi = cpi_all[cpi_all["Kod_Str"] != "PL"]

    frames = [unemp, emp, pkb, pkb_pc, dyn, dyn_pc, wdb_tot] + wdb_sec + [wdb_pw, cpi]

    # --- wynagrodzenia regionalne (z fallbackiem na dane krajowe zaszyte w kodzie) -----
    wage_pl_gus, pre_wage = None, None
    if "WYNA_2797" in T:
        w_all = T["WYNA_2797"].long(r"^ogolem$", "Wynagrodzenie_Region", keep_poland=True)
        wage_pl_gus = w_all[w_all["Kod_Str"] == "PL"].set_index("Rok")["Wynagrodzenie_Region"]
        pre_wage = (w_all[(w_all["Kod_Str"] != "PL") & w_all["Rok"].between(PRESAMPLE[0], PRESAMPLE[1])]
                    .groupby("Kod_Str")["Wynagrodzenie_Region"].mean())
        frames.append(w_all[w_all["Kod_Str"] != "PL"])
    else:
        diag["warnings"].append(
            "Brak pliku WYNA_2797 – wskaźnik Kaitza liczony z KRAJOWEGO przeciętnego wynagrodzenia "
            "(słownik w kodzie). Treatment nie ma wtedy zmienności regionalnej i jest współliniowy z "
            "efektami czasu – wyniki TWFE/DML będą niewiarygodne.")

    df = _merge_all(frames)
    df = df[df["Kod_Str"].isin(WOJEWODZTWA_MAP)].copy()
    df["Wojewodztwo"] = df["Kod_Str"].map(WOJEWODZTWA_MAP)   # nazwy ze słownika, nie z plików
    df = df.sort_values(["Kod_Str", "Rok"]).reset_index(drop=True)

    # --- ludność (wyprowadzona z PKB i PKB per capita – komplet 16 województw) ----------
    df["Populacja"] = df["PKB_mln"] * 1e6 / df["PKB_pc"]
    df["ln_Populacja"] = np.log(df["Populacja"])
    n_lud = 0
    if "LUDN_2137" in T:                       # ludność wg bilansu GUS – tylko kontrola / robustness
        lud = T["LUDN_2137"].long(r"^ogolem \| ogolem$", "Ludnosc_GUS")
        n_lud = lud["Kod_Str"].nunique()
        df = df.merge(lud, on=["Kod_Str", "Rok"], how="left")

    # --- zmienna objaśniana -------------------------------------------------------------
    df["Stopa_Zatrudnienia"] = df["Zatrudnieni_tys"] * 1000 / df["Populacja"] * 100
    df["ln_Zatrudnieni"] = np.log(df["Zatrudnieni_tys"])
    if "Ludnosc_GUS" in df:
        df["Stopa_Zatr_LUDN"] = df["Zatrudnieni_tys"] * 1000 / df["Ludnosc_GUS"] * 100

    # --- płaca minimalna, wynagrodzenia i wskaźnik Kaitza (treatment) -------------------
    df["Placa_Minimalna"] = df["Rok"].map(MIN_WAGE_PL)
    df["Przecietne_Wynagrodzenie_Kraj"] = df["Rok"].map(PRZECIETNE_WYNAGRODZENIE_PL)
    df["Kaitz_Krajowy"] = df["Placa_Minimalna"] / df["Przecietne_Wynagrodzenie_Kraj"]
    if "Wynagrodzenie_Region" in df:
        df["Kaitz_Index"] = df["Placa_Minimalna"] / df["Wynagrodzenie_Region"]
    else:
        df["Wynagrodzenie_Region"] = df["Przecietne_Wynagrodzenie_Kraj"]
        df["Kaitz_Index"] = df["Kaitz_Krajowy"]
    df["Kaitz_pp"] = df["Kaitz_Index"] * 100          # skala w punktach procentowych
    # Kaitz "ekspozycyjny": płaca minimalna / (bazowe wynagrodzenie woj. z lat 2005-2009 x krajowy wzrost płac).
    # Zmienność tylko z krajowej płacy minimalnej x przedsamplowa siła oddziaływania w regionie;
    # wolny od endogeniczności lokalnych płac (popyt na pracę podnosi płace i zatrudnienie jednocześnie).
    if pre_wage is not None:
        wpl_base = np.mean([PRZECIETNE_WYNAGRODZENIE_PL[y] for y in range(PRESAMPLE[0], PRESAMPLE[1] + 1)])
        df["Wynagrodzenie_Bazowe"] = df["Kod_Str"].map(pre_wage)
        wage_pred = df["Wynagrodzenie_Bazowe"] * df["Przecietne_Wynagrodzenie_Kraj"] / wpl_base
        df["Kaitz_Ekspozycja_pp"] = 100 * df["Placa_Minimalna"] / wage_pred
        bite = -np.log(pre_wage)
        df["Bite_z"] = df["Kod_Str"].map((bite - bite.mean()) / bite.std(ddof=0))
    else:
        df["Wynagrodzenie_Bazowe"] = np.nan
        df["Kaitz_Ekspozycja_pp"] = np.nan
        df["Bite_z"] = np.nan

    # --- deflowanie CPI (regionalny indeks cen, średnia roczna, 2010 = 100) -------------
    base_lvl = df[df["Rok"] == YEAR_MIN].set_index("Kod_Str")["Poziom_Cen"]
    df["Indeks_Cen"] = 100 * df["Poziom_Cen"] / df["Kod_Str"].map(base_lvl)
    df["Placa_Min_Realna"] = df["Placa_Minimalna"] / (df["Indeks_Cen"] / 100)
    df["Wynagrodzenie_Realne"] = df["Wynagrodzenie_Region"] / (df["Indeks_Cen"] / 100)

    # --- PKB: realny wzrost, realny PKB per capita (ceny 2010), struktura ---------------
    df["Wzrost_PKB"] = df["Dyn_PKB"] - 100
    g = (df["Dyn_PKB_pc"] / 100).where(df["Rok"] > YEAR_MIN, 1.0)
    base = df[df["Rok"] == YEAR_MIN].set_index("Kod_Str")["PKB_pc"]
    df["Realny_PKB_per_capita"] = df["Kod_Str"].map(base) * g.groupby(df["Kod_Str"]).cumprod()
    df["WDB_Przemysl"] = df[["WDB_B", "WDB_C", "WDB_D", "WDB_E"]].sum(axis=1, min_count=4)
    df["Udzial_Przemyslu"] = df["WDB_Przemysl"] / df["WDB_Ogolem"]
    df["Udzial_Rolnictwa"] = df["WDB_A"] / df["WDB_Ogolem"] if "WDB_A" in df else np.nan
    df["Produktywnosc"] = df["WDB_na_pracujacego"] / (df["Indeks_Cen"] / 100)

    # --- opóźnienia moderatorów (ograniczają problem 'bad controls') --------------------
    lag_cols = MOD_COLS + ["CPI_Region", "Kaitz_pp"]
    df = add_shift(df, lag_cols, 1, "_L1")
    df = add_shift(df, ["Kaitz_pp"], -1, "_F1")        # wyprzedzenie – do testu "lead"

    # --- zakres lat i kompletność -------------------------------------------------------
    df = df[(df["Rok"] >= YEAR_MIN) & (df["Rok"] <= YEAR_MAX)]
    df = df.dropna(subset=["Stopa_Zatrudnienia", "Kaitz_Index", "Wzrost_PKB"]).reset_index(drop=True)
    if df.empty:
        raise DataError("Panel jest pusty po połączeniu plików – sprawdź zgodność lat i kodów jednostek.")

    # --- kontrole poprawności -----------------------------------------------------------
    if wage_pl_gus is not None:
        chk = pd.DataFrame({"GUS_WYNA_2797_Polska": wage_pl_gus,
                            "Slownik_w_kodzie": pd.Series(PRZECIETNE_WYNAGRODZENIE_PL)}).dropna()
        chk["Roznica_%"] = (chk["GUS_WYNA_2797_Polska"] / chk["Slownik_w_kodzie"] - 1) * 100
        diag["checks"]["wynagrodzenie_krajowe"] = chk[(chk.index >= YEAR_MIN) & (chk.index <= YEAR_MAX)]
        mx = diag["checks"]["wynagrodzenie_krajowe"]["Roznica_%"].abs().max()
        if mx > 3:
            diag["warnings"].append(
                f"Przeciętne wynagrodzenie w WYNA_2797 (Polska) różni się od słownika w kodzie o maks. {mx:.1f}% – "
                "sprawdź definicję (gospodarka narodowa vs sektor przedsiębiorstw).")
    if "Ludnosc_GUS" in df:
        m = df.dropna(subset=["Ludnosc_GUS"]).copy()
        m["Roznica_%"] = (m["Populacja"] / m["Ludnosc_GUS"] - 1) * 100
        diag["checks"]["ludnosc"] = m[["Wojewodztwo", "Rok", "Populacja", "Ludnosc_GUS", "Roznica_%"]]
        diag["notes"].append(
            f"Plik LUDN_2137 zawiera {n_lud} z 16 województw – ludność w modelu pochodzi z ilorazu "
            "PKB / PKB per capita (komplet 16 województw); LUDN służy do kontroli i testu odporności "
            f"(maks. rozbieżność {m['Roznica_%'].abs().max():.1f}%, skoki w 2020 i 2022 – rewizja spisowa).")
    grouped = {}
    for t_ in T.values():
        for tbl_name, var, where in t_.notes:
            grouped.setdefault((tbl_name, where), []).append(var)
    for (tbl_name, where), vars_ in grouped.items():
        diag["warnings"].append(
            f"{tbl_name}: zera w szeregach {', '.join(vars_)} potraktowano jako braki danych ({where}).")
    diag["notes"].append(
        "Zmienna objaśniana: pracujący wg BAEL (15–89 lat, średnia z 4 kwartałów) na 100 mieszkańców. "
        "Płaca minimalna: wartości styczniowe ze słownika w kodzie.")
    nreg = df["Kod_Str"].nunique()
    diag["panel"] = {"n": len(df), "regions": nreg, "years": (int(df.Rok.min()), int(df.Rok.max())),
                     "balanced": bool((df.groupby("Kod_Str").size() == df.Rok.nunique()).all())}
    if nreg != 16:
        diag["warnings"].append(f"W panelu jest {nreg} z 16 województw.")
    return df, diag


# =============================================================================
# 6. EKONOMETRIA LINIOWA (numpy/scipy) – OLS/FE/TWFE, SE klastrowane, dzikie bootstrapowanie
# =============================================================================
@dataclass
class LinRes:
    names: list
    beta: np.ndarray
    V: np.ndarray
    se: np.ndarray
    t: np.ndarray
    p: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    n: int
    G: int
    K: int
    r2: float
    resid: np.ndarray
    X: np.ndarray          # macierz regresorów po transformacji (within)
    y: np.ndarray          # zmienna objaśniana po transformacji
    gid: np.ndarray        # numer klastra 0..G-1
    fe: str

    def tidy(self, keep=None) -> pd.DataFrame:
        d = pd.DataFrame({"zmienna": self.names, "współczynnik": self.beta, "SE (klaster)": self.se,
                          "t": self.t, "p-value": self.p, "CI95 dół": self.lo, "CI95 góra": self.hi})
        if keep is not None:
            d = d[d["zmienna"].isin(keep)]
        return d.reset_index(drop=True)

    def coef(self, name):
        i = self.names.index(name)
        return float(self.beta[i]), float(self.se[i]), float(self.p[i])


def _cr_vcov(X, u, gid, G, K):
    """Klastrowa macierz wariancji CR1: G/(G-1) * (N-1)/(N-K) * (X'X)^-1 [sum_g X_g'u_g u_g'X_g] (X'X)^-1."""
    N = X.shape[0]
    XtX_inv = np.linalg.pinv(X.T @ X)
    scores = np.zeros((G, X.shape[1]))
    np.add.at(scores, gid, X * u[:, None])
    meat = scores.T @ scores
    c = G / (G - 1) * (N - 1) / max(N - K, 1)
    return c * XtX_inv @ meat @ XtX_inv


def fit_linear(df: pd.DataFrame, y: str, xs: list, fe: str = "twfe", cluster: str = "Kod_Str") -> LinRes:
    """fe: 'pooled' (stała), 'entity' (FE województw), 'twfe' (FE województw + lat).
    Błędy standardowe klastrowane po województwach (CR1), wnioskowanie z rozkładu t(G-1)."""
    d = df.dropna(subset=[y] + list(xs)).reset_index(drop=True)
    names = list(xs)
    X = d[xs].to_numpy(float)
    yy = d[y].to_numpy(float)
    ent = d["Kod_Str"].to_numpy()
    if fe == "twfe":
        D = pd.get_dummies(d["Rok"], prefix="Rok", drop_first=True, dtype=float)
        X = np.column_stack([X, D.to_numpy()])
        names += list(D.columns)
    if fe in ("entity", "twfe"):
        Xdf = pd.DataFrame(X)
        X = (Xdf - Xdf.groupby(ent).transform("mean")).to_numpy()
        yy = (pd.Series(yy) - pd.Series(yy).groupby(ent).transform("mean")).to_numpy()
        K = X.shape[1]
    else:  # pooled
        X = np.column_stack([np.ones(len(d)), X])
        names = ["const"] + names
        K = X.shape[1]
    gid, uniq = pd.factorize(d[cluster])
    G = len(uniq)
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    u = yy - X @ beta
    V = _cr_vcov(X, u, gid, G, K)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df=G - 1)
    crit = stats.t.ppf(0.975, df=G - 1)
    tss = np.sum((yy - (yy.mean() if fe == "pooled" else 0)) ** 2)
    r2 = 1 - np.sum(u ** 2) / tss if tss > 0 else np.nan
    return LinRes(names, beta, V, se, t, p, beta - crit * se, beta + crit * se,
                  len(d), G, K, float(r2), u, X, yy, gid, fe)


def wald_F(res: LinRes, vars_: list):
    """Łączny test Walda (wersja F, mianownik G-1) dla wskazanych współczynników."""
    idx = [res.names.index(v) for v in vars_]
    b = res.beta[idx]
    Vb = res.V[np.ix_(idx, idx)]
    q = len(idx)
    F = float(b @ np.linalg.pinv(Vb) @ b / q)
    return F, float(stats.f.sf(F, q, res.G - 1)), q


def wild_cluster_p(res: LinRes, var: str, B: int = 999, seed: int = 42) -> float:
    """Dzikie bootstrapowanie klastrów (WCR, wagi Rademachera) – wartość p dla H0: beta_var = 0.
    Zalecane przy małej liczbie klastrów (tu G = 16 województw)."""
    j = res.names.index(var)
    X, y, gid, G = res.X, res.y, res.gid, res.G
    rng = np.random.default_rng(seed)
    keep = [i for i in range(X.shape[1]) if i != j]
    Xr = X[:, keep]
    br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
    fit_r = Xr @ br
    u_r = y - fit_r
    XtX_inv = np.linalg.pinv(X.T @ X)
    P = XtX_inv @ X.T
    N, K = X.shape
    c = G / (G - 1) * (N - 1) / max(N - K, 1)
    t_obs = abs(res.t[j])
    cnt = 0
    for _ in range(B):
        w = rng.choice([-1.0, 1.0], size=G)[gid]
        ys = fit_r + w * u_r
        b = P @ ys
        u = ys - X @ b
        sc = np.zeros((G, K))
        np.add.at(sc, gid, X * u[:, None])
        Vb = c * XtX_inv @ (sc.T @ sc) @ XtX_inv
        tb = b[j] / np.sqrt(max(Vb[j, j], 1e-300))
        cnt += abs(tb) >= t_obs
    return float(cnt / B)


def interaction_model(df: pd.DataFrame, y: str, treat: str, mods: list, controls: list, fe: str = "twfe"):
    """Treatment x (moderator - średnia): klasyczny test heterogeniczności keynesowskiej."""
    d = df.dropna(subset=[y, treat] + mods + controls).copy()
    ints = []
    for m in mods:
        d[f"{m}_c"] = d[m] - d[m].mean()
        d[f"Kaitz × {m}"] = d[treat] * d[f"{m}_c"]
        ints.append(f"Kaitz × {m}")
    xs = [treat] + ints + [c for c in controls]
    res = fit_linear(d, y, xs, fe=fe)
    F, p, q = wald_F(res, ints)
    return res, ints, (F, p, q)


def event_study(df: pd.DataFrame, y: str = "Stopa_Zatrudnienia", ref_year: int = 2015):
    """Event study ciągłej 'intensywności': standaryzowany przedsamplowy bite (niskie wynagrodzenia 2005–2009)
    x dummies lat, FE województw i lat. Lata < ref_year to okres 'przed' (test trendów wstępnych):
    płaca minimalna rosła wolno do 2015 r., a istotnie szybciej po 2016 r."""
    d = df.dropna(subset=["Bite_z", y]).copy()
    years = [yr for yr in sorted(d["Rok"].unique()) if yr != ref_year]
    xs = []
    for yr in years:
        d[f"ES_{yr}"] = d["Bite_z"] * (d["Rok"] == yr)
        xs.append(f"ES_{yr}")
    res = fit_linear(d, y, xs, fe="twfe")
    tab = res.tidy(keep=xs)
    tab["Rok"] = [int(v.split("_")[1]) for v in tab["zmienna"]]
    pre = [f"ES_{yr}" for yr in years if yr < ref_year]
    pre_test = wald_F(res, pre) if pre else None
    return tab.sort_values("Rok").reset_index(drop=True), pre_test


def fit_iv(df: pd.DataFrame, y: str, endog: str, instr: str, exog: list = (), cluster: str = "Kod_Str"):
    """2SLS w modelu TWFE (FE województw + dummies lat), SE klastrowane CR1, wnioskowanie t(G-1).
    Zwraca (słownik wyników dla zmiennej endogenicznej, pierwszy stopień, postać zredukowana)."""
    exog = list(exog)
    d = df.dropna(subset=[y, endog, instr] + exog).reset_index(drop=True)
    D = pd.get_dummies(d["Rok"], drop_first=True, dtype=float).to_numpy()
    ent = d["Kod_Str"].to_numpy()

    def within(M):
        M = pd.DataFrame(M)
        return (M - M.groupby(ent).transform("mean")).to_numpy()

    X = within(np.column_stack([d[[endog] + exog].to_numpy(float), D]))
    Z = within(np.column_stack([d[[instr] + exog].to_numpy(float), D]))
    yy = within(d[[y]].to_numpy(float))[:, 0]
    Xhat = Z @ np.linalg.pinv(Z.T @ Z) @ (Z.T @ X)
    bread = np.linalg.pinv(Xhat.T @ Xhat)
    beta = bread @ (Xhat.T @ yy)
    u = yy - X @ beta
    gid, uniq = pd.factorize(d[cluster])
    G, N, K = len(uniq), len(d), X.shape[1]
    sc = np.zeros((G, K))
    np.add.at(sc, gid, Xhat * u[:, None])
    V = G / (G - 1) * (N - 1) / max(N - K, 1) * bread @ (sc.T @ sc) @ bread
    se = float(np.sqrt(V[0, 0]))
    t = float(beta[0] / se)
    out = {"współczynnik": float(beta[0]), "SE": se, "t": t, "p-value": float(2 * stats.t.sf(abs(t), G - 1)),
           "N": N, "G": G}
    first = fit_linear(d, endog, [instr] + exog, fe="twfe")
    reduced = fit_linear(d, y, [instr] + exog, fe="twfe")
    out["F pierwszego stopnia"] = float(first.t[0] ** 2)
    return out, first, reduced


def leave_one_region_out(df, y, xs, target="Kaitz_pp"):
    rows = []
    for k in sorted(df["Kod_Str"].unique()):
        r = fit_linear(df[df["Kod_Str"] != k], y, xs, fe="twfe")
        b, se, p = r.coef(target)
        rows.append({"pominięte województwo": WOJEWODZTWA_MAP[k], "współczynnik": b, "SE": se, "p-value": p})
    return pd.DataFrame(rows)


def robustness_linear(df: pd.DataFrame, controls: list, B: int = 499, seed: int = 42):
    """Zestaw testów odporności modelu TWFE. Zwraca (tabela, szczegóły)."""
    T = "Kaitz_pp"
    y0 = "Stopa_Zatrudnienia"
    rows = []

    def add(label, res, var=T, note=""):
        b, se, p = res.coef(var)
        rows.append({"specyfikacja": label, "współczynnik": b, "SE": se, "p-value": p,
                     "N": res.n, "uwagi": note})

    base = fit_linear(df, y0, [T] + controls, fe="twfe")
    add("Baza: TWFE, klaster po woj.", base)
    pb = wild_cluster_p(base, T, B=B, seed=seed)
    rows.append({"specyfikacja": "Baza: dzikie bootstrapowanie klastrów (WCR)", "współczynnik": base.coef(T)[0],
                 "SE": np.nan, "p-value": pb, "N": base.n, "uwagi": f"B={B}, wagi Rademachera"})
    add("Bez zmiennych kontrolnych", fit_linear(df, y0, [T], fe="twfe"))
    if df["Kaitz_Ekspozycja_pp"].notna().any():
        add("Treatment ekspozycyjny (MW / bazowe wynagrodzenie 2005–09)",
            fit_linear(df, y0, ["Kaitz_Ekspozycja_pp"] + controls, fe="twfe"), var="Kaitz_Ekspozycja_pp",
            note="reduced form; wolny od endogeniczności lokalnych płac")
        iv, first, _ = fit_iv(df, y0, T, "Kaitz_Ekspozycja_pp", controls)
        rows.append({"specyfikacja": "TWFE-IV: Kaitz instrumentowany ekspozycją", "współczynnik": iv["współczynnik"],
                     "SE": iv["SE"], "p-value": iv["p-value"], "N": iv["N"],
                     "uwagi": f"F pierwszego stopnia = {iv['F pierwszego stopnia']:.1f}"})
    add("Bez 2020 (COVID)", fit_linear(df[df["Rok"] != 2020], y0, [T] + controls, fe="twfe"))
    add("Bez 2020–2021", fit_linear(df[~df["Rok"].isin([2020, 2021])], y0, [T] + controls, fe="twfe"))
    add("Bez Mazowieckiego", fit_linear(df[df["Kod_Str"] != "14"], y0, [T] + controls, fe="twfe"))
    add("Okres 2010–2019", fit_linear(df[df["Rok"] <= 2019], y0, [T] + controls, fe="twfe"))
    add("Okres 2015–2024", fit_linear(df[df["Rok"] >= 2015], y0, [T] + controls, fe="twfe"))
    d2 = df.dropna(subset=["Kaitz_pp_L1", "Kaitz_pp_F1"])
    add("Test 'lead' (Kaitz t+1) – oczekiwane ≈ 0",
        fit_linear(d2, y0, [T, "Kaitz_pp_F1"] + controls, fe="twfe"), var="Kaitz_pp_F1",
        note="istotny lead sugeruje naruszenie ścisłej egzogeniczności")
    r_lag = fit_linear(d2, y0, [T, "Kaitz_pp_L1"] + controls, fe="twfe")
    add("Rozkład opóźnień: Kaitz t", r_lag, var=T)
    add("Rozkład opóźnień: Kaitz t-1", r_lag, var="Kaitz_pp_L1")
    add("Wynik alternatywny: ln(pracujący)", fit_linear(df, "ln_Zatrudnieni", [T] + controls, fe="twfe"),
        note="współczynnik ≈ zmiana log. na 1 p.p. Kaitza")
    if "Stopa_Zatr_LUDN" in df and df["Stopa_Zatr_LUDN"].notna().sum() > 60:
        add("Wynik alternatywny: zatrudnieni / ludność wg LUDN (12 woj.)",
            fit_linear(df.dropna(subset=["Stopa_Zatr_LUDN"]), "Stopa_Zatr_LUDN", [T] + controls, fe="twfe"),
            note="mianownik z bilansu ludności GUS")
    add("Wynik alternatywny: stopa bezrobocia BAEL",
        fit_linear(df, "Bezrobocie_BAEL", [T] + [c for c in controls if c != "Bezrobocie_BAEL_L1"], fe="twfe"))
    return pd.DataFrame(rows), base


# =============================================================================
# 7. CAUSAL MACHINE LEARNING: Causal Forest (Double ML) + XGBoost + SHAP
# =============================================================================
def label_of(col: str) -> str:
    base = col[:-3] if col.endswith("_L1") else col
    txt = MOD_LABELS.get(base, base)
    return txt + (" [t-1]" if col.endswith("_L1") else "")


def short_label(col: str) -> str:
    base = col[:-3] if col.endswith("_L1") else col
    return {"Bezrobocie_BAEL": "Bezrobocie", "Wzrost_PKB": "Wzrost PKB", "Udzial_Przemyslu": "Udział przemysłu",
            "Realny_PKB_per_capita": "PKB per capita", "Produktywnosc": "Produktywność"}.get(base, base) + \
        (" (t-1)" if col.endswith("_L1") else "")


def make_gbm(seed: int, n_estimators: int = 150):
    """Model nuisance / zastępczy: XGBoost (a gdy go brak – HistGradientBoosting z scikit-learn)."""
    if HAS_XGB:
        return xgb.XGBRegressor(n_estimators=n_estimators, max_depth=3, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0, random_state=seed,
                                n_jobs=1, tree_method="hist", verbosity=0)
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=n_estimators, random_state=seed)


def causal_columns(lag: bool):
    suf = "_L1" if lag else ""
    xcols = [c + suf for c in MOD_COLS]
    wcols = ["ln_Populacja", "CPI_Region" + suf]
    return xcols, wcols


def prepare_causal_sample(df: pd.DataFrame, treat: str, xcols: list, wcols: list, fe_demean: bool = True,
                          y: str = "Stopa_Zatrudnienia") -> pd.DataFrame:
    """Próba kompletna; Y, T i kontrole W oczyszczone z efektów stałych województw i lat (przekształcenie
    within), dzięki czemu Causal Forest uczy się z wariancji 'różnicy w różnicach', a nie z poziomów."""
    d = df.dropna(subset=[y, treat] + xcols + wcols).reset_index(drop=True)
    V = d[[y, treat] + wcols].to_numpy(float)
    if fe_demean:
        M = np.column_stack([pd.get_dummies(d["Kod_Str"], dtype=float).to_numpy(),
                             pd.get_dummies(d["Rok"], drop_first=True, dtype=float).to_numpy()])
        R = V - M @ np.linalg.lstsq(M, V, rcond=None)[0]
    else:
        R = V - V.mean(axis=0)
    d["Y_cf"], d["T_cf"] = R[:, 0], R[:, 1]
    for i in range(len(wcols)):
        d[f"W_cf_{i}"] = R[:, 2 + i]
    return d


def nuisance_diagnostics(d: pd.DataFrame, xcols: list, wcols: list, seed: int) -> dict:
    """R^2 poza próbą (cross-fitting grupowany po województwach) dla modeli Y~X,W oraz T~X,W."""
    cols = xcols + [f"W_cf_{i}" for i in range(len(wcols))]
    XW = d[cols].to_numpy(float)
    groups = pd.factorize(d["Kod_Str"])[0]
    cv = GroupKFold(n_splits=min(4, len(np.unique(groups))))
    out = {}
    for nm, tgt in (("Y", d["Y_cf"]), ("T", d["T_cf"])):
        pred = cross_val_predict(make_gbm(seed), XW, tgt.to_numpy(float), cv=cv, groups=groups)
        out[f"R² modelu {nm} (poza próbą)"] = float(r2_score(tgt, pred))
    out["SD treatmentu po oczyszczeniu z FE (p.p. Kaitza)"] = float(d["T_cf"].std())
    return out


def _first_leaf(obj):
    while isinstance(obj, dict):
        obj = next(iter(obj.values()))
    return np.asarray(getattr(obj, "values", obj))


def fit_causal_forest(d: pd.DataFrame, xcols: list, wcols: list, seed: int = 42, n_trees: int = 1000,
                      min_leaf: int = 5, try_native_shap: bool = False) -> dict:
    """Causal Forest DML (EconML). Cross-fitting grupowany po województwach (GroupKFold), nuisance = XGBoost."""
    if not HAS_ECONML:
        raise DataError("Brak biblioteki 'econml' – zainstaluj ją (requirements.txt), aby uruchomić Causal Forest.")
    n_trees = int(np.ceil(n_trees / 4) * 4)       # EconML wymaga wielokrotności subforest_size (=4)
    X = d[xcols].to_numpy(float)
    W = d[[f"W_cf_{i}" for i in range(len(wcols))]].to_numpy(float) if wcols else None
    Y, T = d["Y_cf"].to_numpy(float), d["T_cf"].to_numpy(float)
    groups = pd.factorize(d["Kod_Str"])[0]
    notes = []

    def build(cv):
        return CausalForestDML(model_y=make_gbm(seed), model_t=make_gbm(seed), discrete_treatment=False,
                               n_estimators=n_trees, min_samples_leaf=min_leaf, cv=cv, random_state=seed)

    try:
        cf = build(GroupKFold(n_splits=min(4, len(np.unique(groups)))))
        cf.fit(Y, T, X=X, W=W, groups=groups)
    except Exception as e:  # np. starsza wersja EconML bez parametru groups
        notes.append(f"Cross-fitting grupowany niedostępny ({type(e).__name__}: {str(e)[:90]}) – użyto zwykłego 4-fold CV.")
        cf = build(4)
        cf.fit(Y, T, X=X, W=W)
    tau = np.asarray(cf.effect(X)).reshape(-1)
    lo, hi = (np.asarray(v).reshape(-1) for v in cf.effect_interval(X, alpha=0.10))
    try:
        a_lo, a_hi = (float(np.squeeze(v)) for v in cf.ate_interval(X, alpha=0.05))
    except Exception:
        a_lo = a_hi = np.nan
    fi = None
    for getter in (lambda: cf.feature_importances_, lambda: cf.feature_importances()):
        try:
            fi = np.asarray(getter()).reshape(-1)
            break
        except Exception:
            continue
    native = None
    if try_native_shap:
        try:
            native = _first_leaf(cf.shap_values(X, feature_names=xcols))
        except Exception as e:
            notes.append(f"Natywne SHAP z EconML niedostępne ({type(e).__name__}).")
    return {"tau": tau, "lo": lo, "hi": hi, "ate": float(np.mean(tau)), "ate_lo": a_lo, "ate_hi": a_hi,
            "importances": fi, "native_shap": native, "notes": notes}


def shap_surrogate(X_df: pd.DataFrame, tau: np.ndarray, seed: int):
    """SHAP na modelu zastępczym: XGBoost uczony na oszacowanych CATE. Zwraca (wartości SHAP, baza, R² wierności)."""
    if not HAS_SHAP:
        raise DataError("Brak biblioteki 'shap'.")
    sur = make_gbm(seed, n_estimators=300)
    sur.fit(X_df, tau)
    fid = float(r2_score(tau, sur.predict(X_df)))
    explainer = shap.TreeExplainer(sur)
    sv = np.asarray(explainer.shap_values(X_df))
    base = float(np.ravel(explainer.expected_value)[0])
    return sv, base, fid


def cluster_boot_ci(vals, gids, B: int = 400, seed: int = 0):
    """Przedział 95% dla średniej z bootstrapu klastrów (losowanie województw ze zwracaniem)."""
    vals, gids = np.asarray(vals, float), np.asarray(gids)
    uniq = np.unique(gids)
    if len(uniq) < 3:
        return np.nan, np.nan
    by = {g: vals[gids == g] for g in uniq}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        means.append(np.concatenate([by[g] for g in pick]).mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def group_cate_table(d: pd.DataFrame, tau: np.ndarray, col: str, q: int = 3) -> pd.DataFrame:
    names = {3: ["niski", "średni", "wysoki"]}.get(q)
    bins = pd.qcut(d[col], q, duplicates="drop")
    tmp = pd.DataFrame({"bin": bins, "tau": tau, "g": d["Kod_Str"].to_numpy(), "x": d[col].to_numpy()})
    rows = []
    for i, (b, g) in enumerate(tmp.groupby("bin", observed=True)):
        lo, hi = cluster_boot_ci(g["tau"], g["g"], seed=i)
        rows.append({"grupa": names[i] if names and i < len(names) else str(b), "przedział moderatora": str(b),
                     "średnia wartość": g["x"].mean(), "N": len(g), "średni CATE": g["tau"].mean(),
                     "CI95 dół": lo, "CI95 góra": hi})
    return pd.DataFrame(rows)


def cate_projection(d: pd.DataFrame, tau: np.ndarray, xcols: list) -> pd.DataFrame:
    """Najlepsza liniowa projekcja CATE na standaryzowane moderatory (SE klastrowane po województwach)."""
    t = d[["Kod_Str"]].copy()
    t["tau"] = tau
    zc = []
    for c in xcols:
        t[f"z_{c}"] = (d[c] - d[c].mean()) / d[c].std(ddof=0)
        zc.append(f"z_{c}")
    r = fit_linear(t, "tau", zc, fe="pooled")
    out = r.tidy(keep=zc + ["const"])
    out["zmienna"] = [short_label(z[2:]) if z != "const" else "stała (średni CATE)" for z in out["zmienna"]]
    return out


def placebo_causal_forest(d: pd.DataFrame, xcols: list, wcols: list, B: int, seed: int, n_trees: int = 300):
    """Test permutacyjny: losowa zamiana treatmentu między województwami w obrębie roku."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(B):
        dd = d.copy()
        dd["T_cf"] = dd.groupby("Rok")["T_cf"].transform(lambda v: rng.permutation(v.to_numpy()))
        r = fit_causal_forest(dd, xcols, wcols, seed=seed + b + 1, n_trees=n_trees)
        rows.append({"b": b + 1, "ATE": r["ate"], "SD(CATE)": float(np.std(r["tau"]))})
    return pd.DataFrame(rows)


def run_causal_analysis(df: pd.DataFrame, cfg: dict) -> dict:
    """Pełna ścieżka Causal ML. cfg: treat, lag, fe_demean, n_trees, seed, min_leaf, native_shap."""
    xcols, wcols = causal_columns(cfg["lag"])
    d = prepare_causal_sample(df, cfg["treat"], xcols, wcols, cfg["fe_demean"])
    if len(d) < 60:
        raise DataError(f"Za mało kompletnych obserwacji do Causal Forest (N = {len(d)}).")
    out = {"xcols": xcols, "wcols": wcols, "n": len(d)}
    out["nuisance"] = nuisance_diagnostics(d, xcols, wcols, cfg["seed"])
    cf = fit_causal_forest(d, xcols, wcols, cfg["seed"], cfg["n_trees"], cfg["min_leaf"], cfg["native_shap"])
    d = d.copy()
    d["tau"], d["tau_lo"], d["tau_hi"] = cf["tau"], cf["lo"], cf["hi"]
    out.update({"sample": d, "ate": cf["ate"], "ate_lo": cf["ate_lo"], "ate_hi": cf["ate_hi"],
                "importances": cf["importances"], "native_shap": cf["native_shap"], "notes": list(cf["notes"])})
    if np.isnan(cf["ate_lo"]):
        out["ate_lo"], out["ate_hi"] = cluster_boot_ci(cf["tau"], d["Kod_Str"], seed=1)
        out["notes"].append("Przedział ATE: bootstrap klastrów po województwach (średnia z CATE; "
                            "pomija niepewność estymacji lasu).")
    out["groups"] = {c: group_cate_table(d, cf["tau"], c) for c in [k for k in xcols if k.split("_L1")[0] in KEYNES_COLS]}
    out["projection"] = cate_projection(d, cf["tau"], xcols)
    try:
        sv, base, fid = shap_surrogate(d[xcols], cf["tau"], cfg["seed"])
        out["shap"] = {"values": sv, "base": base, "fidelity": fid}
    except Exception as e:
        out["shap"] = None
        out["notes"].append(f"SHAP niedostępne: {type(e).__name__}: {str(e)[:120]}")
    return out


# =============================================================================
# 8. OBLICZENIA POD INTERFEJS (zwracają tylko DataFrame / słowniki / tablice – bezpieczne dla cache)
# =============================================================================
def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _fmt_models(models: dict, keep: list) -> pd.DataFrame:
    """Tabela w stylu publikacyjnym: współczynnik*** (SE) obok siebie dla kilku modeli."""
    rows = {}
    for name, m in models.items():
        t = m["tidy"].set_index("zmienna")
        for v in keep:
            if v in t.index:
                r = t.loc[v]
                rows.setdefault(v, {})[name] = f"{r['współczynnik']:.3f}{stars(r['p-value'])} ({r['SE (klaster)']:.3f})"
    tab = pd.DataFrame(rows).T.reindex(keep)
    tab.index = [label_of(v) if v not in ("Kaitz_pp", "Kaitz_Ekspozycja_pp", "const") else
                 {"Kaitz_pp": "Wskaźnik Kaitza (p.p.)", "Kaitz_Ekspozycja_pp": "Kaitz ekspozycyjny (p.p.)",
                  "const": "Stała"}[v] for v in tab.index]
    foot = {n: f"N={m['n']}, klastry={m['G']}, R²={m['r2']:.3f}" for n, m in models.items()}
    tab.loc["Statystyki"] = pd.Series(foot)
    return tab.fillna("")


@st.cache_data(show_spinner=False)
def compute_linear(panel: pd.DataFrame, lag: bool, treat: str, B: int, seed: int) -> dict:
    suf = "_L1" if lag else ""
    ctrl = [c + suf for c in KEYNES_COLS]
    y = "Stopa_Zatrudnienia"
    out = {"controls": ctrl, "treat": treat, "models": {}}
    for label, fe in (("(1) Pooled OLS", "pooled"), ("(2) FE województw", "entity"), ("(3) TWFE: woj. + lata", "twfe")):
        r = fit_linear(panel, y, [treat] + ctrl, fe=fe)
        out["models"][label] = {"tidy": r.tidy(), "n": r.n, "G": r.G, "r2": r.r2}
    base = fit_linear(panel, y, [treat] + ctrl, fe="twfe")
    b, se, p = base.coef(treat)
    out["main"] = {"b": b, "se": se, "p": p, "p_wild": wild_cluster_p(base, treat, B=B, seed=seed),
                   "ci": (float(base.lo[0]), float(base.hi[0])), "n": base.n, "G": base.G}
    ires, ints, (F, pF, q) = interaction_model(panel, y, treat, ctrl, [])
    out["interactions"] = {"tidy": ires.tidy(keep=[treat] + ints), "F": F, "p": pF, "q": q, "n": ires.n}
    out["event"], out["event_pre"] = event_study(panel)
    out["robust"], _ = robustness_linear(panel, ctrl, B=min(B, 499), seed=seed)
    out["loo"] = leave_one_region_out(panel, y, ["Kaitz_pp"] + ctrl)
    d_fe = prepare_causal_sample(panel, treat, [], [], True)
    out["fwl"] = d_fe[["Wojewodztwo", "Rok", "Y_cf", "T_cf"]]
    out["sm_text"] = None
    if HAS_SM:
        try:
            dd = panel.dropna(subset=[y, treat] + ctrl)
            Xs = sm.add_constant(dd[[treat] + ctrl])
            mod = sm.OLS(dd[y], Xs).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd["Kod_Str"])[0]})
            out["sm_text"] = str(mod.summary())
        except Exception as e:  # pragma: no cover
            out["sm_text"] = f"statsmodels: {type(e).__name__}: {e}"
    return out


@st.cache_data(show_spinner=False)
def compute_causal(panel: pd.DataFrame, cfg: dict) -> dict:
    return run_causal_analysis(panel, cfg)


@st.cache_data(show_spinner=False)
def compute_cf_robustness(panel: pd.DataFrame, cfg: dict, kind: str, B: int) -> dict:
    xcols, wcols = causal_columns(cfg["lag"])
    d = prepare_causal_sample(panel, cfg["treat"], xcols, wcols, cfg["fe_demean"])
    if kind == "seeds":
        rows, ref = [], None
        for s in (1, 2, 3, 4):
            r = fit_causal_forest(d, xcols, wcols, seed=s, n_trees=cfg["n_trees"], min_leaf=cfg["min_leaf"])
            ref = r["tau"] if ref is None else ref
            rows.append({"ziarno": s, "ATE": r["ate"], "SD(CATE)": float(np.std(r["tau"])),
                         "korelacja CATE z ziarnem 1": float(np.corrcoef(ref, r["tau"])[0, 1])})
        return {"table": pd.DataFrame(rows)}
    real = fit_causal_forest(d, xcols, wcols, seed=cfg["seed"], n_trees=300, min_leaf=cfg["min_leaf"])
    tab = placebo_causal_forest(d, xcols, wcols, B=B, seed=cfg["seed"], n_trees=300)
    sd_real = float(np.std(real["tau"]))
    return {"table": tab, "sd_real": sd_real, "ate_real": real["ate"],
            "p_sd": float((np.sum(tab["SD(CATE)"] >= sd_real) + 1) / (len(tab) + 1)),
            "p_ate": float((np.sum(np.abs(tab["ATE"]) >= abs(real["ate"])) + 1) / (len(tab) + 1))}


def export_excel(panel, lin, cres) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        panel.to_excel(xw, sheet_name="Panel", index=False)
        if lin:
            _fmt_models(lin["models"], [lin["treat"]] + lin["controls"] + ["const"]).to_excel(xw, sheet_name="Modele_liniowe")
            lin["interactions"]["tidy"].to_excel(xw, sheet_name="Interakcje", index=False)
            lin["event"].to_excel(xw, sheet_name="EventStudy", index=False)
            lin["robust"].to_excel(xw, sheet_name="Robustness", index=False)
            lin["loo"].to_excel(xw, sheet_name="LeaveOneOut", index=False)
        if cres:
            cols = ["Wojewodztwo", "Rok", "tau", "tau_lo", "tau_hi"] + cres["xcols"]
            cres["sample"][cols].to_excel(xw, sheet_name="CATE", index=False)
            cres["projection"].to_excel(xw, sheet_name="Projekcja_CATE", index=False)
    return buf.getvalue()


# =============================================================================
# 9. INTERFEJS STREAMLIT
# =============================================================================
def fig_show(fig):
    st.pyplot(fig)
    plt.close(fig)


def collect_sources(uploaded):
    src = discover_files()
    origin = {k: "folder" for k in src}
    for f in uploaded or []:
        key = prefix_of(f.name)
        if key in FILE_SPECS:
            src[key] = (f.name, f.getvalue())
            origin[key] = "Drag & Drop"
    return src, origin


@st.cache_data(show_spinner=False)
def load_panel(sources: dict):
    return build_panel(sources)


def variation_table(panel: pd.DataFrame, cols: list) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = panel[c]
        within = s - s.groupby(panel["Kod_Str"]).transform("mean")
        resid = within - within.groupby(panel["Rok"]).transform("mean")
        between = s.groupby(panel["Kod_Str"]).transform("mean")
        rows.append({"zmienna": c, "SD całkowite": s.std(), "SD między woj.": between.std(),
                     "SD wewnątrz woj.": within.std(), "SD po usunięciu FE woj. i lat": resid.std()})
    return pd.DataFrame(rows).set_index("zmienna")


def render_data(panel, diag, origin, sources):
    st.subheader("Oczyszczony Panel Wojewódzki (nazwy z kodów TERYT, bez 'chińskich znaczków')")
    for w in diag["warnings"]:
        st.warning(w)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Obserwacje", diag["panel"]["n"])
    c2.metric("Województwa", diag["panel"]["regions"])
    c3.metric("Lata", f"{diag['panel']['years'][0]}–{diag['panel']['years'][1]}")
    c4.metric("Panel zbilansowany", "tak" if diag["panel"]["balanced"] else "nie")
    show_cols = ["Wojewodztwo", "Rok", "Kaitz_Index", "Stopa_Zatrudnienia", "Wzrost_PKB", "Udzial_Przemyslu",
                 "Bezrobocie_BAEL"]
    st.dataframe(panel[show_cols].head(15).round(3), use_container_width=True)
    with st.expander("Definicje zmiennych i decyzje metodologiczne", expanded=False):
        for n in diag["notes"]:
            st.markdown(f"- {n}")
        st.markdown(
            "- **Kaitz_Index** = płaca minimalna (styczeń, słownik w kodzie) / przeciętne wynagrodzenie brutto w województwie "
            "(WYNA_2797). Oba człony nominalne – CPI skraca się w ilorazie; wersje realne (`Placa_Min_Realna`, "
            "`Wynagrodzenie_Realne`, ceny 2010) są w panelu.\n"
            "- **Kaitz_Ekspozycja_pp** = MW / (wynagrodzenie bazowe woj. z lat 2005–2009 × krajowy wzrost płac) – odporny na "
            "endogeniczność lokalnych płac.\n"
            "- **Wzrost_PKB** – realny wzrost PKB (wolumen, rachunki regionalne). **Udzial_Przemyslu** = (sekcje B+C+D+E) / WDB ogółem.\n"
            "- **Produktywnosc** – WDB na pracującego w cenach stałych 2010 (deflator: regionalny CPI).\n"
            "- Moderatory z opóźnieniem `_L1` ograniczają problem zmiennych pośredniczących (post-treatment).")
    st.markdown("**Rozkład wskaźnika Kaitza i jego zmienność** – identyfikacja pochodzi tylko z ostatniej kolumny:")
    st.dataframe(variation_table(panel, ["Kaitz_pp", "Kaitz_Ekspozycja_pp"]).round(3), use_container_width=True)
    ca, cb = st.columns(2)
    with ca:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(panel["Kaitz_Index"], bins=20, kde=True, ax=ax)
        ax.set_title("Rozkład Wskaźnika Kaitza w panelu badawczym")
        ax.set_xlabel("MW / przeciętne wynagrodzenie w województwie")
        fig_show(fig)
    with cb:
        fig, ax = plt.subplots(figsize=(6, 4))
        for k, g in panel.groupby("Kod_Str"):
            ax.plot(g["Rok"], g["Kaitz_Index"], color="lightgrey", lw=1)
        for k, col in (("14", "tab:red"), ("32", "tab:blue")):
            g = panel[panel["Kod_Str"] == k]
            ax.plot(g["Rok"], g["Kaitz_Index"], color=col, lw=2, label=WOJEWODZTWA_MAP[k])
        nat = panel.groupby("Rok")["Kaitz_Krajowy"].first()
        ax.plot(nat.index, nat.values, color="black", ls="--", lw=2, label="Polska (słownik w kodzie)")
        ax.set_title("Wskaźnik Kaitza: województwa i Polska")
        ax.legend(fontsize=8)
        fig_show(fig)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    hm = panel.pivot(index="Wojewodztwo", columns="Rok", values="Kaitz_pp")
    sns.heatmap(hm, cmap="viridis", ax=ax, cbar_kws={"label": "Kaitz (p.p.)"})
    ax.set_title("Wskaźnik Kaitza wg województw i lat")
    ax.set_ylabel("")
    fig_show(fig)
    with st.expander("Kontrole poprawności danych", expanded=False):
        if "wynagrodzenie_krajowe" in diag["checks"]:
            st.markdown("**Przeciętne wynagrodzenie krajowe: słownik w kodzie vs GUS (WYNA_2797, Polska)**")
            st.dataframe(diag["checks"]["wynagrodzenie_krajowe"].round(2), use_container_width=True)
        if "ludnosc" in diag["checks"]:
            st.markdown("**Ludność wyprowadzona z PKB / PKB per capita vs bilans ludności GUS (12 województw)**")
            lud = diag["checks"]["ludnosc"].pivot_table(index="Rok", columns="Wojewodztwo", values="Roznica_%")
            st.dataframe(lud.round(1), use_container_width=True)
        st.markdown("**Statystyki opisowe**")
        st.dataframe(panel[["Stopa_Zatrudnienia", "Kaitz_pp", "Bezrobocie_BAEL", "Wzrost_PKB", "Udzial_Przemyslu",
                            "Realny_PKB_per_capita", "Produktywnosc", "CPI_Region"]].describe().T.round(3),
                     use_container_width=True)
    with st.expander("Wykryte pliki wejściowe", expanded=False):
        rows = [{"prefiks": k, "opis": v[0], "status": v[1], "wykryty plik": sources[k][0] if k in sources else "—",
                 "źródło": origin.get(k, "—")} for k, v in FILE_SPECS.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_linear(lin, panel):
    if lin is None:
        st.info("Naciśnij 'Uruchom Pełną Analizę' w panelu bocznym.")
        return
    m = lin["main"]
    st.subheader("Model klasyczny (ekonometria): benchmark dla Causal ML")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TWFE: efekt 1 p.p. Kaitza na stopę zatrudnienia (p.p.)", f"{m['b']:.3f}")
    c2.metric("SE (klaster po woj.)", f"{m['se']:.3f}")
    c3.metric("p-value t(G−1)", f"{m['p']:.3f}")
    c4.metric("p-value, dzikie bootstrap klastrów", f"{m['p_wild']:.3f}")
    st.caption(f"Zmienna objaśniana: pracujący (BAEL) na 100 mieszkańców. Treatment: "
               f"{'Kaitz rzeczywisty' if lin['treat'] == 'Kaitz_pp' else 'Kaitz ekspozycyjny'} w p.p. "
               f"N={m['n']}, klastry (województwa)={m['G']}, CI95 = [{m['ci'][0]:.2f}; {m['ci'][1]:.2f}].")
    keep = [lin["treat"]] + lin["controls"] + ["const"]
    st.markdown("**Porównanie specyfikacji** – współczynnik (SE klastrowany); * p<0,10, ** p<0,05, *** p<0,01")
    st.dataframe(_fmt_models(lin["models"], keep), use_container_width=True)
    st.markdown("**Heterogeniczność w modelu liniowym:** Kaitz × (moderator − średnia); współczynnik przy Kaitzu = efekt "
                "przy przeciętnych wartościach moderatorów.")
    it = lin["interactions"]
    st.dataframe(it["tidy"].round(4), use_container_width=True)
    st.metric(f"Łączny test Walda dla interakcji (F({it['q']}, G−1))", f"{it['F']:.2f}", f"p = {it['p']:.3f}", delta_color="off")
    st.markdown("**Event study** – efekt standaryzowanego (przedsamplowego) 'bite' regionu w kolejnych latach; "
                "rok odniesienia 2015; lata do 2014 to test trendów wstępnych.")
    ev = lin["event"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ref = 2015
    xs = list(ev["Rok"]) + [ref]
    ys = list(ev["współczynnik"]) + [0.0]
    ax.errorbar(ev["Rok"], ev["współczynnik"], yerr=[ev["współczynnik"] - ev["CI95 dół"], ev["CI95 góra"] - ev["współczynnik"]],
                fmt="o", color="tab:blue", capsize=3)
    ax.scatter([ref], [0.0], color="black", zorder=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(ref + 0.5, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Rok")
    ax.set_ylabel("p.p. stopy zatrudnienia na 1 SD 'bite'")
    ax.set_title("Event study: województwa o niskich płacach bazowych vs reszta")
    fig_show(fig)
    if lin["event_pre"]:
        F, p, q = lin["event_pre"]
        (st.error if p < 0.05 else st.success)(
            f"Test trendów wstępnych (łączna istotność {q} współczynników sprzed 2015): F = {F:.2f}, p = {p:.3f}. "
            + ("Odrzucamy równoległe trendy – interpretacja przyczynowa wymaga ostrożności." if p < 0.05
               else "Brak podstaw do odrzucenia równoległych trendów."))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fw = lin["fwl"]
    ax.scatter(fw["T_cf"], fw["Y_cf"], s=14, alpha=0.5)
    slope = np.polyfit(fw["T_cf"], fw["Y_cf"], 1)
    xx = np.linspace(fw["T_cf"].min(), fw["T_cf"].max(), 50)
    ax.plot(xx, np.polyval(slope, xx), color="tab:red", lw=2, label=f"nachylenie = {slope[0]:.2f}")
    ax.set_xlabel("Kaitz po usunięciu FE woj. i lat (p.p.)")
    ax.set_ylabel("Stopa zatrudnienia po usunięciu FE (p.p.)")
    ax.set_title("Wykres reszt częściowych (Frisch–Waugh–Lovell)")
    ax.legend()
    fig_show(fig)
    if lin["sm_text"]:
        with st.expander("Pełny raport statsmodels (pooled OLS, SE klastrowane)"):
            st.text(lin["sm_text"])


def render_causal(cres, lin, cfg):
    if cres is None:
        st.info("Naciśnij 'Uruchom Pełną Analizę' w panelu bocznym.")
        return
    st.subheader("Causal Forest & Double ML (nieliniowość, XGBoost jako nuisance)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Causal Forest ATE (p.p. zatrudnienia na 1 p.p. Kaitza)", f"{cres['ate']:.3f}")
    c2.metric("CI95 dla ATE", f"[{cres['ate_lo']:.2f}; {cres['ate_hi']:.2f}]")
    if lin:
        c3.metric("TWFE (benchmark)", f"{lin['main']['b']:.3f}")
    for n in cres["notes"]:
        st.caption("ℹ️ " + n)
    with st.expander("Diagnostyka modeli nuisance i identyfikacji", expanded=True):
        st.dataframe(pd.DataFrame({k: [v] for k, v in cres["nuisance"].items()}).T.rename(columns={0: "wartość"}).round(3),
                     use_container_width=True)
        st.caption(f"N = {cres['n']} obserwacji. Niskie R² modelu T oznacza, że po usunięciu FE treatment jest niemal "
                   "niezależny od moderatorów (dobry znak dla nieobciążoności, ale i mała zmienność do uczenia heterogeniczności).")
    d = cres["sample"]
    xcols = cres["xcols"]
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Efekty regionalne (średni CATE na województwo)**")
        g = d.groupby("Wojewodztwo")[["tau", "tau_lo", "tau_hi"]].mean().sort_values("tau")
        fig, ax = plt.subplots(figsize=(6, 7))
        ax.barh(g.index, g["tau"], color=plt.cm.viridis(np.linspace(0.1, 0.9, len(g))),
                xerr=[g["tau"] - g["tau_lo"], g["tau_hi"] - g["tau"]], error_kw={"alpha": 0.4, "capsize": 2})
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Średni estymowany efekt zatrudnieniowy (pasek: średni przedział 90% obserwacji)")
        fig_show(fig)
    with col2:
        st.write("**Nieliniowość efektu (Kaitz vs CATE)**")
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(d["Kaitz_pp"], d["tau"], c=d[xcols[0]], cmap="coolwarm", s=22)
        plt.colorbar(sc, ax=ax, label=short_label(xcols[0]))
        ax.axhline(0, color="black", ls="--")
        ax.set_xlabel("Wskaźnik Kaitza (p.p.)")
        ax.set_ylabel("CATE")
        ax.set_title("Wykryte progi wrażliwości")
        fig_show(fig)
    st.markdown("### Heterogeniczność keynesowska: CATE a popyt, rezerwy pracy i struktura sektorowa")
    keyn = [c for c in xcols if c.split("_L1")[0] in KEYNES_COLS]
    fig, axes = plt.subplots(1, len(keyn), figsize=(5 * len(keyn), 4), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), keyn):
        ax.scatter(d[c], d["tau"], s=14, alpha=0.4)
        b = pd.qcut(d[c], 6, duplicates="drop")
        mm = d.groupby(b, observed=True).agg(x=(c, "mean"), y=("tau", "mean"))
        ax.plot(mm["x"], mm["y"], color="tab:red", marker="o", lw=2)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel(short_label(c))
    np.atleast_1d(axes)[0].set_ylabel("CATE")
    fig.tight_layout()
    fig_show(fig)
    cols = st.columns(len(cres["groups"]))
    for cc, (c, tab) in zip(cols, cres["groups"].items()):
        with cc:
            st.write(f"**{short_label(c)}** – terciele")
            st.dataframe(tab[["grupa", "średnia wartość", "N", "średni CATE", "CI95 dół", "CI95 góra"]].round(3),
                         use_container_width=True)
    st.markdown("**Liniowa projekcja CATE na standaryzowane moderatory** (zmiana CATE przy +1 SD moderatora; SE klastrowane):")
    st.dataframe(cres["projection"].round(4), use_container_width=True)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    by = d.groupby("Rok")["tau"].mean()
    ax.plot(by.index, by.values, marker="o")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Średni CATE w czasie")
    ax.set_xlabel("Rok")
    fig_show(fig)


def render_shap(cres):
    if cres is None:
        st.info("Naciśnij 'Uruchom Pełną Analizę' w panelu bocznym.")
        return
    st.subheader("Interpretacja SHAP (wpływ cech gospodarki na efekt płacy minimalnej)")
    st.markdown("Wartości SHAP liczone na **modelu zastępczym XGBoost wytrenowanym na oszacowanych CATE** (TreeExplainer). "
                "Wartość SHAP **w lewo (ujemna)** = cecha czyni efekt bardziej ujemnym, czyli silniej **niszczy zatrudnienie**; "
                "czerwony kolor = wysoka wartość cechy.")
    sh = cres["shap"]
    if sh is None:
        st.warning("SHAP niedostępne w tym środowisku (zob. uwagi w zakładce Causal ML).")
        return
    X_df = cres["sample"][cres["xcols"]].rename(columns=short_label)
    sv = sh["values"]
    st.caption(f"Wierność modelu zastępczego (R² względem CATE, w próbie): {sh['fidelity']:.3f}. "
               "Przy niskiej wierności interpretacja SHAP jest niepewna.")
    c1, c2 = st.columns(2)
    with c1:
        imp = pd.Series(np.abs(sv).mean(axis=0), index=X_df.columns).sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(imp.index, imp.values, color="tab:blue")
        ax.set_xlabel("średnia |SHAP| (wpływ na CATE)")
        ax.set_title("Ważność cech heterogeniczności")
        fig_show(fig)
    with c2:
        try:
            plt.figure(figsize=(6.5, 4.4))
            shap.summary_plot(sv, X_df, show=False)
            fig_show(plt.gcf())
        except Exception as e:  # pragma: no cover
            st.caption(f"Wykres beeswarm niedostępny: {type(e).__name__}")
    st.markdown("**Wykresy zależności SHAP dla trzech wymiarów keynesowskich**")
    keyn = [c for c in cres["xcols"] if c.split("_L1")[0] in KEYNES_COLS]
    fig, axes = plt.subplots(1, len(keyn), figsize=(5 * len(keyn), 3.8), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), keyn):
        j = cres["xcols"].index(c)
        ax.scatter(cres["sample"][c], sv[:, j], s=14, alpha=0.6)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel(short_label(c))
    np.atleast_1d(axes)[0].set_ylabel("SHAP (wpływ na CATE)")
    fig.tight_layout()
    fig_show(fig)
    if cres.get("importances") is not None and len(cres["importances"]) == len(cres["xcols"]):
        st.markdown("**Ważność cech w lesie przyczynowym (udział w podziałach)**")
        fi = pd.Series(cres["importances"], index=[short_label(c) for c in cres["xcols"]]).sort_values(ascending=False)
        st.dataframe(fi.rename("ważność").round(3), use_container_width=True)
    if cres.get("native_shap") is not None:
        st.markdown("**Natywne SHAP z EconML**")
        try:
            plt.figure(figsize=(7, 4.4))
            shap.summary_plot(cres["native_shap"], X_df, show=False)
            fig_show(plt.gcf())
        except Exception as e:  # pragma: no cover
            st.caption(f"Nie udało się narysować natywnych SHAP: {type(e).__name__}")


def render_robust(lin, panel, cfg):
    if lin is None:
        st.info("Naciśnij 'Uruchom Pełną Analizę' w panelu bocznym.")
        return
    st.subheader("Testy odporności (model TWFE dla Kaitza rzeczywistego)")
    rob = lin["robust"].copy()
    st.dataframe(rob.round(4), use_container_width=True)
    st.caption("Wynik TWFE-IV jest informatywny tylko przy F pierwszego stopnia ≥ 10 (reguła kciuka). "
               "Test 'lead' – istotny współczynnik przy Kaitzu z t+1 sugeruje naruszenie ścisłej egzogeniczności.")
    loo = lin["loo"]
    fig, ax = plt.subplots(figsize=(8, 5))
    loo = loo.sort_values("współczynnik")
    ax.errorbar(loo["współczynnik"], loo["pominięte województwo"], xerr=1.96 * loo["SE"], fmt="o", capsize=3)
    ax.axvline(lin["main"]["b"], color="tab:red", ls="--", label="pełna próba")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Współczynnik TWFE przy pominięciu województwa")
    ax.legend()
    ax.set_title("Leave-one-region-out")
    fig_show(fig)
    st.markdown("### Odporność Causal Forest")
    c1, c2 = st.columns(2)
    if not HAS_ECONML:
        st.warning("Brak biblioteki econml – testy Causal Forest niedostępne.")
        return
    with c1:
        if st.button("Stabilność względem ziarna losowego"):
            with st.spinner("Estymacja lasów dla 4 ziaren…"):
                st.session_state["cf_seeds"] = compute_cf_robustness(panel, cfg, "seeds", 0)
        if "cf_seeds" in st.session_state:
            st.dataframe(st.session_state["cf_seeds"]["table"].round(4), use_container_width=True)
    with c2:
        Bp = st.number_input("Liczba permutacji placebo", 5, 100, 15, step=5)
        if st.button("Test placebo (permutacja treatmentu)"):
            with st.spinner("Estymacja lasów placebo…"):
                st.session_state["cf_placebo"] = compute_cf_robustness(panel, cfg, "placebo", int(Bp))
        if "cf_placebo" in st.session_state:
            pl = st.session_state["cf_placebo"]
            st.metric("p-value: SD(CATE) rzeczywiste vs placebo", f"{pl['p_sd']:.3f}")
            st.metric("p-value: |ATE| rzeczywiste vs placebo", f"{pl['p_ate']:.3f}")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(pl["table"]["SD(CATE)"], bins=12, color="lightgrey")
            ax.axvline(pl["sd_real"], color="tab:red", lw=2, label="rzeczywiste")
            ax.set_xlabel("SD(CATE)")
            ax.legend()
            fig_show(fig)


def render_export(panel, lin, cres):
    st.subheader("Eksport wyników")
    st.download_button("Panel (CSV)", panel.to_csv(index=False).encode("utf-8-sig"), "panel_wojewodztw.csv", "text/csv")
    if lin or cres:
        st.download_button("Wyniki (Excel)", export_excel(panel, lin, cres), "wyniki_magisterka.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if lin:
        m = lin["main"]
        st.markdown("**Szkic zdania do rozdziału empirycznego:**")
        st.code(
            f"W specyfikacji z efektami stałymi województw i lat wzrost wskaźnika Kaitza o 1 p.p. wiąże się ze zmianą stopy "
            f"zatrudnienia o {m['b']:.2f} p.p. (SE klastrowany = {m['se']:.2f}; p = {m['p']:.3f}; p z dzikiego bootstrapu "
            f"klastrów = {m['p_wild']:.3f}; N = {m['n']}, {m['G']} województw).", language="text")


def main():
    st.set_page_config(page_title="Magisterka - Płaca Minimalna", layout="wide")
    st.title("Heterogeniczny wpływ płacy minimalnej na zatrudnienie")
    st.markdown("Analiza przyczynowa: **DiD Baseline -> Causal Forest -> XGBoost -> SHAP**")

    st.sidebar.header("Panel Sterowania")
    uploaded = st.sidebar.file_uploader("Panel awaryjny: przeciągnij pliki GUS (xlsx / csv)", type=["xlsx", "csv"],
                                        accept_multiple_files=True)
    sources, origin = collect_sources(uploaded)
    miss = [k for k in REQUIRED_FILES if k not in sources]
    st.sidebar.caption(f"Wykryte pliki: {len(sources)} · brakuje wymaganych: {len(miss)}" + (f" ({', '.join(miss)})" if miss else ""))

    with st.sidebar.expander("Ustawienia modeli", expanded=False):
        lag = st.checkbox("Moderatory opóźnione o rok (zalecane)", value=True)
        treat_lab = st.selectbox("Treatment", ["Kaitz rzeczywisty (regionalny)", "Kaitz ekspozycyjny (przedsamplowy)"])
        fe_demean = st.checkbox("Oczyść Y i T z FE woj. i lat przed DML", value=True)
        n_trees = st.select_slider("Liczba drzew w lesie", options=[200, 500, 1000, 2000], value=1000)
        min_leaf = st.slider("Minimalna liczebność liścia", 3, 15, 5)
        B = st.select_slider("Replikacje bootstrapu klastrów", options=[199, 499, 999, 1999], value=499)
        seed = st.number_input("Ziarno losowe", 0, 9999, 42)
        native = st.checkbox("Spróbuj także natywnych SHAP z EconML", value=False)
    treat = "Kaitz_pp" if treat_lab.startswith("Kaitz rzeczywisty") else "Kaitz_Ekspozycja_pp"
    cfg = {"treat": treat, "lag": bool(lag), "fe_demean": bool(fe_demean), "n_trees": int(n_trees),
           "min_leaf": int(min_leaf), "seed": int(seed), "native_shap": bool(native)}
    run_analysis = st.sidebar.button("Uruchom Pełną Analizę")

    try:
        panel, diag = load_panel(sources)
    except DataError as e:
        st.error(f"Nie udało się zbudować panelu: {e}")
        st.info("Wgraj brakujące pliki w panelu bocznym (Drag & Drop) lub umieść je w folderze z aplikacją.")
        st.stop()
    except Exception as e:  # pragma: no cover
        st.error(f"Błąd przetwarzania plików: {type(e).__name__}: {e}")
        st.stop()
    if panel[treat].notna().sum() == 0:
        st.error("Wybrany treatment jest niedostępny (brak pliku WYNA_2797).")
        st.stop()

    if not (HAS_ECONML and HAS_XGB and HAS_SHAP):
        lacking = [n for n, f in (("econml", HAS_ECONML), ("xgboost", HAS_XGB), ("shap", HAS_SHAP)) if not f]
        st.sidebar.warning("Brak bibliotek: " + ", ".join(lacking) + ". Część Causal ML będzie niedostępna.")

    if run_analysis:
        with st.spinner("Przetwarzanie danych i trenowanie modeli (OLS/TWFE → Causal Forest → SHAP)…"):
            st.session_state["lin"] = compute_linear(panel, cfg["lag"], treat, int(B), cfg["seed"])
            st.session_state["cres"], st.session_state["cres_err"] = None, None
            try:
                st.session_state["cres"] = compute_causal(panel, cfg)
            except Exception as e:
                st.session_state["cres_err"] = f"{type(e).__name__}: {e}"
            st.session_state["cfg"] = cfg
            for k in ("cf_seeds", "cf_placebo"):
                st.session_state.pop(k, None)
    lin, cres = st.session_state.get("lin"), st.session_state.get("cres")
    if st.session_state.get("cfg") not in (None, cfg):
        st.sidebar.warning("Zmieniono ustawienia – naciśnij 'Uruchom Pełną Analizę', aby odświeżyć wyniki.")
    if st.session_state.get("cres_err"):
        st.error("Część Causal ML nie została wykonana: " + st.session_state["cres_err"])

    tabs = st.tabs(["📊 Dane & EDA", "📈 Modele Liniowe (Baseline)", "🌲 Causal Machine Learning",
                    "🔍 Heterogeniczność (SHAP)", "🛡️ Testy odporności", "💾 Eksport"])
    with tabs[0]:
        render_data(panel, diag, origin, sources)
    with tabs[1]:
        render_linear(lin, panel)
    with tabs[2]:
        render_causal(cres, lin, st.session_state.get("cfg", cfg))
    with tabs[3]:
        render_shap(cres)
    with tabs[4]:
        render_robust(lin, panel, st.session_state.get("cfg", cfg))
    with tabs[5]:
        render_export(panel, lin, cres)
    if lin is None:
        st.info("Naciśnij przycisk po lewej stronie, aby rozpocząć proces Causal ML.")


if __name__ == "__main__":
    main()
