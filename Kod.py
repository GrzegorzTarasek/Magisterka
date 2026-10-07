import streamlit as st
import pandas as pd
import numpy as np
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt

# 1. Konfiguracja aplikacji Streamlit
st.set_page_config(page_title="Magisterka - Płaca Minimalna", layout="wide")
st.title("Heterogeniczny wpływ płacy minimalnej na zatrudnienie")
st.markdown("Analiza z wykorzystaniem **Causal Machine Learning (Double ML, Causal Forest)** w ujęciu Keynesowskim.")

# 2. Funkcja przygotowująca i czyszcząca dane
@st.cache_data
def prepare_keynesian_panel(wyna_df, ryne_df, rach_df, ceny_df, min_wage_dict):
    """
    Funkcja łączy surowe dane GUS i wylicza wskaźniki makroekonomiczne.
    Upewnij się, że Twoje pliki mają spójne nazwy kolumn ('Kod' dla województwa, 'Rok' dla lat).
    """
    try:
        # Poniżej zakładamy standardowe nazwy kolumn. Jeśli GUS użył innych nazw, 
        # konieczna będzie zmiana np. 'Rok' na 'Year' lub 'Przecietne_Wynagrodzenie_Nominalne' na 'Wartosc'
        
        df = wyna_df.merge(ryne_df, on=['Kod', 'Rok'], suffixes=('_wyna', '_ryne'))
        df = df.merge(rach_df, on=['Kod', 'Rok'])
        df = df.merge(ceny_df, on=['Rok']) 
        
        # Deflowanie i wskaźniki
        df['Realne_Wynagrodzenie'] = df['Przecietne_Wynagrodzenie_Nominalne'] / (df['Wskaznik_CPI'] / 100)
        df['Realny_PKB'] = df['PKB_Nominalny'] / (df['Wskaznik_CPI'] / 100)
        
        # Zmienna Treatment - Wskaźnik Kaitza
        df['Placa_Minimalna_Ustawowa'] = df['Rok'].map(min_wage_dict)
        df['Kaitz_Index'] = df['Placa_Minimalna_Ustawowa'] / df['Realne_Wynagrodzenie']
        
        # Zmienne makroekonomiczne
        df['Stopa_Bezrobocia'] = df['Bezrobocie_Rejestrowane'] 
        df['Wzrost_PKB'] = df.groupby('Kod')['Realny_PKB'].pct_change() * 100
        df['Udzial_Przemyslu'] = df['WDB_Przemysl'] / df['WDB_Ogolem']
        
        # Odrzucenie braków danych (np. pierwszy rok przy obliczaniu wzrostu PKB wygeneruje NaN)
        return df.dropna()
    except KeyError as e:
        st.error(f"Błąd nazw kolumn: Brakuje kolumny {e} w plikach GUS. Sprawdź nazwy kolumn w plikach Excel.")
        return pd.DataFrame()

# 3. Modelowanie przyczynowe
@st.cache_resource
def run_causal_analysis(df):
    """Estymacja heterogenicznych efektów przyczynowych za pomocą Causal Forest."""
    Y = df['Stopa_Zatrudnienia'] 
    T = df['Kaitz_Index'] 
    
    # Cechy heterogeniczności
    X_cols = ['Stopa_Bezrobocia', 'Wzrost_PKB', 'Realny_PKB_per_capita', 'Udzial_Przemyslu']
    X = df[X_cols]
    
    # Konfundery (np. wielkość populacji, inflacja)
    W = df[['Populacja', 'Wskaznik_CPI']] 

    # Model Double ML
    causal_forest = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=5),
        model_t=RandomForestRegressor(n_estimators=100, max_depth=5),
        discrete_treatment=False,
        n_estimators=500,
        random_state=42
    )
    
    causal_forest.fit(Y, T, X=X, W=W)
    cate_estimates = causal_forest.effect(X)
    df['Estimated_Employment_Effect'] = cate_estimates
    ate = np.mean(cate_estimates)
    shap_values = causal_forest.shap_values(X)
    
    return df, causal_forest, ate, shap_values, X

# 4. Interfejs użytkownika
st.sidebar.header("Ustawienia panelu")
if st.sidebar.button("Załaduj dane z GUS i trenuj model"):
    with st.spinner("Wczytywanie plików Excel z repozytorium..."):
        try:
            # Wczytywanie plików rzeczywistych. Upewnij się, że są one umieszczone w repozytorium GitHub
            wyna_df = pd.read_excel("WYNA_2504.xlsx")
            ryne_df = pd.read_excel("RYNE_4098.xlsx")
            rach_df = pd.read_excel("RACH_3510.xlsx")
            ceny_df = pd.read_excel("CENY_2496_XTAB_20261007223438.xlsx")
            
            # Historia płacy minimalnej w Polsce (brutto) dla lat, które obejmują Twoje dane
            # Możesz uzupełnić lub skorygować ten słownik w zależności od zakresu dat w plikach
            min_wage_history = {
                2010: 1317, 2011: 1386, 2012: 1500, 2013: 1600, 2014: 1680,
                2015: 1750, 2016: 1850, 2017: 2000, 2018: 2100, 2019: 2250, 
                2020: 2600, 2021: 2800, 2022: 3010, 2023: 3600
            }
            
            st.info("Dane pomyślnie wczytane. Rozpoczynam przygotowanie panelu...")
            panel_df = prepare_keynesian_panel(wyna_df, ryne_df, rach_df, ceny_df, min_wage_history)
            
            if not panel_df.empty:
                st.info(f"Panel gotowy. Liczba obserwacji po wyczyszczeniu: {len(panel_df)}. Trenowanie modelu...")
                results_df, model, ate, shap_values, X = run_causal_analysis(panel_df)
                
                # Wyświetlanie wyników
                st.success("Model Causal Forest przetrenowany pomyślnie!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Średni Efekt na Zatrudnienie (ATE)", value=f"{ate:.4f}")
                    st.caption("Ujemna wartość oznacza średni spadek stopy zatrudnienia na skutek wzrostu wskaźnika Kaitza.")
                
                st.subheader("Próbka złączonego panelu województw")
                st.
