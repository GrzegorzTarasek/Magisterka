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

# 2. Funkcje przetwarzające (cachowane, aby aplikacja nie liczyła tego przy każdym kliknięciu)
@st.cache_data
def prepare_keynesian_panel(wyna_df, ryne_df, rach_df, ceny_df, min_wage_dict):
    df = wyna_df.merge(ryne_df, on=['Kod', 'Rok'], suffixes=('_wyna', '_ryne'))
    df = df.merge(rach_df, on=['Kod', 'Rok'])
    df = df.merge(ceny_df, on=['Rok']) 
    
    df['Realne_Wynagrodzenie'] = df['Przecietne_Wynagrodzenie_Nominalne'] / (df['Wskaznik_CPI'] / 100)
    df['Realny_PKB'] = df['PKB_Nominalny'] / (df['Wskaznik_CPI'] / 100)
    
    df['Placa_Minimalna_Ustawowa'] = df['Rok'].map(min_wage_dict)
    df['Kaitz_Index'] = df['Placa_Minimalna_Ustawowa'] / df['Realne_Wynagrodzenie']
    
    df['Stopa_Bezrobocia'] = df['Bezrobocie_Rejestrowane'] 
    df['Wzrost_PKB'] = df.groupby('Kod')['Realny_PKB'].pct_change() * 100
    df['Udzial_Przemyslu'] = df['WDB_Przemysl'] / df['WDB_Ogolem']
    
    return df.dropna()

@st.cache_resource
def run_causal_analysis(df):
    Y = df['Stopa_Zatrudnienia'] 
    T = df['Kaitz_Index'] 
    X_cols = ['Stopa_Bezrobocia', 'Wzrost_PKB', 'Realny_PKB_per_capita', 'Udzial_Przemyslu']
    X = df[X_cols]
    W = df[['Populacja', 'Wskaznik_CPI']] 

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

# 3. Interfejs użytkownika
st.sidebar.header("Ustawienia panelu")
if st.sidebar.button("Załaduj dane i trenuj model"):
    with st.spinner("Trwa wczytywanie plików i trenowanie modelu Causal Forest..."):
        
        # TUTAJ PODMIEŃ NA RZECZYWISTE ŚCIEŻKI DO SWOICH PLIKÓW Z REPOZYTORIUM
        # wyna_df = pd.read_excel("WYNA_2504.xlsx")
        # ryne_df = pd.read_excel("RYNE_4098.xlsx")
        # rach_df = pd.read_excel("RACH_3510.xlsx")
        # ceny_df = pd.read_excel("CENY_2496_XTAB_20261007223438.xlsx")
        
        # ZAMIAST TEGO TWORZĘ MOCKUP DO TESTÓW (Usuń to i odkomentuj linie wyżej):
        dates = range(2015, 2023)
        mock_data = pd.DataFrame({'Kod': [1]*8, 'Rok': dates, 'Przecietne_Wynagrodzenie_Nominalne': np.random.randint(3000, 6000, 8), 'Wskaznik_CPI': np.random.uniform(98, 115, 8), 'PKB_Nominalny': np.random.randint(10000, 20000, 8), 'Bezrobocie_Rejestrowane': np.random.uniform(3, 10, 8), 'WDB_Przemysl': np.random.uniform(1000, 5000, 8), 'WDB_Ogolem': np.random.uniform(8000, 15000, 8), 'Stopa_Zatrudnienia': np.random.uniform(50, 75, 8), 'Realny_PKB_per_capita': np.random.uniform(40000, 80000, 8), 'Populacja': np.random.randint(1000000, 5000000, 8)})
        wyna_df = mock_data[['Kod', 'Rok', 'Przecietne_Wynagrodzenie_Nominalne']]
        ryne_df = mock_data[['Kod', 'Rok', 'Bezrobocie_Rejestrowane', 'Stopa_Zatrudnienia', 'Populacja']]
        rach_df = mock_data[['Kod', 'Rok', 'PKB_Nominalny', 'WDB_Przemysl', 'WDB_Ogolem', 'Realny_PKB_per_capita']]
        ceny_df = mock_data[['Rok', 'Wskaznik_CPI']].drop_duplicates()
        
        min_wage_history = {2015: 1750, 2016: 1850, 2017: 2000, 2018: 2100, 2019: 2250, 2020: 2600, 2021: 2800, 2022: 3010}
        
        # Wywołanie funkcji
        try:
            panel_df = prepare_keynesian_panel(wyna_df, ryne_df, rach_df, ceny_df, min_wage_history)
            results_df, model, ate, shap_values, X = run_causal_analysis(panel_df)
            
            # Wyrzucenie wyników na ekran
            st.success("Model przetrenowany pomyślnie!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Średni Efekt na Zatrudnienie (ATE)", value=f"{ate:.4f}")
                st.caption("Wynik estymacji Double ML. Ujemna wartość oznacza średni spadek stopy zatrudnienia.")
            
            st.subheader("Próbka złączonego panelu danych")
            st.dataframe(results_df[['Rok', 'Kaitz_Index', 'Stopa_Bezrobocia', 'Estimated_Employment_Effect']].head())

            st.subheader("Wartości SHAP - Zależności Heterogeniczne (Keynesowskie)")
            st.markdown("Poniższy wykres pokazuje, które zmienne (np. bezrobocie, struktura przemysłu) najmocniej wpływają na siłę oddziaływania płacy minimalnej.")
            
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.summary_plot(shap_values['Stopa_Zatrudnienia']['Kaitz_Index'], X, show=False)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Błąd podczas analizy: {e}")
else:
    st.info("Naciśnij przycisk w panelu bocznym, aby załadować pliki i rozpocząć estymację.")
