import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import shap
import statsmodels.api as sm

st.set_page_config(page_title="Magisterka - Płaca Minimalna", layout="wide")
st.title("Heterogeniczny wpływ płacy minimalnej na zatrudnienie")
st.markdown("Analiza przyczynowa: **DiD Baseline -> Causal Forest -> XGBoost -> SHAP**")

# 1. Twarde dane zaszyte w kodzie (na podstawie przesłanych materiałów)
PRZECIETNE_WYNAGRODZENIE_PL = {
    1999: 1706.74, 2000: 1923.81, 2001: 2061.85, 2002: 2133.21, 2003: 2201.47, 
    2004: 2289.57, 2005: 2380.29, 2006: 2477.23, 2007: 2691.03, 2008: 2943.88, 
    2009: 3102.96, 2010: 3224.98, 2011: 3399.52, 2012: 3521.67, 2013: 3650.06, 
    2014: 3783.46, 2015: 3899.78, 2016: 4047.21, 2017: 4271.51, 2018: 4585.03, 
    2019: 4918.17, 2020: 5167.47, 2021: 5662.53, 2022: 6346.15, 2023: 7155.48, 
    2024: 8181.72, 2025: 8903.56
}

MIN_WAGE_PL = {
    2010: 1317, 2011: 1386, 2012: 1500, 2013: 1600, 2014: 1680,
    2015: 1750, 2016: 1850, 2017: 2000, 2018: 2100, 2019: 2250,
    2020: 2600, 2021: 2800, 2022: 3010, 2023: 3490, 2024: 4242, 2025: 4666
}

WOJEWODZTWA_MAP = {
    '02': 'Dolnośląskie', '04': 'Kujawsko-Pomorskie', '06': 'Lubelskie',
    '08': 'Lubuskie', '10': 'Łódzkie', '12': 'Małopolskie',
    '14': 'Mazowieckie', '16': 'Opolskie', '18': 'Podkarpackie',
    '20': 'Podlaskie', '22': 'Pomorskie', '24': 'Śląskie',
    '26': 'Świętokrzyskie', '28': 'Warmińsko-Mazurskie',
    '30': 'Wielkopolskie', '32': 'Zachodniopomorskie'
}

@st.cache_data
def load_and_clean_data(ryne_path, rach_path, ceny_path):
    try:
        # Wczytywanie z próbą naprawy kodowania (na wypadek CSV z polskimi znakami)
        ryne_df = pd.read_excel(ryne_path) if ryne_path.endswith('.xlsx') else pd.read_csv(ryne_path, encoding='cp1250', sep=';')
        rach_df = pd.read_excel(rach_path) if rach_path.endswith('.xlsx') else pd.read_csv(rach_path, encoding='cp1250', sep=';')
        ceny_df = pd.read_excel(ceny_path) if ceny_path.endswith('.xlsx') else pd.read_csv(ceny_path, encoding='cp1250', sep=';')
        
        # Łączenie danych regionalnych
        df = ryne_df.merge(rach_df, on=['Kod', 'Rok'], how='inner')
        df = df.merge(ceny_df, on=['Rok'], how='inner')

        # Naprawa nazw województw (omijamy chińskie znaki z plików)
        df['Kod_Str'] = df['Kod'].astype(str).str.zfill(7).str[:2]
        df['Wojewodztwo'] = df['Kod_Str'].map(WOJEWODZTWA_MAP)

        # Doklejanie zaszytych danych o wynagrodzeniach
        df['Przecietne_Wynagrodzenie_Kraj'] = df['Rok'].map(PRZECIETNE_WYNAGRODZENIE_PL)
        df['Placa_Minimalna'] = df['Rok'].map(MIN_WAGE_PL)
        
        # Deflowanie i wskaźniki (Keynesowskie)
        df['Wskaznik_CPI_Dec'] = df['Wskaznik_CPI'] / 100
        df['Realne_Wynagrodzenie'] = df['Przecietne_Wynagrodzenie_Kraj'] / df['Wskaznik_CPI_Dec']
        df['Realny_PKB'] = df['PKB_Nominalny'] / df['Wskaznik_CPI_Dec']
        
        # Treatment: Wskaźnik Kaitza
        df['Kaitz_Index'] = df['Placa_Minimalna'] / df['Realne_Wynagrodzenie']
        
        # Heterogeniczność
        df['Wzrost_PKB'] = df.groupby('Kod')['Realny_PKB'].pct_change() * 100
        df['Udzial_Przemyslu'] = df['WDB_Przemysl'] / df['WDB_Ogolem']
        df['Produktywnosc'] = df['Realny_PKB'] / df['Zatrudnieni_Ogolem']
        
        return df.dropna()
    except Exception as e:
        st.error(f"Błąd przetwarzania plików: {e}. Sprawdź czy nazwy kolumn ('Kod', 'Rok', 'PKB_Nominalny' itp.) są poprawne.")
        return pd.DataFrame()

@st.cache_resource
def run_models(df):
    results = {}
    
    # 1. Baseline: Klasyczny model liniowy (Proxy dla Fixed Effects / OLS)
    X_ols = sm.add_constant(df[['Kaitz_Index', 'Bezrobocie_Rejestrowane', 'Wzrost_PKB', 'Udzial_Przemyslu']])
    y_ols = df['Stopa_Zatrudnienia']
    ols_model = sm.OLS(y_ols, X_ols).fit()
    results['ols_summary'] = ols_model.summary()
    results['ols_ate'] = ols_model.params['Kaitz_Index']

    # 2. Causal Machine Learning: Causal Forest DML z rdzeniem XGBoost
    Y = df['Stopa_Zatrudnienia'] 
    T = df['Kaitz_Index'] 
    X_cols = ['Bezrobocie_Rejestrowane', 'Wzrost_PKB', 'Realny_PKB_per_capita', 'Udzial_Przemyslu', 'Produktywnosc']
    X = df[X_cols]
    W = df[['Populacja', 'Wskaznik_CPI']] 

    causal_forest = CausalForestDML(
        model_y=xgb.XGBRegressor(n_estimators=100, max_depth=3),
        model_t=xgb.XGBRegressor(n_estimators=100, max_depth=3),
        discrete_treatment=False,
        n_estimators=1000, 
        random_state=42
    )
    causal_forest.fit(Y, T, X=X, W=W)
    
    df['Estimated_Effect_CF'] = causal_forest.effect(X)
    results['ate_cf'] = np.mean(df['Estimated_Effect_CF'])
    results['shap_values'] = causal_forest.shap_values(X)
    results['X_data'] = X
    results['df_final'] = df
    
    return results

# --- INTERFEJS APLIKACJI ---
st.sidebar.header("Panel Sterowania")
run_analysis = st.sidebar.button("Uruchom Pełną Analizę")

if run_analysis:
    with st.spinner("Przetwarzanie danych i trenowanie potężnych modeli ML..."):
        # UWAGA: Podaj dokładne nazwy plików, które masz na GitHubie w dziale RYNE, RACH i CENY
        df = load_and_clean_data("RYNE_4098.xlsx", "RACH_3510.xlsx", "CENY_2496_XTAB_20261007223438.xlsx")
        
        if not df.empty:
            res = run_models(df)
            df_final = res['df_final']
            
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Dane & EDA", "📈 Modele Liniowe (Baseline)", "🌲 Causal Machine Learning", "🔍 Heterogeniczność (SHAP)"])
            
            with tab1:
                st.subheader("Oczyszczony Panel Wojewódzki (Naprawione nazwy)")
                st.dataframe(df_final[['Wojewodztwo', 'Rok', 'Kaitz_Index', 'Stopa_Zatrudnienia', 'Wzrost_PKB', 'Udzial_Przemyslu']].head(10))
                
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.histplot(df_final['Kaitz_Index'], bins=20, kde=True, ax=ax)
                ax.set_title("Rozkład Wskaźnika Kaitza w panelu badawczym")
                st.pyplot(fig)
                
            with tab2:
                st.subheader("Model Klasyczny (Ekonometria)")
                st.metric(label="OLS ATE (Zmiana zatrudnienia przy wzroście Kaitza o 1 jednostkę)", value=f"{res['ols_ate']:.4f}")
                st.text(res['ols_summary'])
                
            with tab3:
                st.subheader("Causal Forest & Double ML (Nieliniowość)")
                st.metric(label="Causal Forest ATE", value=f"{res['ate_cf']:.4f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Efekty Regionalne (Średnia na województwo)**")
                    regional_effects = df_final.groupby('Wojewodztwo')['Estimated_Effect_CF'].mean().sort_values()
                    fig2, ax2 = plt.subplots(figsize=(6, 8))
                    sns.barplot(x=regional_effects.values, y=regional_effects.index, palette="viridis", ax=ax2)
                    ax2.set_xlabel("Średni Estymowany Efekt Zatrudnieniowy")
                    st.pyplot(fig2)
                
                with col2:
                    st.write("**Nieliniowość Efektu (Kaitz Index vs Zmiana Zatrudnienia)**")
                    fig3, ax3 = plt.subplots(figsize=(6, 5))
                    sns.scatterplot(data=df_final, x='Kaitz_Index', y='Estimated_Effect_CF', hue='Bezrobocie_Rejestrowane', palette='coolwarm', ax=ax3)
                    ax3.axhline(0, color='black', linestyle='--')
                    ax3.set_title("Wykryte progi wrażliwości")
                    st.pyplot(fig3)
                    
            with tab4:
                st.subheader("Interpretacja SHAP (Wpływ cech gospodarki)")
                st.markdown("Czerwony kolor = wysoka wartość danej zmiennej. Odchylenie w lewo = niszczenie zatrudnienia.")
                fig4, ax4 = plt.subplots(figsize=(10, 6))
                shap.summary_plot(res['shap_values']['Stopa_Zatrudnienia']['Kaitz_Index'], res['X_data'], show=False)
                st.pyplot(fig4)
        else:
            st.error("Brak danych po połączeniu. Sprawdź, czy pliki zostały wgrane i mają poprawne nagłówki.")
else:
    st.info("Naciśnij przycisk po lewej stronie, aby rozpocząć proces Causal ML.")
