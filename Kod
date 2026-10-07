import pandas as pd
import numpy as np
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt

def prepare_keynesian_panel(wyna_df, ryne_df, rach_df, ceny_df, min_wage_dict):
    """
    Funkcja łączy surowe dane GUS i wylicza kluczowe wskaźniki makroekonomiczne.
    Oczekuje wstępnie oczyszczonych ramek danych z kolumnami: ['Kod', 'Nazwa', 'Rok', 'Wartosc'].
    """
    # 1. Łączenie danych (Merge) po kodzie regionu i roku
    df = wyna_df.merge(ryne_df, on=['Kod', 'Rok'], suffixes=('_wyna', '_ryne'))
    df = df.merge(rach_df, on=['Kod', 'Rok'])
    df = df.merge(ceny_df, on=['Rok']) # Inflacja zazwyczaj na poziomie kraju/roku
    
    # 2. Deflowanie wartości nominalnych (CENY) do wartości realnych
    # Zakładamy, że 'Wskaźnik_CPI' to np. 105.0 dla 5% inflacji
    df['Realne_Wynagrodzenie'] = df['Przecietne_Wynagrodzenie_Nominalne'] / (df['Wskaznik_CPI'] / 100)
    df['Realny_PKB'] = df['PKB_Nominalny'] / (df['Wskaznik_CPI'] / 100)
    
    # 3. Zmienna Treatment (H2: Kaitz Index)
    df['Placa_Minimalna_Ustawowa'] = df['Rok'].map(min_wage_dict)
    df['Kaitz_Index'] = df['Placa_Minimalna_Ustawowa'] / df['Realne_Wynagrodzenie']
    
    # 4. Zmienne Keynesowskie i heterogeniczne
    # H3: Rezerwy na rynku pracy
    df['Stopa_Bezrobocia'] = df['Bezrobocie_Rejestrowane'] 
    # H6: Cykl koniunkturalny (popyt zagregowany)
    df['Wzrost_PKB'] = df.groupby('Kod')['Realny_PKB'].pct_change() * 100
    # Struktura gospodarki (Przemysł vs Usługi)
    df['Udzial_Przemyslu'] = df['WDB_Przemysl'] / df['WDB_Ogolem']
    
    return df.dropna()

def run_causal_analysis(df):
    """
    Estymacja heterogenicznych efektów przyczynowych za pomocą Causal Forest.
    """
    # Zmienna objaśniana (Outcome - H1)
    Y = df['Stopa_Zatrudnienia'] 
    
    # Zmienna decyzyjna (Treatment - H2)
    T = df['Kaitz_Index'] 
    
    # Zmienne warunkujące heterogeniczność efektu (Heterogeneity Features)
    # Odpowiadają za H3 (Bezrobocie), H4 (PKB pc), H6 (Cykl koniunkturalny) i strukturę sektorową
    X_cols = ['Stopa_Bezrobocia', 'Wzrost_PKB', 'Realny_PKB_per_capita', 'Udzial_Przemyslu']
    X = df[X_cols]
    
    # Konfundery (Zmienne kontrolujące bias - W)
    W = df[['Populacja', 'Wskaznik_CPI']] 

    # Inicjalizacja modelu Causal Forest DML
    # Model_y i model_t to algorytmy "odszumiające" (Double ML)
    causal_forest = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=5),
        model_t=RandomForestRegressor(n_estimators=100, max_depth=5),
        discrete_treatment=False,
        n_estimators=500,
        random_state=42
    )
    
    print("Trenowanie modelu Causal Forest...")
    causal_forest.fit(Y, T, X=X, W=W)
    
    # Estymacja CATE (Conditional Average Treatment Effect) dla każdego regionu/roku
    cate_estimates = causal_forest.effect(X)
    df['Estimated_Employment_Effect'] = cate_estimates
    
    # Średni efekt (H1)
    ate = np.mean(cate_estimates)
    print(f"\nŚredni Efekt (ATE) wzrostu wskaźnika Kaitza na zatrudnienie: {ate:.4f}")
    
    # Analiza SHAP (Interpretacja czynników heterogeniczności)
    print("\nObliczanie wartości SHAP dla Causal Forest...")
    shap_values = causal_forest.shap_values(X)
    
    # Wykres SHAP pokazujący, co najbardziej wpływa na zróżnicowanie efektu
    shap.summary_plot(shap_values['Stopa_Zatrudnienia']['Kaitz_Index'], X, show=False)
    plt.title("Analiza SHAP: Determinanty efektu płacy minimalnej")
    plt.tight_layout()
    plt.savefig("shap_heterogeneity.png")
    
    return df, causal_forest

# Sposób użycia (wymaga podmienienia na wczytane ramki z plików GUS):
# min_wage_history = {2015: 1750, 2016: 1850, 2017: 2000, 2018: 2100} # Przykładowe dane
# panel_df = prepare_keynesian_panel(df_wynagrodzenia, df_rynek, df_rachunki, df_ceny, min_wage_history)
# results_df, model = run_causal_analysis(panel_df)
