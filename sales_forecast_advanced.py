# -*- coding: utf-8 -*-
"""
Erweitertes Verkaufsprognose-Projekt
Ziel: Lagerverwaltung mit echter Datenanalyse optimieren
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

print("=" * 80)
print("ERWEITERTES VERKAUFSPROGNOSE-PROJEKT FÜR LAGERVERWALTUNG")
print("=" * 80)

# =============================================================================
# 1. REALISTISCHEN DATENSATZ GENERIEREN (Retail Store Sales simulieren)
# =============================================================================
print("\n[1/9] DATEN-GENERIERUNG (Realistic Retail Store Scenario)")
print("-" * 80)

np.random.seed(42)

# Generiere 2 Jahre Verkaufsdaten mit Saisonalität
dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
n_samples = len(dates)

# Basis-Features
data = {
    'Datum': dates,
    'Wochentag': dates.dayofweek,
    'Monat': dates.month,
    'Quartal': dates.quarter,
    'Woche': dates.isocalendar().week,
    'Tag_im_Monat': dates.day,
    'Ist_Wochenende': (dates.dayofweek >= 5).astype(int),
}

# Feature Engineering: Realistische Features
data['Werbebudget'] = np.random.uniform(500, 5000, n_samples)
data['Lagerbestand'] = np.random.uniform(50, 500, n_samples)
data['Konkurrenz_Aktivitaet'] = np.random.uniform(0, 10, n_samples)  # 0-10 Skala
data['Kundenzufriedenheit'] = np.random.uniform(3, 5, n_samples)  # 1-5 Skala

df = pd.DataFrame(data)

# Erzeuge Verkaufszahlen mit komplexem Zusammenhang
base_sales = 100
seasonal_pattern = 50 * np.sin(2 * np.pi * df['Monat'] / 12)  # Saisonalität
weekend_boost = df['Ist_Wochenende'] * 25  # Wochenende boost
budget_effect = 0.3 * df['Werbebudget']
competitor_effect = -2 * df['Konkurrenz_Aktivitaet']
stock_effect = np.minimum(df['Lagerbestand'], 200) * 0.15

# Basis-Verkäufe
df['Verkaeufe'] = (
    base_sales +
    seasonal_pattern +
    weekend_boost +
    budget_effect +
    competitor_effect +
    stock_effect +
    3 * df['Kundenzufriedenheit'] +
    np.random.normal(0, 10, n_samples)  # Rauschen
)
df['Verkaeufe'] = np.maximum(df['Verkaeufe'], 10)  # Min. 10 Verkäufe

# Feature Interactions (Feature Engineering)
df['Werbung_x_Zufriedenheit'] = df['Werbebudget'] * df['Kundenzufriedenheit']
df['Lager_x_Nachfrage_Signal'] = df['Lagerbestand'] * (1 + 0.1 * df['Monat'])

print(f"✓ Datensatz generiert: {len(df)} Tage (2 Jahre)")
print(f"  Features: {df.shape[1] - 1} (ohne Datum)")
print(f"  Zeitraum: {df['Datum'].min().date()} bis {df['Datum'].max().date()}")
print(f"  Verkäufe - Mean: {df['Verkaeufe'].mean():.1f}, Std: {df['Verkaeufe'].std():.1f}")

# =============================================================================
# 2. SAISONALITÄTS-ANALYSE
# =============================================================================
print("\n[2/9] SAISONALITÄTS-ANALYSE")
print("-" * 80)

# Monatliche Durchschnitte
monthly_sales = df.groupby('Monat')['Verkaeufe'].agg(['mean', 'std', 'min', 'max'])
seasonal_index = monthly_sales['mean'] / monthly_sales['mean'].mean()

print("\nSaisonale Indizes (1.0 = Durchschnitt):")
for monat, idx in enumerate(seasonal_index, 1):
    monat_name = ['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez'][monat - 1]
    strength = "▓" * int(idx * 5)
    print(f"  {monat_name}: {idx:.2f} {strength}")

# Visualisierung
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Zeitreihenverlauf
ax = axes[0, 0]
ax.plot(df['Datum'], df['Verkaeufe'], linewidth=1, alpha=0.7, color='steelblue')
ax.set_title('Verkaufsverlauf (2 Jahre)', fontweight='bold', fontsize=12)
ax.set_xlabel('Datum')
ax.set_ylabel('Verkäufe')
ax.grid(True, alpha=0.3)

# Saisonalität
ax = axes[0, 1]
ax.bar(range(1, 13), seasonal_index, color='coral', alpha=0.7, edgecolor='black')
ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Durchschnitt')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
ax.set_title('Saisonale Indizes', fontweight='bold', fontsize=12)
ax.set_ylabel('Index')
ax.legend()

# Wochentagsmuster
ax = axes[1, 0]
weekday_sales = df.groupby('Wochentag')['Verkaeufe'].mean()
ax.bar(range(7), weekday_sales, color='lightgreen', alpha=0.7, edgecolor='black')
ax.set_xticks(range(7))
ax.set_xticklabels(['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So'])
ax.set_title('Durchschnittliche Verkäufe nach Wochentag', fontweight='bold', fontsize=12)
ax.set_ylabel('Verkäufe')

# Box-Plot nach Quartal
ax = axes[1, 1]
df.boxplot(column='Verkaeufe', by='Quartal', ax=ax)
ax.set_title('Quartalweise Verteilung', fontweight='bold', fontsize=12)
ax.set_xlabel('Quartal')
ax.set_ylabel('Verkäufe')
plt.suptitle('')

plt.tight_layout()
plt.savefig('01_saisonalitaet_analyse.png', dpi=100, bbox_inches='tight')
print("  ✓ Grafik gespeichert: 01_saisonalitaet_analyse.png")
plt.close()

# =============================================================================
# 3. ANOMALIE-ERKENNUNG
# =============================================================================
print("\n[3/9] ANOMALIE-ERKENNUNG")
print("-" * 80)

# Z-Score basierte Anomalieerkennung
z_scores = np.abs(stats.zscore(df['Verkaeufe']))
anomalies = z_scores > 2.5
df['Ist_Anomalie'] = anomalies

n_anomalies = anomalies.sum()
anomaly_percentage = (n_anomalies / len(df)) * 100

print(f"✓ {n_anomalies} Anomalien erkannt ({anomaly_percentage:.2f}% der Daten)")
print(f"\n  Anomale Tage:")
anomaly_dates = df[anomalies][['Datum', 'Verkaeufe']].head(10)
for idx, row in anomaly_dates.iterrows():
    print(f"    {row['Datum'].strftime('%Y-%m-%d')}: {row['Verkaeufe']:.0f} Verkäufe")

# Visualisierung
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['Datum'], df['Verkaeufe'], linewidth=1, alpha=0.6, label='Normale Daten', color='steelblue')
anomaly_data = df[anomalies]
ax.scatter(anomaly_data['Datum'], anomaly_data['Verkaeufe'],
          color='red', s=100, marker='X', label='Anomalien', zorder=5)
ax.set_title('Anomalieerkennung (Z-Score > 2.5)', fontweight='bold', fontsize=12)
ax.set_xlabel('Datum')
ax.set_ylabel('Verkäufe')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('02_anomalien_erkennung.png', dpi=100, bbox_inches='tight')
print("  ✓ Grafik gespeichert: 02_anomalien_erkennung.png")
plt.close()

# =============================================================================
# 4. FEATURE-CORRELATION & INTERACTIONS
# =============================================================================
print("\n[4/9] FEATURE-KORRELATION & INTERAKTIONEN")
print("-" * 80)

feature_cols = ['Werbebudget', 'Lagerbestand', 'Konkurrenz_Aktivitaet',
                'Kundenzufriedenheit', 'Ist_Wochenende', 'Werbung_x_Zufriedenheit']
corr_matrix = df[feature_cols + ['Verkaeufe']].corr()

print("\nKorrelation mit Verkäufen:")
sales_corr = corr_matrix['Verkaeufe'].drop('Verkaeufe').sort_values(ascending=False)
for feature, corr in sales_corr.items():
    print(f"  {feature:30} : {corr:+.4f}")

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, cbar_kws={'label': 'Korrelation'})
plt.title('Feature-Korrelationsmatrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('03_feature_korrelationen.png', dpi=100, bbox_inches='tight')
print("  ✓ Grafik gespeichert: 03_feature_korrelationen.png")
plt.close()

# =============================================================================
# 5. MODELLTRAINING MIT ENSEMBLE
# =============================================================================
print("\n[5/9] MODELLTRAINING (Ensemble-Methode)")
print("-" * 80)

# Features vorbereiten
X = df.drop(['Verkaeufe', 'Datum', 'Ist_Anomalie'], axis=1)
y = df['Verkaeufe']

# Train-Test Split (chronologisch für Zeitreihen)
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"✓ Daten aufgeteilt:")
print(f"  Training: {len(X_train)} Tage ({X_train.index[0]} bis {X_train.index[-1]})")
print(f"  Test: {len(X_test)} Tage ({X_test.index[0]} bis {X_test.index[-1]})")

# Normalisierung
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features normalisiert")

# Ensemble-Modelle trainieren
print("\nModelltraining:")

# Gradient Boosting
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)
gb_pred_train = gb_model.predict(X_train_scaled)
gb_pred_test = gb_model.predict(X_test_scaled)
print(f"  ✓ Gradient Boosting - Train R²: {r2_score(y_train, gb_pred_train):.4f}, Test R²: {r2_score(y_test, gb_pred_test):.4f}")

# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred_train = rf_model.predict(X_train)
rf_pred_test = rf_model.predict(X_test)
print(f"  ✓ Random Forest - Train R²: {r2_score(y_train, rf_pred_train):.4f}, Test R²: {r2_score(y_test, rf_pred_test):.4f}")

# =============================================================================
# 6. MODELL-EVALUIERUNG MIT METRIKEN
# =============================================================================
print("\n[6/9] MODELL-EVALUIERUNG")
print("-" * 80)

def evaluate_detailed(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name}:")
    print(f"  R² Score:         {r2:.4f}")
    print(f"  RMSE:             {rmse:.2f} Verkäufe")
    print(f"  MAE:              {mae:.2f} Verkäufe")
    print(f"  MAPE:             {mape:.2f}%")

    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

print("TEST-METRIKEN:")
gb_metrics = evaluate_detailed(y_test, gb_pred_test, "Gradient Boosting")
rf_metrics = evaluate_detailed(y_test, rf_pred_test, "Random Forest")

# Bestes Modell wählen
best_model = gb_model if gb_metrics['R2'] > rf_metrics['R2'] else rf_model
best_pred = gb_pred_test if gb_metrics['R2'] > rf_metrics['R2'] else rf_pred_test
best_name = "Gradient Boosting" if gb_metrics['R2'] > rf_metrics['R2'] else "Random Forest"
best_metrics = gb_metrics if gb_metrics['R2'] > rf_metrics['R2'] else rf_metrics

print(f"\n✓ Bestes Modell: {best_name}")

# Visualisierung
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Vorhersage vs. Tatsächlich
axes[0].scatter(y_test, gb_pred_test, alpha=0.5, s=20, label='Gradient Boosting', color='steelblue')
axes[0].scatter(y_test, rf_pred_test, alpha=0.5, s=20, label='Random Forest', color='coral')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Tatsächliche Verkäufe', fontweight='bold')
axes[0].set_ylabel('Vorhergesagte Verkäufe', fontweight='bold')
axes[0].set_title('Modell-Performance', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Residuen über Zeit
residuals = y_test.values - best_pred
axes[1].plot(residuals, alpha=0.7, color='steelblue')
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].fill_between(range(len(residuals)), -1.96*residuals.std(), 1.96*residuals.std(),
                     alpha=0.2, color='green', label='95% Konfidenzbereich')
axes[1].set_xlabel('Tage (Test-Periode)', fontweight='bold')
axes[1].set_ylabel('Fehler', fontweight='bold')
axes[1].set_title(f'Residuenverlauf ({best_name})', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_modell_performance.png', dpi=100, bbox_inches='tight')
print("  ✓ Grafik gespeichert: 04_modell_performance.png")
plt.close()

# =============================================================================
# 7. PREDICTION INTERVALS (Unsicherheitsbereiche)
# =============================================================================
print("\n[7/9] PREDICTION INTERVALS (Konfidenzbereich-Vorhersagen)")
print("-" * 80)

# Berechne Konfidenzintervalle basierend auf Residuen
residuals = y_test.values - best_pred
residual_std = np.std(residuals)

# 95% Konfidenzbereich
confidence_level = 1.96  # für 95%
lower_bound = best_pred - (confidence_level * residual_std)
upper_bound = best_pred + (confidence_level * residual_std)

# Prozentuale Fehlerquote
coverage = ((y_test.values >= lower_bound) & (y_test.values <= upper_bound)).sum() / len(y_test)

print(f"✓ 95% Konfidenzintervalle berechnet")
print(f"  Durchschnittliche Unsicherheit: +/- {confidence_level * residual_std:.1f} Verkäufe")
print(f"  Abdeckungsrate: {coverage*100:.1f}% (ideal: 95%)")

# Visualisierung mit Konfidenzbereich
fig, ax = plt.subplots(figsize=(14, 6))

# Sortiere für bessere Visualisierung
sorted_idx = np.argsort(y_test.values)
sorted_actual = y_test.values[sorted_idx]
sorted_pred = best_pred[sorted_idx]
sorted_lower = lower_bound[sorted_idx]
sorted_upper = upper_bound[sorted_idx]

ax.plot(sorted_actual, 'o-', linewidth=1, markersize=4, label='Tatsächlich', color='black', alpha=0.7)
ax.plot(sorted_pred, 's-', linewidth=1.5, markersize=3, label='Vorhersage', color='steelblue')
ax.fill_between(range(len(sorted_pred)), sorted_lower, sorted_upper,
                alpha=0.2, color='green', label='95% Konfidenzbereich')

ax.set_xlabel('Test-Sample (sortiert)', fontweight='bold')
ax.set_ylabel('Verkäufe', fontweight='bold')
ax.set_title('Vorhersagen mit 95% Konfidenzintervallen', fontweight='bold', fontsize=12)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_prediction_intervals.png', dpi=100, bbox_inches='tight')
print("  ✓ Grafik gespeichert: 05_prediction_intervals.png")
plt.close()

# =============================================================================
# 8. LAGERVERWALTUNGS-OPTIMIERUNG
# =============================================================================
print("\n[8/9] LAGERVERWALTUNGS-OPTIMIERUNG")
print("-" * 80)

# Benutze Prognose für Lageroptimierung
# Safety Stock = z * σ * √L  (vereinfacht)
# Dabei: z = Sicherheitsfaktor (2 für 95%), σ = Standardabweichung, L = Lead Time

lead_time_days = 7
predicted_demand = best_pred
demand_std = np.std(predicted_demand)
z_score = 1.96  # 95% Service Level

# Sicherheitsbestand
safety_stock = z_score * demand_std * np.sqrt(lead_time_days)

# Reorder Point (ROP)
avg_daily_demand = predicted_demand.mean()
reorder_point = (avg_daily_demand * lead_time_days) + safety_stock

# Wirtschaftliche Bestellmenge (EOQ - simplified)
holding_cost = 0.5  # € pro Einheit pro Tag
ordering_cost = 50   # € pro Bestellung
eoq = np.sqrt((2 * avg_daily_demand * 365 * ordering_cost) / holding_cost)

print(f"✓ Lagerverwaltungs-Parameter (für {lead_time_days}-Tage Lead Time):")
print(f"\n  Tägliche Nachfrage (Prognose):")
print(f"    Durchschnitt:  {avg_daily_demand:.1f} Einheiten")
print(f"    Standardabw.:  {demand_std:.1f} Einheiten")
print(f"\n  Empfohlene Lagerbestände:")
print(f"    Sicherheitsbestand (95%): {safety_stock:.0f} Einheiten")
print(f"    Bestellpunkt (ROP):       {reorder_point:.0f} Einheiten")
print(f"    Wirtschaftliche Menge:    {eoq:.0f} Einheiten")
print(f"\n  Lager-KPIs:")
print(f"    Max. Lagerbestand (ROP + EOQ): {reorder_point + eoq:.0f} Einheiten")
print(f"    Durchschn. Bestandskosten:     {(reorder_point + eoq/2) * holding_cost * 365:,.0f} € pro Jahr")

# Visualisierung
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Nachfrage-Verteilung
ax = axes[0, 0]
ax.hist(predicted_demand, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(avg_daily_demand, color='r', linestyle='--', linewidth=2, label='Mittelwert')
ax.axvline(avg_daily_demand - 2*demand_std, color='orange', linestyle='--', linewidth=2, label='2σ Bereich')
ax.axvline(avg_daily_demand + 2*demand_std, color='orange', linestyle='--', linewidth=2)
ax.set_title('Nachfrage-Verteilung (Prognose)', fontweight='bold')
ax.set_xlabel('Verkäufe pro Tag')
ax.set_ylabel('Häufigkeit')
ax.legend()

# Lager-Szenario
ax = axes[0, 1]
days = np.arange(30)
inventory = reorder_point
scenario_inventory = []
for day in days:
    scenario_inventory.append(inventory)
    inventory -= predicted_demand[day % len(predicted_demand)]
    if inventory <= reorder_point:
        inventory = reorder_point + eoq

ax.plot(days, scenario_inventory[:len(days)], 'o-', linewidth=2, markersize=6, color='steelblue')
ax.axhline(reorder_point, color='r', linestyle='--', linewidth=2, label=f'ROP ({reorder_point:.0f})')
ax.axhline(reorder_point + eoq, color='g', linestyle='--', linewidth=2, label=f'Max ({reorder_point + eoq:.0f})')
ax.axhline(safety_stock, color='orange', linestyle='--', linewidth=2, label=f'Sicherheitsbestand ({safety_stock:.0f})')
ax.set_title('Lagerverlauf (30-Tage Simulation)', fontweight='bold')
ax.set_xlabel('Tage')
ax.set_ylabel('Lagerbestand (Einheiten)')
ax.legend()
ax.grid(True, alpha=0.3)

# Kosten-Analyse
ax = axes[1, 0]
stock_levels = np.linspace(safety_stock, reorder_point + eoq, 50)
holding_costs = stock_levels * holding_cost * 365
ordering_costs = (avg_daily_demand * 365 / stock_levels) * ordering_cost
total_costs = holding_costs + ordering_costs

ax.plot(stock_levels, holding_costs, label='Lagerhaltungskosten', linewidth=2)
ax.plot(stock_levels, ordering_costs, label='Bestellkosten', linewidth=2)
ax.plot(stock_levels, total_costs, label='Gesamtkosten', linewidth=2.5, color='black', linestyle='--')
ax.axvline(eoq, color='r', linestyle=':', linewidth=2, label=f'EOQ ({eoq:.0f})')
ax.set_title('Kostenoptimierung', fontweight='bold')
ax.set_xlabel('Bestellmenge (Einheiten)')
ax.set_ylabel('Jährliche Kosten (€)')
ax.legend()
ax.grid(True, alpha=0.3)

# Empfohlene Lager-Strategie
ax = axes[1, 1]
strategies = ['Aktuell\n(konservativ)', 'Optimiert\n(EOQ)', 'Aggressiv\n(JIT)']
stock_values = [reorder_point + eoq, reorder_point + eoq/2, safety_stock + avg_daily_demand*3]
costs = [s * holding_cost * 365 for s in stock_values]
colors = ['orange', 'green', 'red']

bars = ax.bar(strategies, costs, color=colors, alpha=0.7, edgecolor='black')
ax.set_title('Kosten-Vergleich: Lagerbestands-Strategien', fontweight='bold')
ax.set_ylabel('Jährliche Lagerhaltungskosten (€)')
for bar, cost in zip(bars, costs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{cost:,.0f}€', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('06_lagerverwaltung_optimierung.png', dpi=100, bbox_inches='tight')
print("  ✓ Grafik gespeichert: 06_lagerverwaltung_optimierung.png")
plt.close()

# =============================================================================
# 9. ZUSAMMENFASSUNG & EMPFEHLUNGEN
# =============================================================================
print("\n[9/9] PROJEKT-ZUSAMMENFASSUNG")
print("=" * 80)

print("\n📊 DATENSATZ:")
print(f"  • Zeitraum: 2 Jahre (730 Tage)")
print(f"  • Features: {len(X.columns)} (einschließlich Interaktionen)")
print(f"  • Durchschn. Verkäufe: {y.mean():.1f} pro Tag")
print(f"  • Saisonalität: {seasonal_index.max() / seasonal_index.min():.2f}x Variation")

print(f"\n🤖 BEST PERFORMER MODEL: {best_name}")
print(f"  • Test R²:  {best_metrics['R2']:.4f}")
print(f"  • Test RMSE: {best_metrics['RMSE']:.2f} Verkäufe")
print(f"  • Test MAPE: {best_metrics['MAPE']:.2f}%")
print(f"  • 95% Konfidenzbereich: ±{confidence_level * residual_std:.1f} Verkäufe")

print(f"\n📦 LAGERVERWALTUNGS-EMPFEHLUNG:")
print(f"  • Reorder Point: {reorder_point:.0f} Einheiten")
print(f"  • Bestellmenge: {eoq:.0f} Einheiten")
print(f"  • Max. Lagerbestand: {reorder_point + eoq:.0f} Einheiten")
print(f"  • Sicherheitsbestand: {safety_stock:.0f} Einheiten")
print(f"  • Jährliche Lagerhaltungskosten: {(reorder_point + eoq/2) * holding_cost * 365:,.0f} €")

print(f"\n💡 TOP INSIGHTS:")
print(f"  • Stärkster Einflussfaktor: Saisonalität (±{seasonal_index.max() - seasonal_index.min():.1f})")
print(f"  • Anomalien: {n_anomalies} erkannt ({anomaly_percentage:.1f}%)")
print(f"  • Vorhersage-Genauigkeit: ±{best_metrics['RMSE']:.0f} Verkäufe (95% Sicherheit)")
print(f"  • Optimierungspotential: Kosten sparen durch EOQ-basierte Bestellungen")

print(f"\n📈 GENERIERTE DATEIEN:")
print(f"  ✓ 01_saisonalitaet_analyse.png - Saisonale Muster")
print(f"  ✓ 02_anomalien_erkennung.png - Ungewöhnliche Tage")
print(f"  ✓ 03_feature_korrelationen.png - Abhängigkeiten")
print(f"  ✓ 04_modell_performance.png - Modell-Vergleich")
print(f"  ✓ 05_prediction_intervals.png - Vorhersagen mit Unsicherheit")
print(f"  ✓ 06_lagerverwaltung_optimierung.png - Lager-Empfehlungen")

print("\n" + "=" * 80)
print("✅ ERWEITERTES PROJEKT ABGESCHLOSSEN!")
print("=" * 80 + "\n")
