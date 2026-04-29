#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Umsatz-Prognose Projekt
Ziel: Vorhersage von Verkaufsmengen basierend auf verschiedenen Features
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

print("=" * 70)
print("UMSATZ-PROGNOSE PROJEKT - Datenanalyse und Modellierung")
print("=" * 70)

# =============================================================================
# 1. DATEN GENERIEREN
# =============================================================================
print("\n1️⃣  DATEN GENERIEREN")
print("-" * 70)

np.random.seed(42)
n_samples = 500

data = {
    'Monat': np.tile(range(1, 13), n_samples // 12 + 1)[:n_samples],
    'Werbebudget': np.random.uniform(1000, 10000, n_samples),
    'Lagerbestand': np.random.uniform(100, 1000, n_samples),
    'Mitarbeiter': np.random.randint(5, 50, n_samples),
    'Kundenanzahl': np.random.randint(100, 1000, n_samples),
}

# Erzeuge Umsatz mit realistischem Zusammenhang
data['Umsatz'] = (
    0.5 * data['Werbebudget'] +
    2 * data['Lagerbestand'] +
    100 * data['Mitarbeiter'] +
    5 * data['Kundenanzahl'] +
    200 * np.sin(np.array(data['Monat']) * 2 * np.pi / 12) +  # Saisonalität
    np.random.normal(0, 500, n_samples)  # Rauschen
)

df = pd.DataFrame(data)

print(f"✓ Datensatz mit {len(df)} Zeilen erstellt")
print(f"  Shape: {df.shape}")
print(f"\n  Statistiken:")
print(df.describe().to_string())

# =============================================================================
# 2. EXPLORATIVE DATENANALYSE
# =============================================================================
print("\n2️⃣  EXPLORATIVE DATENANALYSE (EDA)")
print("-" * 70)

print(f"✓ Fehlende Werte: {df.isnull().sum().sum()}")

# Korrelationsmatrix
correlation = df.corr()
print(f"\n  Korrelation mit Umsatz (absteigend):")
umsatz_corr = correlation['Umsatz'].sort_values(ascending=False)
for feature, corr in umsatz_corr.items():
    print(f"    {feature:20} : {corr:+.4f}")

# Visualisierung - Verteilungen
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('Datenverteilungen', fontsize=16, fontweight='bold')

columns = df.columns
for idx, col in enumerate(columns):
    ax = axes[idx // 3, idx % 3]
    ax.hist(df[col], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel('Wert')
    ax.set_ylabel('Häufigkeit')

plt.tight_layout()
plt.savefig('01_verteilungen.png', dpi=100, bbox_inches='tight')
print("  ✓ Grafik gespeichert: 01_verteilungen.png")
plt.close()

# Korrelationsheatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, cbar_kws={'label': 'Korrelation'})
plt.title('Korrelationsmatrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('02_korrelationen.png', dpi=100, bbox_inches='tight')
print("  ✓ Grafik gespeichert: 02_korrelationen.png")
plt.close()

# =============================================================================
# 3. FEATURE ENGINEERING & VORBEREITUNG
# =============================================================================
print("\n3️⃣  FEATURE ENGINEERING & DATENVORBEREITUNG")
print("-" * 70)

X = df.drop('Umsatz', axis=1)
y = df['Umsatz']

print(f"✓ Features (X): {X.shape}")
print(f"✓ Target (y): {y.shape}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  Trainingsmenge: {X_train.shape[0]} Samples")
print(f"  Testmenge: {X_test.shape[0]} Samples")

# Normalisierung
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Daten normalisiert (StandardScaler)")

# =============================================================================
# 4. MODELLTRAINING
# =============================================================================
print("\n4️⃣  MODELLTRAINING")
print("-" * 70)

# Linear Regression
print("\n  📊 Linear Regression:")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred_train = lr_model.predict(X_train_scaled)
lr_pred_test = lr_model.predict(X_test_scaled)

lr_train_r2 = r2_score(y_train, lr_pred_train)
lr_test_r2 = r2_score(y_test, lr_pred_test)
print(f"    Train R²: {lr_train_r2:.4f}")
print(f"    Test R²:  {lr_test_r2:.4f}")

# Random Forest
print("\n  🌲 Random Forest:")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred_train = rf_model.predict(X_train)
rf_pred_test = rf_model.predict(X_test)

rf_train_r2 = r2_score(y_train, rf_pred_train)
rf_test_r2 = r2_score(y_test, rf_pred_test)
print(f"    Train R²: {rf_train_r2:.4f}")
print(f"    Test R²:  {rf_test_r2:.4f}")

# =============================================================================
# 5. MODELL-EVALUIERUNG
# =============================================================================
print("\n5️⃣  MODELL-EVALUIERUNG")
print("-" * 70)

def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name}:")
    print(f"  MSE:  {mse:>12,.2f}")
    print(f"  RMSE: {rmse:>12,.2f}")
    print(f"  MAE:  {mae:>12,.2f}")
    print(f"  R²:   {r2:>12.4f}")
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

print("\n📈 TRAININGS-METRIKEN:")
evaluate_model(y_train, lr_pred_train, '  Linear Regression (Train)')
evaluate_model(y_train, rf_pred_train, '  Random Forest (Train)')

print("\n📊 TEST-METRIKEN:")
lr_test_metrics = evaluate_model(y_test, lr_pred_test, '  Linear Regression (Test)')
rf_test_metrics = evaluate_model(y_test, rf_pred_test, '  Random Forest (Test)')

# Vorhersage vs. Tatsächliche Werte
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear Regression
axes[0].scatter(y_test, lr_pred_test, alpha=0.6, s=50, color='steelblue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Tatsächlicher Umsatz', fontsize=12)
axes[0].set_ylabel('Vorhergesagter Umsatz', fontsize=12)
axes[0].set_title(f'Linear Regression (R²={lr_test_metrics["R2"]:.4f})', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Random Forest
axes[1].scatter(y_test, rf_pred_test, alpha=0.6, s=50, color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Tatsächlicher Umsatz', fontsize=12)
axes[1].set_ylabel('Vorhergesagter Umsatz', fontsize=12)
axes[1].set_title(f'Random Forest (R²={rf_test_metrics["R2"]:.4f})', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_vorhersage_vs_tatsaechlich.png', dpi=100, bbox_inches='tight')
print("\n✓ Grafik gespeichert: 03_vorhersage_vs_tatsaechlich.png")
plt.close()

# =============================================================================
# 6. FEATURE IMPORTANCE
# =============================================================================
print("\n6️⃣  FEATURE IMPORTANCE (Random Forest)")
print("-" * 70)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + feature_importance.to_string(index=False))

plt.figure(figsize=(10, 6))
bars = plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
plt.xlabel('Wichtigkeit', fontsize=12, fontweight='bold')
plt.title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
             ha='left', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('04_feature_importance.png', dpi=100, bbox_inches='tight')
print("\n✓ Grafik gespeichert: 04_feature_importance.png")
plt.close()

# =============================================================================
# 7. RESIDUEN-ANALYSE
# =============================================================================
print("\n7️⃣  RESIDUEN-ANALYSE")
print("-" * 70)

lr_residuals = y_test - lr_pred_test
rf_residuals = y_test - rf_pred_test

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear Regression Residuen
axes[0].scatter(lr_pred_test, lr_residuals, alpha=0.6, color='steelblue', s=50)
axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0].set_xlabel('Vorhergesagter Umsatz', fontsize=12)
axes[0].set_ylabel('Residuen', fontsize=12)
axes[0].set_title('Linear Regression - Residuenplot', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Random Forest Residuen
axes[1].scatter(rf_pred_test, rf_residuals, alpha=0.6, color='green', s=50)
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Vorhergesagter Umsatz', fontsize=12)
axes[1].set_ylabel('Residuen', fontsize=12)
axes[1].set_title('Random Forest - Residuenplot', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_residuen_analyse.png', dpi=100, bbox_inches='tight')
print("✓ Grafik gespeichert: 05_residuen_analyse.png")
plt.close()

print(f"\nLinear Regression:")
print(f"  Residuen Mittelwert: {lr_residuals.mean():>10.2f}")
print(f"  Residuen Std.abw.:   {lr_residuals.std():>10.2f}")
print(f"\nRandom Forest:")
print(f"  Residuen Mittelwert: {rf_residuals.mean():>10.2f}")
print(f"  Residuen Std.abw.:   {rf_residuals.std():>10.2f}")

# =============================================================================
# 8. ZUSAMMENFASSUNG & EMPFEHLUNGEN
# =============================================================================
print("\n" + "=" * 70)
print("PROJEKT-ZUSAMMENFASSUNG: UMSATZ-PROGNOSE")
print("=" * 70)

print("\n📊 DATENSATZ:")
print(f"  • Samples: {len(df)}")
print(f"  • Features: {len(X.columns)}")
print(f"    - Werbebudget (€)")
print(f"    - Lagerbestand (Einheiten)")
print(f"    - Mitarbeiter (Anzahl)")
print(f"    - Kundenanzahl (Anzahl)")
print(f"    - Monat (Saisonalität)")
print(f"  • Target: Umsatz (€)")

print("\n🤖 MODELLE:")
print(f"  • Linear Regression")
print(f"    - Train R²: {lr_train_r2:.4f} | Test R²: {lr_test_r2:.4f}")
print(f"    - Test RMSE: {lr_test_metrics['RMSE']:.2f} €")
print(f"\n  • Random Forest")
print(f"    - Train R²: {rf_train_r2:.4f} | Test R²: {rf_test_r2:.4f}")
print(f"    - Test RMSE: {rf_test_metrics['RMSE']:.2f} €")

print("\n🏆 EMPFEHLUNG:")
if rf_test_metrics['R2'] > lr_test_metrics['R2']:
    improvement = (rf_test_metrics['R2'] - lr_test_metrics['R2']) * 100
    print(f"  ✓ Random Forest ist das bessere Modell!")
    print(f"    - R² Verbesserung: +{improvement:.2f}%")
    print(f"    - Erklärt {rf_test_metrics['R2']*100:.1f}% der Varianz")
    print(f"    - Durchschnittlicher Fehler: {rf_test_metrics['MAE']:.2f} €")
else:
    improvement = (lr_test_metrics['R2'] - rf_test_metrics['R2']) * 100
    print(f"  ✓ Linear Regression ist das bessere Modell!")
    print(f"    - R² Verbesserung: +{improvement:.2f}%")

print(f"\n💡 TOP INSIGHTS:")
print(f"  • Wichtigster Feature: {feature_importance.iloc[0, 0]}")
print(f"    (Importance Score: {feature_importance.iloc[0, 1]:.4f})")
print(f"\n  • Top 3 Features:")
for i, (idx, row) in enumerate(feature_importance.head(3).iterrows(), 1):
    print(f"    {i}. {row['Feature']:20} ({row['Importance']:.4f})")

print("\n📈 AUSGABEDATEIEN:")
print("  ✓ 01_verteilungen.png - Histogramme aller Features")
print("  ✓ 02_korrelationen.png - Korrelationsmatrix Heatmap")
print("  ✓ 03_vorhersage_vs_tatsaechlich.png - Modell Performance")
print("  ✓ 04_feature_importance.png - Feature Relevanz")
print("  ✓ 05_residuen_analyse.png - Fehleranalyse")

print("\n" + "=" * 70)
print("✅ PROJEKT ABGESCHLOSSEN!")
print("=" * 70 + "\n")
