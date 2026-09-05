# ============================================================
# src/utils.py
# Fonctions réutilisables pour tout le projet Olist
# Importé par tous les notebooks avec : from src.utils import *
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------
# AUDIT & INSPECTION
# ------------------------------------------------------------

def audit_table(name, df):
    """
    Affiche un résumé complet d'une table :
    dimensions, types, nulls, aperçu.
    Utilisé dans : 01_data_cleaning
    """
    print(f"\n{'='*55}")
    print(f"  TABLE : {name.upper()}  — {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"{'='*55}")
    print(df.dtypes)
    print(f"\nAperçu :")
    print(df.head(2).to_string())


def rapport_nulls(tables_dict):
    """
    Affiche les valeurs manquantes par colonne pour chaque table.
    Paramètre : dict {nom: dataframe}
    Utilisé dans : 01_data_cleaning
    """
    print("=== VALEURS MANQUANTES PAR COLONNE ===\n")
    for name, df in tables_dict.items():
        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0]
        if len(nulls) > 0:
            print(f"--- {name.upper()} ---")
            for col, count in nulls.items():
                pct = count / len(df) * 100
                flag = " ⚠️" if pct > 5 else ""
                print(f"  {col:50} {count:>6} nulls  ({pct:.1f}%){flag}")
            print()
        else:
            print(f"--- {name.upper()} --- ✓ aucun null\n")


# ------------------------------------------------------------
# NETTOYAGE
# ------------------------------------------------------------

def imputer_mediane(df, colonnes):
    """
    Impute les valeurs manquantes avec la médiane.
    Pourquoi médiane : robuste aux outliers (Cours 3 §3).
    Retourne le df modifié + log des imputations.
    Utilisé dans : 01_data_cleaning
    """
    df = df.copy()
    for col in colonnes:
        if df[col].isnull().sum() > 0:
            med = df[col].median()
            df[col] = df[col].fillna(med)
            print(f"✓ {col:35} imputé avec médiane = {med:.2f}")
    return df


def convertir_dates(df, colonnes):
    """
    Convertit les colonnes spécifiées en datetime.
    errors='coerce' transforme les invalides en NaT.
    Utilisé dans : 01_data_cleaning
    """
    df = df.copy()
    for col in colonnes:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    print(f"✓ {len(colonnes)} colonne(s) converties en datetime")
    return df


def convertir_categories(df, colonnes):
    """
    Convertit les colonnes nominales en dtype 'category'.
    Cours 3 §2 : variables nominales — pas de calcul numérique.
    Utilisé dans : 01_data_cleaning, 03_feature_engineering
    """
    df = df.copy()
    for col in colonnes:
        df[col] = df[col].astype('category')
        print(f"✓ {col} → category ({df[col].nunique()} valeurs uniques)")
    return df


# ------------------------------------------------------------
# DÉTECTION D'OUTLIERS
# ------------------------------------------------------------

def detecter_outliers_iqr(df, colonne, facteur=1.5):
    """
    Détecte les outliers par la méthode IQR.
    Cours 1 P3B + Cours 3 §4 — méthode robuste pour données asymétriques.
    Retourne : df avec colonne '{colonne}_is_outlier' ajoutée + stats.
    Utilisé dans : 01_data_cleaning, 02_eda
    """
    df = df.copy()
    Q1  = df[colonne].quantile(0.25)
    Q3  = df[colonne].quantile(0.75)
    IQR = Q3 - Q1

    borne_basse = Q1 - facteur * IQR
    borne_haute = Q3 + facteur * IQR

    flag_col = f"{colonne}_is_outlier"
    df[flag_col] = (df[colonne] < borne_basse) | (df[colonne] > borne_haute)

    n = df[flag_col].sum()
    pct = n / len(df) * 100

    print(f"=== IQR — {colonne} ===")
    print(f"  Q1={Q1:.2f}  Q3={Q3:.2f}  IQR={IQR:.2f}")
    print(f"  Bornes : [{borne_basse:.2f} ; {borne_haute:.2f}]")
    print(f"  Outliers : {n} ({pct:.2f}%)\n")

    return df, borne_basse, borne_haute


def detecter_outliers_zscore(df, colonne, seuil=3):
    """
    Détecte les outliers par Z-score.
    Cours 1 P3A — à utiliser sur distributions proches de la normale.
    ATTENTION : moins fiable sur données skewed (ex : prix).
    Utilisé dans : 01_data_cleaning (comparaison avec IQR)
    """
    from scipy import stats
    df = df.copy()
    z = np.abs(stats.zscore(df[colonne].dropna()))
    flag_col = f"{colonne}_zscore_outlier"
    df[flag_col] = False
    df.loc[df[colonne].dropna().index, flag_col] = z > seuil

    n = df[flag_col].sum()
    print(f"=== Z-score — {colonne} (seuil={seuil}) ===")
    print(f"  Outliers : {n} ({n/len(df)*100:.2f}%)\n")

    return df


# ------------------------------------------------------------
# VISUALISATION
# ------------------------------------------------------------

def plot_boxplot_hist(df, colonne, save_path=None, xlim=None):
    """
    Trace boxplot + histogramme côte à côte pour une colonne.
    Cours 1 P2A (boxplot) + P2B (histogramme).
    Utilisé dans : 01_data_cleaning, 02_eda
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot
    axes[0].boxplot(df[colonne].dropna(), patch_artist=True,
                    boxprops=dict(facecolor='#1D9E75', alpha=0.6))
    axes[0].set_title(f'Boxplot — {colonne}')
    axes[0].set_ylabel(colonne)

    # Histogramme avec règle de Sturges
    n = len(df[colonne].dropna())
    bins = int(1 + np.log2(n))  # Cours 1 P2B — règle de Sturges
    axes[1].hist(df[colonne].dropna(), bins=bins,
                 color='#534AB7', edgecolor='white', alpha=0.8)
    axes[1].set_title(f'Histogramme — {colonne} (Sturges: {bins} bins)')
    axes[1].set_xlabel(colonne)
    axes[1].set_ylabel('Fréquence')
    if xlim:
        axes[1].set_xlim(xlim)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Figure sauvegardée : {save_path}")
    plt.show()


def plot_scatter_outliers(df, col_x, col_y,
                          flag_col=None, save_path=None,
                          xlim=None, ylim=None):
    """
    Scatter plot avec mise en évidence des outliers.
    Cours 1 P2C — détection d'anomalies bivariées.
    Utilisé dans : 01_data_cleaning, 02_eda
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if flag_col and flag_col in df.columns:
        normaux  = df[~df[flag_col]]
        outliers = df[df[flag_col]]
        ax.scatter(normaux[col_x], normaux[col_y],
                   alpha=0.2, s=8, color='#534AB7', label='Normal')
        ax.scatter(outliers[col_x], outliers[col_y],
                   alpha=0.6, s=20, color='#E63946', label='Outlier')
        ax.legend()
    else:
        ax.scatter(df[col_x], df[col_y],
                   alpha=0.2, s=8, color='#534AB7')

    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.set_title(f'{col_x} vs {col_y}')
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Figure sauvegardée : {save_path}")
    plt.show()


# ------------------------------------------------------------
# EXPORT
# ------------------------------------------------------------

def exporter_tables(tables_dict, dossier):
    """
    Exporte un dict de dataframes en CSV dans le dossier spécifié.
    Paramètre : {'nom': df, ...}, chemin dossier
    Utilisé dans : 01_data_cleaning, 03_feature_engineering
    """
    import os
    os.makedirs(dossier, exist_ok=True)
    for name, df in tables_dict.items():
        path = f"{dossier}/{name}_clean.csv"
        df.to_csv(path, index=False)
        print(f"✓ {name:15} → {path}")
    print(f"\n✓ {len(tables_dict)} tables exportées dans {dossier}/")