import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV



def search_description(all_descriptions, name_variable):
  # Vérifier si l'index existe
  if name_variable in all_descriptions.index:
    d = all_descriptions.loc[name_variable]
    print(f"{name_variable}: {d['Description']}")
  else:
    print(f"L'index '{name_variable}' n'existe pas dans le DataFrame.")


import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
import numpy as np

def plot_time_series(df, category, categories, numeric_col, col_date="date", scaled=False):
    """
    Plot des séries temporelles groupées par catégorie avec des couleurs distinctes.

    :param df: DataFrame contenant les données.
    :param category: Nom de la colonne catégorielle (ex: 'country').
    :param categories: Liste des catégories uniques à tracer.
    :param numeric_col: Liste des colonnes numériques à tracer.
    :param col_date: Nom de la colonne contenant les dates.
    :param scaled: Booléen pour standardiser les variables (scaling).
    """
    n_subplots = len(categories)
    cols = 4  # Nombre de colonnes dans la grille.
    rows = (n_subplots + cols - 1) // cols  # Calcul du nombre de lignes.

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()

    # Générer une palette de couleurs uniforme à partir d'une colormap
    cmap = cm.get_cmap('tab20', len(numeric_col))  # Colormap pour les couleurs distinctes
    couleurs = [cmap(i) for i in range(len(numeric_col))]

    for i, cat in enumerate(categories):
        if i >= len(axes):  # Vérifie si tous les axes sont remplis.
            break
        
        ax = axes[i]
        df_per_category = df[df[category] == cat]
        
        for var, color in zip(numeric_col, couleurs):
            x = df_per_category[col_date]
            
            if scaled:
                scaler = StandardScaler()
                df_per_category[var] = scaler.fit_transform(df_per_category[[var]])
            
            y = df_per_category[var]
            ax.scatter(x, y, color=color, label=f"{var}", alpha=0.8, s=10)
        
        ax.set_title(f"{category}: {cat}", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(alpha=0.4)
    
    # Supprime les subplots inutilisés (si le nombre total n'est pas un multiple de cols)
    for i in range(n_subplots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_corr(df, highlight_vars=None, highlight_target=None):
  # Calculer la matrice de corrélation
  correlation_matrix = df.corr()

  # Plot de la matrice de corrélation
  plt.figure(figsize=(25, 18))
  ax = sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    cbar=True,
    linewidths=0.5,
    linecolor='gray'
  )
  plt.title("Matrice de Corrélation")
  
  # Encadrer les variables spécifiées
  if highlight_vars:
    for var in highlight_vars:
      if var in correlation_matrix.columns:
        idx = list(correlation_matrix.columns).index(var)
        # Encadrer la colonne entière (ligne horizontale et verticale)
        ax.add_patch(plt.Rectangle((idx, 0), 1, correlation_matrix.shape[0], 
                                    fill=False, edgecolor='black', lw=2))  # Vertical
        ax.add_patch(plt.Rectangle((0, idx), correlation_matrix.shape[1], 1, 
                                    fill=False, edgecolor='black', lw=2))  # Horizontal
  
  # Mettre en rouge les intersections entre highlight_vars et highlight_target
  if highlight_target and highlight_target in correlation_matrix.columns:
    target_idx = list(correlation_matrix.columns).index(highlight_target)
    if highlight_vars:
      for var in highlight_vars:
        if var in correlation_matrix.columns:
          var_idx = list(correlation_matrix.columns).index(var)
          # Rectangle pour l'intersection (ligne <-> colonne rouge)
          ax.add_patch(plt.Rectangle((var_idx, target_idx), 1, 1, 
                                    fill=False, edgecolor='red', lw=3))  # Intersection
  
  plt.show()


def calculate_vif(df, filter_vars=None, target="GDPconstant2015US", plot=False):
  df = df.dropna()
  df.drop([target], axis=1, inplace=True)
  
  if filter_vars:
    df = df[filter_vars]
  
  # Vérifier que le DataFrame ne contient que des variables numériques
  if not all([np.issubdtype(dtype, np.number) for dtype in df.dtypes]):
    raise ValueError("Le DataFrame doit contenir uniquement des variables numériques.")

  # Calculer le VIF pour chaque variable
  vif_data = {
    "Variable": df.columns,
    "VIF": [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
  }

  # Créer un DataFrame avec les résultats
  vif_df = pd.DataFrame(vif_data)
    
  if plot:
    print(vif_df)
  
  return vif_df

def find_highly_correlated(df, threshold=0.8, plot=False):
  corr_matrix = df.corr()
  
  # Prendre la valeur absolue des corrélations
  corr_matrix = corr_matrix.abs()
  
  # Extraire les paires uniques avec corrélation >= seuil
  corr_pairs = [
    (col1, col2, corr_matrix.loc[col1, col2])
    for col1 in corr_matrix.columns
    for col2 in corr_matrix.columns
    if col1 != col2 and corr_matrix.loc[col1, col2] >= threshold
  ]
  
  # Retirer les doublons, car (A, B) et (B, A) apparaissent tous les deux
  corr_pairs = list(set(tuple(sorted(pair[:2])) + (pair[2],) for pair in corr_pairs))
  
  # Trier par force de corrélation décroissante
  corr_pairs.sort(key=lambda x: -x[2])
  
  if plot:
    print(corr_pairs)
  
  return corr_pairs

def get_nan_values_sup_ratio(data, colonne_groupe=None,ratio_nan_values=50, to_display=True):
    """
    Trace les variables (et leurs groupes si spécifié) ayant un pourcentage de valeurs manquantes >= ratio_nan_values%.
    
    Args:
        data (pd.DataFrame): Le DataFrame d'entrée.
        ratio_nan_values (float): Le ratio des NaN values dans le DataFrame
        colonne_groupe (str, optional): Nom de la colonne pour regrouper les données. Si None, pas de regroupement.
    
    Returns:
        pd.DataFrame: Tableau des colonnes ou catégories ayant >= ratio_nan_values% de valeurs manquantes.
    """
    ratio_nan_values = float(ratio_nan_values)
    
    if colonne_groupe:
        # Calcul des valeurs nulles par groupe
        pourcentage_null = data.groupby(colonne_groupe).apply(
            lambda grp: grp.isnull().mean() * 100
        ).T
        # Filtrer les colonnes ayant au moins une catégorie avec >= ratio_nan_values% de valeurs manquantes
        pourcentage_null = pourcentage_null.loc[(pourcentage_null >= ratio_nan_values).any(axis=1)]
    else:
        # Calcul global des pourcentages de valeurs nulles
        pourcentage_null = data.isnull().mean() * 100
        # Filtrer les colonnes avec >= ratio_nan_values% de valeurs nulles
        pourcentage_null = pourcentage_null[pourcentage_null >= ratio_nan_values]

    # Vérifier si des variables à afficher existent
    if pourcentage_null.empty:
        print(f"Aucune variable ou catégorie avec >= {ratio_nan_values}% de valeurs manquantes.")
        return pd.DataFrame()

    if to_display:
      # Tracé des résultats
      if colonne_groupe:
          pourcentage_null.plot(kind='bar', figsize=(12, 6))
          plt.title(f"Variables avec >= {ratio_nan_values}% de valeurs nulles (groupé par catégorie)")
      else:
          pourcentage_null.sort_values(ascending=False).plot(kind='bar', figsize=(12, 6))
          plt.title(f"Variables avec >= {ratio_nan_values}% de valeurs nulles")
      
      plt.ylabel("Pourcentage de valeurs nulles (%)")
      plt.xlabel("Colonnes" if not colonne_groupe else "Colonnes et catégories")
      plt.xticks(rotation=45, ha='right')
      plt.grid(axis='y', linestyle='--', alpha=0.7)
      plt.tight_layout()
      plt.show()
      
      print(pourcentage_null)

    return pourcentage_null


def run_ols_for_significance_var(df, target, explanatory_variables, drop_nan=True, scaled=True, display=True):
  if drop_nan:
    df.dropna(inplace=True)

  if scaled:
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    # Convertir le résultat en DataFrame avec les mêmes colonnes
    df = pd.DataFrame(X, columns=df.columns)

  X = sm.add_constant(df[explanatory_variables])
  model = sm.OLS(df[target], X).fit()
  
  if print:
    print(model.summary())
  
  return model


def drop_cols_from_dataframe(df, cols_to_drop, verbose=True):
  for col in cols_to_drop:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
  
  if verbose:
    print(df.columns)

  return df

def split_columns_by_type(df, verbose=True):
    """
    Sépare les colonnes d'un DataFrame en numériques et catégorielles.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.

    Returns:
        tuple: Une paire contenant la liste des colonnes numériques et la liste des colonnes catégorielles.
    """
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()
    
    if verbose:
      print(f"numeric_columns = {numeric_columns} \ncategorial_columns = {categorical_columns}")
    
    return numeric_columns, categorical_columns


def run_comparative_ols(df, target, pair_var_to_check):
  
  print(f"OLS target1 ~ {pair_var_to_check}")
  run_ols_for_significance_var(df, target, pair_var_to_check)

  for i in range(len(pair_var_to_check)):
    print(f"\n\nOLS target1 ~ {pair_var_to_check[i]}")
    run_ols_for_significance_var(df, target, pair_var_to_check[i])
  
  print("-------------------------------------------------")
  
  print("\n\nOLS target1 ~ all variables")
  run_ols_for_significance_var(df, target, df.columns)



def impute_nan_values_time_series(df, date_col_name='year'):
  countries = df['country'].unique()
  data_interpolated = []

  for c in countries:
    df_per_country = df[df['country'] == c]

    # Identifier les années existantes et la plage d'années complète
    all_years = pd.DataFrame({date_col_name: range(df_per_country[date_col_name].min(), df_per_country[date_col_name].max() + 1)})
    all_years['country'] = c

    # Fusionner les années complètes avec les données existantes
    df_full = pd.merge(all_years, df_per_country, on=[date_col_name, 'country'], how='left')

    # Interpolation des valeurs manquantes
    df_full = df_full.sort_values(by=date_col_name).reset_index(drop=True)
    df_full.iloc[:, 2:] = df_full.iloc[:, 2:].interpolate(method='linear')  # Interpolation des colonnes numériques

    data_interpolated.append(df_full)

  # Concaténer tous les résultats pour chaque pays
  final_df = pd.concat(data_interpolated, ignore_index=True)
  return final_df

def plot_diff_time_series_interpolated(df_original, df_interpolated, var_to_plot='GDPconstant2015US', date_col_name='year'):
  countries = df_original['country'].unique()  # Obtenir les pays uniques
  n_countries = len(countries)
  
  # Calculer le nombre de lignes nécessaires (4 colonnes par ligne)
  n_cols = 4
  n_rows = math.ceil(n_countries / n_cols)

  # Configurer les subplots
  fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), sharex=False, sharey=False)
  axes = axes.flatten()  # Aplatir les axes pour pouvoir itérer facilement

  for i, country in enumerate(countries):
    # Filtrer les données pour chaque pays
    original_country_data = df_original[df_original['country'] == country]
    interpolated_country_data = df_interpolated[df_interpolated['country'] == country]

    # Tracer la série originale (en bleu) et interpolée (en rouge)
    axes[i].scatter(
      original_country_data[date_col_name], 
      original_country_data[var_to_plot], 
      label='Série originale', 
      marker='o', 
      color='blue'
    )
    axes[i].plot(
      interpolated_country_data[date_col_name], 
      interpolated_country_data[var_to_plot], 
      label='Série interpolée', 
      marker='x', 
      color='red', 
      linestyle='-'
    )

    # Ajouter des labels, une légende et un titre spécifique pour chaque pays
    axes[i].set_title(f'{country} - {var_to_plot}')
    axes[i].set_xlabel(date_col_name)
    axes[i].set_ylabel(var_to_plot)
    axes[i].legend()
    axes[i].grid(True)

  # Supprimer les subplots inutilisés si le nombre de pays n'est pas un multiple de 4
  for j in range(n_countries, len(axes)):
    fig.delaxes(axes[j])

  # Ajuster l'espacement entre les graphiques
  plt.tight_layout()
  plt.show()


def align_date_ranges(df, category, date_col, variables):
  """
  Synchronise les plages de dates pour toutes les variables dans chaque catégorie (par pays).
  Pour chaque pays, on aligne les dates de manière à ce que toutes les variables aient des valeurs.
  
  :param df: DataFrame contenant les données.
  :param category: Colonne catégorielle (ex: 'country').
  :param date_col: Colonne de dates.
  :param variables: Liste des colonnes numériques à synchroniser.
  :return: DataFrame filtré avec les plages de dates alignées pour chaque pays.
  """
  aligned_dfs = []
  
  for cat in df[category].unique():
    df_per_category = df[df[category] == cat]

    # Trouver la plage de dates commune où toutes les variables ont des valeurs non-NaN
    valid_dates = set(df_per_category[date_col])
    for var in variables:
      valid_dates &= set(df_per_category[df_per_category[var].notna()][date_col])
    
    # Filtrer les lignes en fonction des dates communes
    df_filtered = df_per_category[df_per_category[date_col].isin(valid_dates)]
    aligned_dfs.append(df_filtered)
  
  # Recombiner tous les DataFrames alignés
  return pd.concat(aligned_dfs, ignore_index=True)

def train_test_with_cv(
    model,
    param_grid,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name="Model",
    scoring="roc_auc",  # or "balanced_accuracy"
    cv=5,
    n_jobs=-1,
    verbose=1,
    plot_roc=True,
    threshold=None  # New parameter for custom threshold
):
    # Initialize GridSearchCV
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True  # Best params re-fitted automatically
    )

    # Fit on the entire training set (CV happens internally)
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    print(f"Best params ({model_name}): {search.best_params_}")
    print(f"Best CV {scoring} Score: {search.best_score_:.3f}\n")
    print("Class ordering:", best_model.classes_)
    
    # Determine if model uses decision_function or predict_proba
    if hasattr(best_model, "predict_proba"):
        pos_index = list(best_model.classes_).index(1)
        y_train_proba = best_model.predict_proba(X_train)[:, pos_index]
        y_test_proba = best_model.predict_proba(X_test)[:, pos_index]
    elif hasattr(best_model, "decision_function"):
        # Use decision function for thresholding
        decision_train = best_model.decision_function(X_train)
        y_train_proba = (decision_train - decision_train.min()) / (decision_train.max() - decision_train.min())
        
        decision_test = best_model.decision_function(X_test)
        y_test_proba = (decision_test - decision_test.min()) / (decision_test.max() - decision_test.min())
    else:
        raise AttributeError("Model does not have predict_proba or decision_function methods.")
    
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    train_auc = roc_auc_score(y_train, y_train_proba)
    train_bal_acc = balanced_accuracy_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    
    # Print metrics
    print(f"{model_name} Train AUC: {train_auc:.3f}")
    print(f"{model_name} Train Balanced Acc: {train_bal_acc:.3f}")
    print(f"{model_name} Train Recall: {train_recall:.3f}")
    print(f"{model_name} Test AUC: {test_auc:.3f}")
    print(f"{model_name} Test Balanced Acc: {test_bal_acc:.3f}")
    print(f"{model_name} Test Recall: {test_recall:.3f}\n")
    
    # Plot ROC curves
    if plot_roc:
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    
        plt.figure(figsize=(8,6))
        plt.plot(fpr_train, tpr_train, label=f"Train (AUC={auc(fpr_train, tpr_train):.3f})")
        plt.plot(fpr_test, tpr_test, label=f"Test (AUC={auc(fpr_test, tpr_test):.3f})")
        plt.plot([0,1],[0,1], "--", color="gray")
        plt.title(f"{model_name} ROC curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend(loc="lower right")
        plt.show()
    
    # Handle custom threshold if provided
    if threshold is not None:
        # For SVM without predict_proba, use decision function
        if hasattr(best_model, "decision_function"):
            y_test_custom = (best_model.decision_function(X_test) > 0).astype(int)
        else:
            y_test_custom = (y_test_proba > threshold).astype(int)
        
        # Calculate metrics based on custom threshold
        recall_custom = recall_score(y_test, y_test_custom)
        precision_custom = precision_score(y_test, y_test_custom)
        f1_custom = f1_score(y_test, y_test_custom)
        roc_auc_custom = roc_auc_score(y_test, y_test_proba)  # AUC remains based on probabilities
    
        print(f"Metrics with Custom Threshold ({threshold}):")
        print(f"Recall: {recall_custom:.3f}")
        print(f"Precision: {precision_custom:.3f}")
        print(f"F1 Score: {f1_custom:.3f}")
        print(f"ROC AUC: {roc_auc_custom:.3f}\n")
    
        # Optional: Plot confusion matrix or other relevant plots
        cm = confusion_matrix(y_test, y_test_custom)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"{model_name} Confusion Matrix at Threshold {threshold}")
        plt.show()
    
        if hasattr(best_model, "decision_function"):
            # For SVM, compare custom threshold with default predict
            y_test_custom_compare = (best_model.decision_function(X_test) > 0).astype(int)
            print("Default Predictions:", y_test_pred[:10])
            print("Custom Threshold Predictions:", y_test_custom_compare[:10])
            print("Are they identical?", np.array_equal(y_test_pred, y_test_custom_compare))
        else:
            print("Default and Custom threshold predictions may differ based on threshold logic.")
    
    return best_model
