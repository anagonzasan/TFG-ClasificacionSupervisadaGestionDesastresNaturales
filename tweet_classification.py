# =================================================================================================
# |                      Clasificación supervisada de texto en redes sociales                     |
# |                              para la gestión de desastres naturales                           |
# |                                       TRABAJO FIN DE GRADO                                    |
# |                                                                                               |
# |                                   Autora: Ana González Sánchez                                |
# =================================================================================================

#pip install pandas numpy matplotlib seaborn scikit-learn scipy torch transformers statsmodels
#pip install emoji==0.6.0
import os, sys, json, time, copy, random, re, unicodedata, warnings
import pandas as pd
import numpy as np
import re
import unicodedata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import random
import torch
import joblib
from pathlib import Path
warnings.filterwarnings("ignore")

# Rutas para guardado de resultados
base_dir=Path.cwd()
results_dir=base_dir/"resultados_TFG"
fig_dir=results_dir/"figuras"
bin_dir=results_dir/"binario"
multi_dir=results_dir/"multiclase"
model_dir=results_dir/"modelos"
search_dir=results_dir/"busquedas"
log_dir=results_dir/"logs"
for d in [results_dir, fig_dir, bin_dir, multi_dir, model_dir, search_dir, log_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Rutas del dataset y comprobación
dataset_dir=base_dir/"Dataset"/"crisismmd_datasplit_all"
train_path=dataset_dir/"task_humanitarian_text_img_train.tsv"
dev_path=dataset_dir/"task_humanitarian_text_img_dev.tsv"
test_path=dataset_dir/"task_humanitarian_text_img_test.tsv"
if not train_path.exists():
    candidates=list(base_dir.rglob("task_humanitarian_text_img_train.tsv"))
    if candidates:
        dataset_dir=candidates[0].parent
        train_path=dataset_dir/"task_humanitarian_text_img_train.tsv"
        dev_path=dataset_dir/"task_humanitarian_text_img_dev.tsv"
        test_path=dataset_dir/"task_humanitarian_text_img_test.tsv"
if not train_path.exists():
    raise FileNotFoundError(
        "No se encontró task_humanitarian_text_img_train.tsv. "
    )

# Semillas
seed=11
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# =================================================================================================
# FASE 1: Preparación y limpieza
# =================================================================================================

# 1.1.Carga de datos
# ------------------
# Carga del conjunto TRAIN con identificación valores nulos
ruta_train=str(train_path)
df_train=pd.read_csv(
    ruta_train,
    sep='\t',
    na_values=["", "NA", "NaN", "null"]
)
print(f"Dimensiones iniciales del dataset de entrenamiento:\n - {df_train.shape[0]} observaciones\n - {df_train.shape[1]} variables")
print(f"Encabezado del dataset:")
print(df_train.head())

# 1.2. Análisis de consistencia de etiquetas y filtrado de 
#      ruido estocástico (Label Noise)
# ---------------------------------------------------------
match=(df_train['label']==df_train['label_text'])
match_pct=match.mean()*100
print(f"Análisis de Concordancia:")
print(f" - Coincidencia exacta: {match_pct:.2f}%")
print(f" - Discrepancia: {100-match_pct:.2f}%\n")
# Análisis de tablas cruzadas para ver las discrepancias según la categoría
cont_table=pd.crosstab(df_train['label_text'], df_train['label'])
print(cont_table)
# Filtrado de las observaciones discordantes debido a la alta discrepancia
print("Filtro de ruido estocástico (Label Noise):")
df_train=df_train[df_train['label']==df_train['label_text']].copy()
print(f"Dimensiones tras el filtro de discordantes: {df_train.shape[0]} observaciones\n")

# 1.3. Filtrado inicial: selección de variables de interés
#----------------------------------------------------------
# Selección exclusiva de las variables que aportan información (se eliminan los IDs, y todas las relacionadas con las imágenes)
nlp_cols=['event_name', 'tweet_text', 'label_text']
df_text=df_train[nlp_cols].copy()
# Cambio de nombre de la variable original para definirla como la variable respuesta de la clasificación multiclase
df_text.rename(columns={'label_text': 'label_multiclass'}, inplace=True)
# Definición de la variable respuesta de la clasificación binaria
df_text['label_binary']=np.where(
    df_text['label_multiclass']=='not_humanitarian', 'not_informative', 'informative'
)
print("Muestra de las variables respuesta construidas:")
print(df_text[['event_name', 'tweet_text', 'label_binary', 'label_multiclass']].head())

# 1.4.Detección y tratamiento de errores y NA's
# ---------------------------------------------
# Revisión del dominio
binaryclass_cat=['informative',
                 'not_informative'
]

multiclass_cat=['affected_individuals',
                'infrastructure_and_utility_damage',
                'injured_or_dead_people',
                'missing_or_found_people',
                'rescue_volunteering_or_donation_effort',
                'vehicle_damage',
                'other_relevant_information',
                'not_humanitarian'
]

event_cat=['hurricane_harvey',
           'hurricane_irma',
           'hurricane_maria',
           'california_wildfires',
           'mexico_earthquake',
           'iraq_iran_earthquake',
           'srilanka_floods'
]

# Detección de los errores tipográficos y transformación a NA
err_label_binary=~df_text['label_binary'].isin(binaryclass_cat)
err_label_multi=~df_text['label_multiclass'].isin(multiclass_cat)
err_event=~df_text['event_name'].isin(event_cat)
print(f"Errores en etiquetas binarias: {err_label_binary.sum()}")
print(f"Errores en etiquetas multiclase: {err_label_multi.sum()}")
print(f"Errores en nombre del desastre: {err_event.sum()}")
df_text.loc[err_label_binary, 'label_binary']=pd.NA
df_text.loc[err_label_multi, 'label_multiclass']=pd.NA
df_text.loc[err_event, 'event_name']=pd.NA

# Eliminación de NAs y cadenas vacías
print(f"Número total de NAs:")
print(df_text.isnull().sum())
df_text=df_text.dropna(subset=['event_name', 'tweet_text', 'label_binary', 'label_multiclass'])
df_text=df_text[df_text['tweet_text'].str.strip().astype(bool)]

# Conversión de variables categóricas
cols_cat=['event_name', 'label_binary', 'label_multiclass']
for col in cols_cat:
    df_text[col]=df_text[col].astype('category')

print(f"\nDimensiones finales: {df_text.shape[0]} observaciones")
print(df_text.dtypes)

# 1.5. Normalización sintáctica y deduplicación de retweets (RTs)
# ---------------------------------------------------------
# Normalización sintáctica
def normalize_text(texto):
    texto=str(texto).lower()  #Minúsculas
    texto=unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8', 'ignore')  #Decodificación ASCII
    texto=re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)  #Eliminar URLs
    texto=re.sub(r'@\w+', '', texto)  #Eliminar menciones
    texto=re.sub(r'\brt\b', '', texto)  #Eliminar RT
    texto=re.sub(r'[^a-z0-9\s]', '', texto)  #Dominio alfanumérico y espacios
    texto=re.sub(r'\s+', ' ', texto).strip()  #Colapsar espacios
    return texto
df_text['tweet_clean']=df_text['tweet_text'].apply(normalize_text)
# Limpieza de seguridad
df_text=df_text[df_text['tweet_clean'].astype(bool)]

# Deduplicación semántica
df_text=df_text.drop_duplicates(subset=['tweet_clean'], keep='first')
print(f"Dimensiones dataset de entrenamiento final deduplicado: {df_text.shape[0]} observaciones")
for col in ['event_name', 'label_binary', 'label_multiclass']:
    df_text[col]=df_text[col].astype('category')
print(df_text[['tweet_text', 'tweet_clean']].head())

# Dataset de entrenamiento tras la limpieza
df_train_clean=df_text

# 1.6.Automatización del preprocesamiento y partición 
# ---------------------------------------------------
def pipeline_preprocess(ruta):
    # 1.1. Carga inicial
    df=pd.read_csv(ruta, sep='\t', na_values=["", "NA", "NaN", "null"])

    # 1.2. Filtro de ruido estocástico
    df=df[df['label']==df['label_text']].copy()

    # 1.3 Selección y modificación
    df=df[['event_name', 'tweet_text', 'label_text']].copy()
    df.rename(columns={'label_text': 'label_multiclass'}, inplace=True)

    # 1.4. Filtrado Binario
    df['label_binary']=np.where(
        df['label_multiclass']=='not_humanitarian', 'not_informative', 'informative'
    )

    # 1.5. Detección y tratamiento de NAs
    binaryclass_cat=['informative',
                     'not_informative'
    ]
    multiclass_cat=['affected_individuals',
                    'infrastructure_and_utility_damage',
                    'injured_or_dead_people',
                    'missing_or_found_people',
                    'rescue_volunteering_or_donation_effort',
                    'vehicle_damage',
                    'other_relevant_information',
                    'not_humanitarian'
    ]
    event_cat=['hurricane_harvey',
               'hurricane_irma',
               'hurricane_maria',
               'california_wildfires',
               'mexico_earthquake',
               'iraq_iran_earthquake',
               'srilanka_floods'
    ]
    df.loc[~df['label_binary'].isin(binaryclass_cat), 'label_binary']=pd.NA
    df.loc[~df['label_multiclass'].isin(multiclass_cat), 'label_multiclass']=pd.NA
    df.loc[~df['event_name'].isin(event_cat), 'event_name']=pd.NA
    df=df.dropna(subset=['event_name', 'tweet_text', 'label_binary', 'label_multiclass'])

    # 1.6. Normalización y deduplicado (ASCII)
    def normalize_text(texto):
        texto=str(texto).lower()
        texto=unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        texto=re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
        texto=re.sub(r'@\w+', '', texto)
        texto=re.sub(r'\brt\b', '', texto)
        texto=re.sub(r'[^a-z0-9\s]', '', texto)
        texto=re.sub(r'\s+', ' ', texto).strip()
        return texto
    df['tweet_clean']=df['tweet_text'].apply(normalize_text)
    df=df[df['tweet_clean'].astype(bool)]
    df=df.drop_duplicates(subset=['tweet_clean'], keep='first')
    for col in ['event_name', 'label_binary', 'label_multiclass']:
        df[col]=df[col].astype('category')

    return df

# Ejecución del pipeline sobre los archivos de Test y Validación
df_test_clean=pipeline_preprocess(str(test_path))
df_dev_clean=pipeline_preprocess(str(dev_path))
print("\nDIMENSIONES FINALES")
print(f" - Train: {df_train_clean.shape[0]} observaciones")
print(f" - Dev:   {df_dev_clean.shape[0]} observaciones")
print(f" - Test:  {df_test_clean.shape[0]} observaciones")


# =================================================================================================
# FASE 2: Análisis exploratorio (EDA)
# =================================================================================================

# Configuración de los gráficos
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm"
})
sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.2,
    rc={"text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm"
    }
)

# Paleta para la clasificación binaria
binary_colors={'informative': "#62aa77",
               'not_informative': "#cb6760"
}

# Paleta para la clasificación multiclase
multiclass_colors={'affected_individuals': '#8da0cb',
                   'infrastructure_and_utility_damage': '#e78ac3',
                   'injured_or_dead_people': '#a6d854',
                   'missing_or_found_people': "#fcde58",
                   'rescue_volunteering_or_donation_effort': "#f1b252",
                   'vehicle_damage': '#b3b3b3',
                   'other_relevant_information': '#66c2a5',
                   'not_humanitarian': "#fc6262"
}

# 2.1. Análisis de frecuencias a priori y balanceo de clases
# ---------------------------------------------------------
def class_distribution(df):
    fig, axes=plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 2]})

    # 1. Distribución de la clasificación binaria
    prop_bin=df['label_binary'].value_counts(normalize=True)*100
    ax1=sns.barplot(x=prop_bin.index, y=prop_bin.values, ax=axes[0],
                      palette=binary_colors, hue=prop_bin.index, legend=False,
                      dodge=False, width=0.5)
    axes[0].set_title('Distribución de probabilidad - Clasificación binaria', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Clase binaria', fontsize=11)
    axes[0].set_ylabel('Frecuencia relativa (%)', fontsize=11)
    axes[0].set_ylim(0, 100)
    # Probabilidad a priori
    for p in ax1.patches:
        pct=f'{p.get_height():.1f}%'
        ax1.annotate(pct, (p.get_x() + p.get_width()/2., p.get_height()),
                     ha='center', va='bottom', fontsize=11, color='black',
                     xytext=(0, 5), textcoords='offset points')

    # 2. Distribución de la clasificación multiclase
    prop_multi=df['label_multiclass'].value_counts(normalize=True)*100
    ax2=sns.barplot(y=prop_multi.index, x=prop_multi.values, ax=axes[1],
                      palette=multiclass_colors, hue=prop_multi.index, legend=False,
                      dodge=False)
    axes[1].set_title('Distribución de probabilidad - Clasificación multiclase', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Frecuencia relativa (%)', fontsize=11)
    axes[1].set_ylabel('Categoría humanitaria', fontsize=11)
    axes[1].set_xlim(0, 50)
    # Probabilidad a priori
    for p in ax2.patches:
        pct=f'{p.get_width():.1f}%'
        ax2.annotate(pct, (p.get_width(), p.get_y() + p.get_height()/2.),
                     ha='left', va='center', fontsize=11, color='black',
                     xytext=(5, 0), textcoords='offset points')
    plt.tight_layout()
    fig.savefig('imagenes/distribuciones.png')
    plt.close(fig)
class_distribution(df_train_clean)

# 2.2. Análisis por evento
# -----------------------
def conditional_analysis(df):
    fig, axes=plt.subplots(2, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [1, 1.5]})

    # 1. Probabilidad condicional clasificación binaria
    crosstab_bin=pd.crosstab(df['event_name'], df['label_binary'], normalize='index')*100
    sns.heatmap(crosstab_bin, annot=True, fmt=".1f", cmap="coolwarm", ax=axes[0],
                cbar_kws={'label': 'Probabilidad condicional (%)'}, linewidths=.5)
    axes[0].set_title('Probabilidad condicional P(Y|X) - Clasificación binaria', fontsize=11, fontweight='bold', pad=15)
    axes[0].set_ylabel('Desastre natural', fontsize=11)
    axes[0].set_xlabel('Clase binaria', fontsize=11)
    axes[0].tick_params(axis='x', rotation=0)

    # 2. Probabilidad condicional clasificación multiclase
    crosstab_multi=pd.crosstab(df['event_name'], df['label_multiclass'], normalize='index')*100
    sns.heatmap(crosstab_multi, annot=True, fmt=".1f", cmap="coolwarm", ax=axes[1],
                cbar_kws={'label': 'Probabilidad condicional (%)'}, linewidths=.5)
    axes[1].set_title('Probabilidad condicional P(Y|X) - Clasificación multiclase', fontsize=11, fontweight='bold', pad=15)
    axes[1].set_ylabel('Desastre natural', fontsize=11)
    axes[1].set_xlabel('Categoría humanitaria', fontsize=11)

    # Rotación de las etiquetas X
    axes[1].tick_params(axis='x', rotation=45)
    for tick in axes[1].get_xticklabels():
        tick.set_horizontalalignment('right')
    plt.tight_layout()
    fig.savefig('imagenes/condicional.png')
    plt.close(fig)
conditional_analysis(df_train_clean)

# 2.3. Medidas de asociación - V de Cramer
# ----------------------------------------
# Función matemática para calcular la V de Cramer con corrección de sesgo
def cramers_v(x, y):
    confusion_matrix=pd.crosstab(x, y)
    chi2=ss.chi2_contingency(confusion_matrix)[0]
    n=confusion_matrix.sum().sum()
    phi2=chi2/n
    r, k=confusion_matrix.shape

    # Aplicación de la corrección de sesgo 
    phi2corr=max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr=r - ((r-1)**2)/(n-1)
    kcorr=k - ((k-1)**2)/(n-1)

    # Detección de las divisiones por cero
    min_dim=min((kcorr-1), (rcorr-1))
    if min_dim==0:
        return 0.0
    return np.sqrt(phi2corr/min_dim)

# Construcción de la matriz de asociación
variables=['event_name', 'label_multiclass', 'label_binary']
matriz_cramer=pd.DataFrame(np.zeros((len(variables), len(variables))),
                           index=variables, columns=variables)

for col1 in variables:
    for col2 in variables:
        matriz_cramer.loc[col1, col2]=cramers_v(df_train_clean[col1], df_train_clean[col2])

# Visualización mediante heatmap
fig, ax=plt.subplots(figsize=(8, 6))
sns.heatmap(matriz_cramer, annot=True, fmt=".3f", cmap="coolwarm", vmin=0, vmax=1,
            cbar_kws={'label': 'V de Cramer'},
            linewidths=.5, ax=ax)

ax.set_title('Matriz de asociación categórica', fontsize=11, fontweight='bold', pad=15)
ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', rotation=0)
for tick in ax.get_xticklabels():
    tick.set_horizontalalignment('right')
plt.tight_layout()
fig.savefig('imagenes/vcramer.png')
plt.close(fig)


# =================================================================================================
# FASE 3: Modelización comparativa
# =================================================================================================

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.preprocessing import label_binarize

# 3.1.Tratamiento del desbalanceo (Pesos de clase)
# ------------------------------------------------
# Mapeo de las etiquetas binarias a formato numérico (0 y 1) para las métricas PR-AUC
# 1: informative, 0: not_informative
df_train_clean['label_binary_num']=(df_train_clean['label_binary'] == 'informative').astype(int)
df_dev_clean['label_binary_num']=(df_dev_clean['label_binary'] == 'informative').astype(int)
df_test_clean['label_binary_num']=(df_test_clean['label_binary'] == 'informative').astype(int)

def class_weights(df):
    # Pesos Binaria
    class_bin=np.unique(df['label_binary_num'])
    weights_bin_array=compute_class_weight(class_weight='balanced', classes=class_bin, y=df['label_binary_num'])
    dict_weights_bin=dict(zip(class_bin, weights_bin_array))

    # Pesos Multiclase
    class_multi=np.unique(df['label_multiclass'])
    weights_multi_array=compute_class_weight(class_weight='balanced', classes=class_multi, y=df['label_multiclass'])
    dict_weights_multi=dict(zip(class_multi, weights_multi_array))
    return dict_weights_bin, dict_weights_multi

dict_weights_bin, dict_weights_multi=class_weights(df_train_clean)

# 3.2. Estrategia de validación y búsqueda de hiperparámetros
# ---------------------------------------------------------
# Conversión a listas de Python y arrays de Numpy para evitar errores de GridSearchCV
X_train_val=pd.concat([df_train_clean['tweet_clean'], df_dev_clean['tweet_clean']]).tolist()
y_train_val_bin=pd.concat([df_train_clean['label_binary_num'], df_dev_clean['label_binary_num']]).values
y_train_val_multi=pd.concat([df_train_clean['label_multiclass'], df_dev_clean['label_multiclass']]).values

# PredefinedSplit: Marcamos el Train con -1 (no se usa para validación) y Dev con 0.
test_fold=np.concatenate([np.full(df_train_clean.shape[0], -1),
                         np.zeros(df_dev_clean.shape[0])])
cv_split=PredefinedSplit(test_fold)

# Hiperespacio de búsqueda con C, kernel y gamma
hiperparameters_svm={
    'svm__C': [0.1, 1.0, 10.0],
    'svm__kernel': ['linear', 'rbf'],
    'svm__gamma': ['scale', 'auto', 0.1, 0.01]
}

# 3.3. Modelo 1: Baseline Clásico (TF-IDF + SVM)
# ----------------------------------------------
X_train_strict=df_train_clean['tweet_clean'].tolist()
y_train_bin_strict=df_train_clean['label_binary_num'].values
y_train_multi_strict=df_train_clean['label_multiclass'].values

# ----- CLASIFICACIÓN BINARIA -----

# Definición del pipeline binario TF-IDF + SVM
pipeline_svm_bin=Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('svm', SVC(class_weight=dict_weights_bin, random_state=11))
])

# Malla de hiperparámetros
grid_svm_bin=GridSearchCV(
    estimator=pipeline_svm_bin,
    param_grid=hiperparameters_svm,
    cv=cv_split,
    scoring='average_precision',  # PR-AUC de Scikit-Learn
    error_score='raise',          # Detección de errores
    n_jobs=None,
    verbose=3,
    refit=False
)

print("\n--- OPTIMIZACIÓN HIPERPLANO BINARIO ---\n")
grid_svm_bin.fit(X_train_val, y_train_val_bin)
print("\n--- RESULTADOS SVM BINARIO ---")
print(f"Mejor combinación de hiperparámetros: {grid_svm_bin.best_params_}")
print(f"Mejor rendimiento en Dev (PR-AUC): {grid_svm_bin.best_score_:.4f}")

# ----- CLASIFICACIÓN MULTICLASE -----

# Definición del pipeline multiclase TF-IDF + SVM
pipeline_svm_multi=Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('svm', SVC(class_weight=dict_weights_multi, decision_function_shape='ovr', random_state=11))
])
# Forzamos un array estricto con las categorías para no desalinear la matriz en el scorer
ordered_classes=np.unique(y_train_val_multi)
# kwargs para absorber cualquier parámetro residual de Scikit-Learn
def pr_auc_macro_multiclass(y_true, y_score, **kwargs):
    y_true_bin=label_binarize(y_true, classes=ordered_classes)
    return average_precision_score(y_true_bin, y_score, average='macro')
# Uso de response_method='decision_function'
pr_auc_scorer_multi=make_scorer(pr_auc_macro_multiclass, response_method='decision_function')

# Malla de hiperparámetros
grid_svm_multi=GridSearchCV(
    estimator=pipeline_svm_multi,
    param_grid=hiperparameters_svm,
    cv=cv_split,
    scoring=pr_auc_scorer_multi,  # PR-AUC de Scikit-Learn
    error_score='raise',          # Detección de errores
    n_jobs=None,
    verbose=3,
    refit=False
)

print("\n--- OPTIMIZACIÓN HIPERPLANO MULTICLASE ---\n")
grid_svm_multi.fit(X_train_val, y_train_val_multi)
print("\n--- RESULTADOS SVM MULTICLASE ---")
print(f"Mejor combinación de hiperparámetros: {grid_svm_multi.best_params_}")
print(f"Mejor rendimiento en Dev (PR-AUC-Macro): {grid_svm_multi.best_score_:.4f}")

# ----- FINE-TUNING -----
# 1. Redefinición de los Scorers
# Scorer Binario
pr_auc_scorer_bin=make_scorer(average_precision_score, response_method='decision_function')
# Scorer Multiclase
ordered_classes=np.unique(y_train_val_multi)
def pr_auc_macro_multiclass(y_true, y_score, **kwargs):
    y_true_bin=label_binarize(y_true, classes=ordered_classes)
    return average_precision_score(y_true_bin, y_score, average='macro')
pr_auc_scorer_multi=make_scorer(pr_auc_macro_multiclass, response_method='decision_function')

# 2. Fine-Tuning de hiperparámetros
# Rangos continuos
# Rango continuo binario: [0.5, 2.0] con salto de 0.1
c_range_bin=np.arange(0.5, 2.1, 0.1)
# Rango continuo multiclase: [7.0, 15.0] con salto de 0.5
c_range_multi=np.arange(7.0, 15.5, 0.5)
# Mallas de búsqueda
fine_grid_bin=[
    {'svm__C': c_range_bin, 'svm__kernel': ['linear']},
    {'svm__C': c_range_bin, 'svm__kernel': ['rbf'], 'svm__gamma': ['scale', 0.05, 0.1, 0.2]}
]
fine_grid_multi=[
    {'svm__C': c_range_multi, 'svm__kernel': ['linear']},
    {'svm__C': c_range_multi, 'svm__kernel': ['rbf'], 'svm__gamma': ['scale', 0.08, 0.1, 0.15]}
]

# 3. Optimización binaria
grid_svm_bin_fine=GridSearchCV(
    estimator=pipeline_svm_bin,
    param_grid=fine_grid_bin,
    cv=cv_split,
    scoring=pr_auc_scorer_bin,
    error_score='raise',
    n_jobs=None,
    verbose=1,
    refit=False
)
print("\n--- FINE-TUNING DEL HIPERPLANO BINARIO ---\n")
grid_svm_bin_fine.fit(X_train_val, y_train_val_bin)
print(f" -> Mejor configuración Binaria: {grid_svm_bin_fine.best_params_}")
print(f" -> Nuevo PR-AUC: {grid_svm_bin_fine.best_score_:.4f}")

# 4. Optimización multiclase
grid_svm_multi_fine=GridSearchCV(
    estimator=pipeline_svm_multi,
    param_grid=fine_grid_multi,
    cv=cv_split,
    scoring=pr_auc_scorer_multi,
    error_score='raise',
    n_jobs=None,
    verbose=1,
    refit=False
)

print("\n--- FINE-TUNING DEL HIPERPLANO MULTICLASE ---\n")
grid_svm_multi_fine.fit(X_train_val, y_train_val_multi)
print(f" -> Mejor configuración Multiclase: {grid_svm_multi_fine.best_params_}")
print(f" -> Nuevo PR-AUC-Macro: {grid_svm_multi_fine.best_score_:.4f}")

# 5. Entrenamiento sobre Train
svm_bin_final=Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('svm', SVC(class_weight=dict_weights_bin, random_state=11))
])
svm_bin_final.set_params(**grid_svm_bin_fine.best_params_)
svm_bin_final.fit(X_train_strict, y_train_bin_strict)

svm_multi_final=Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('svm', SVC(class_weight=dict_weights_multi, decision_function_shape='ovr', random_state=11))
])
svm_multi_final.set_params(**grid_svm_multi_fine.best_params_)
svm_multi_final.fit(X_train_strict, y_train_multi_strict)

# 6. Copia de seguridad de los modelos entrenados
joblib.dump(svm_bin_final, model_dir/'modelo_svm_binario_final.pkl')
joblib.dump(svm_multi_final, model_dir/'modelo_svm_multiclase_final.pkl')


# 3.4. Modelo 2: Deep Learning 
#     (Redes neuronales Text-CNN y Bi-LSTM)
# ---------------------------------------------------------
import os
import urllib.request
import zipfile
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import copy
import numpy as np

#Descarga de GloVe si no existe
glove_url="https://nlp.stanford.edu/data/glove.6B.zip"
glove_zip=str(base_dir/"glove.6B.zip")
glove_txt=str(base_dir/"glove.6B.100d.txt")

if not os.path.exists(glove_txt):
    print("Descargando modelo GloVe desde Stanford NLP (822 MB en formato ZIP)")
    urllib.request.urlretrieve(glove_url, glove_zip)
    print("Descarga completada. Descomprimiendo archivos")
    with zipfile.ZipFile(glove_zip, 'r') as zip_ref:
        zip_ref.extractall()
    print("El archivo 'glove.6B.100d.txt' ya está listo para usarse.")
    # Limpiamos el archivo zip para ahorrar espacio en tu disco
    os.remove(glove_zip)
else:
    print("El archivo GloVe ya está presente en el directorio listo para usarse.")

# 3.4.1. Construcción del espacio vectorial y DataLoaders
# 1. Vocabulario y diccionarios
tokens_train=[tweet.split() for tweet in df_train_clean['tweet_clean']]
freq=Counter()
for tweet in tokens_train:
    freq.update(tweet)
vocab={'<PAD>': 0, '<UNK>': 1}
idx=2
for word, frec in freq.items():
    if frec >= 2:
        vocab[word]=idx
        idx += 1
print(f"Tamaño del espacio de vocabulario útil: {len(vocab)} dimensiones.")

# 2. Padding y tensores
def text2index(token_list, vocab):
    return [vocab.get(word, vocab['<UNK>']) for word in token_list]

L_MAX=30
def nlp_tensors_prep(df, vocab, max_len):
    tokens=[str(tweet).split() for tweet in df['tweet_clean']]
    seqs=[torch.tensor(text2index(t, vocab)) for t in tokens]
    seqs_trunc=[sec[:max_len] for sec in seqs]
    x_pad=pad_sequence(seqs_trunc, batch_first=True, padding_value=vocab['<PAD>'])
    if x_pad.shape[1] < max_len:
        padding_extra=torch.zeros((x_pad.shape[0], max_len-x_pad.shape[1]), dtype=torch.long)
        x_pad=torch.cat((x_pad, padding_extra), dim=1)
    else:
        x_pad=x_pad[:, :max_len]
    return x_pad

x_train_pad=nlp_tensors_prep(df_train_clean, vocab, L_MAX)
x_dev_pad=nlp_tensors_prep(df_dev_clean, vocab, L_MAX)
x_test_pad=nlp_tensors_prep(df_test_clean, vocab, L_MAX)

# Mapeos
binary_map={'not_informative': 0, 'informative': 1}
y_train_num=df_train_clean['label_binary'].map(binary_map)
y_dev_num=df_dev_clean['label_binary'].map(binary_map)
y_test_num=df_test_clean['label_binary'].map(binary_map)
map_multi={cat: i for i, cat in enumerate(multiclass_cat)}
y_train_multi_num=df_train_clean['label_multiclass'].map(map_multi)
y_dev_multi_num=df_dev_clean['label_multiclass'].map(map_multi)
y_test_multi_num=df_test_clean['label_multiclass'].map(map_multi)

class DisasterTweetsDataset(Dataset):
    def __init__(self, secuencias_numericas, etiquetas):
        self.x=secuencias_numericas.clone().detach().to(torch.long)
        self.y=torch.tensor(etiquetas.values, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

# 3. Generación de batches
BATCH_SIZE=64

# DataLoaders Binarios
train_loader_bin=DataLoader(DisasterTweetsDataset(x_train_pad, y_train_num), batch_size=BATCH_SIZE, shuffle=True)
train_eval_loader_bin=DataLoader(DisasterTweetsDataset(x_train_pad, y_train_num), batch_size=BATCH_SIZE, shuffle=False) 
dev_loader_bin=DataLoader(DisasterTweetsDataset(x_dev_pad, y_dev_num), batch_size=BATCH_SIZE, shuffle=False)
test_loader_bin=DataLoader(DisasterTweetsDataset(x_test_pad, y_test_num), batch_size=BATCH_SIZE, shuffle=False)

# DataLoaders Multiclase
train_loader_multi=DataLoader(DisasterTweetsDataset(x_train_pad, y_train_multi_num), batch_size=BATCH_SIZE, shuffle=True)
train_eval_loader_multi=DataLoader(DisasterTweetsDataset(x_train_pad, y_train_multi_num), batch_size=BATCH_SIZE, shuffle=False) 
dev_loader_multi=DataLoader(DisasterTweetsDataset(x_dev_pad, y_dev_multi_num), batch_size=BATCH_SIZE, shuffle=False)
test_loader_multi=DataLoader(DisasterTweetsDataset(x_test_pad, y_test_multi_num), batch_size=BATCH_SIZE, shuffle=False)

dispositivo=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3.4.2. Inyección de Word Embeddings (GloVe)
def load_glove_embeddings(glove_path, vocab, embed_dim=100):
    print(f"\nCargando vectores GloVe desde {glove_path}")
    embedding_matrix=np.zeros((len(vocab), embed_dim))
    words_found=0
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values=line.split()
            word=values[0]
            if word in vocab:
                idx=vocab[word]
                embedding_matrix[idx]=np.array(values[1:], dtype='float32')
                words_found += 1
    print(f"Palabras encontradas en GloVe: {words_found}/{len(vocab)}")
    return torch.tensor(embedding_matrix, dtype=torch.float32)

pretrained_weights=load_glove_embeddings('glove.6B.100d.txt', vocab, embed_dim=100)

# 3.4.3. Definición topológica de arquitecturas: Text-CNN y Bi-LSTM

# Modelo 2A. Text-CNN
class TextCNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, padding_idx, pretrained_embeddings=None):
        super(TextCNNClassifier, self).__init__()
        if pretrained_embeddings is not None:
            self.embedding=nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=padding_idx)
            embed_dim=pretrained_embeddings.shape[1]
        else:
            self.embedding=nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.convs=nn.ModuleList([nn.Conv1d(in_channels=embed_dim, out_channels=100, kernel_size=k) for k in [3, 4, 5]])
        self.fc=nn.Linear(300, num_classes)
        self.dropout=nn.Dropout(0.5)

    def forward(self, x):
        embedded=self.embedding(x).permute(0, 2, 1) 
        conved=[F.relu(conv(embedded)) for conv in self.convs]
        pooled=[torch.max(c, dim=2)[0] for c in conved]
        cat=self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

# Modelo 2B. Bi-LSTM
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, padding_idx, pretrained_embeddings=None):
        super(BiLSTMClassifier, self).__init__()
        if pretrained_embeddings is not None:
            self.embedding=nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True, padding_idx=padding_idx)
            embed_dim=pretrained_embeddings.shape[1]
        else:
            self.embedding=nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.lstm=nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc=nn.Linear(hidden_dim * 2, num_classes)
        self.dropout=nn.Dropout(0.3)

    def forward(self, x):
        embedded=self.embedding(x)
        lstm_out, (hidden, cell)=self.lstm(embedded)
        hidden_cat=torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(self.dropout(hidden_cat))

# 3.4.4. Función de optimización guiada por PR-AUC
def plot_learning_curve(train_losses, dev_losses, title):
    fig=plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Loss Entrenamiento', color='#1f77b4', linewidth=2)
    plt.plot(dev_losses, label='Loss Validación', color='#ff7f0e', linewidth=2, linestyle='--')
    plt.title(title, fontweight='bold', pad=15)
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (Entropía cruzada)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    fig.savefig(fig_dir/f"{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def train_dl_model(model, train_loader, dev_loader, criteria, optimizer, num_classes, epochs=30, patience=5, model_name="Modelo DL"):
    best_dev_prauc=0.0
    counter=0
    best_model_wts=None
    train_losses, dev_losses=[], []

    for epoch in range(epochs):
        model.train()
        loss_train=0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch=x_batch.to(dispositivo), y_batch.to(dispositivo)
            optimizer.zero_grad()
            preds=model(x_batch)
            loss=criteria(preds, y_batch)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        model.eval()
        loss_dev=0.0
        all_targets, all_probs=[], []
        with torch.no_grad():
            for x_val, y_val in dev_loader:
                x_val, y_val=x_val.to(dispositivo), y_val.to(dispositivo)
                preds_val=model(x_val)
                loss_dev += criteria(preds_val, y_val).item()
                probs=F.softmax(preds_val, dim=1)
                all_targets.extend(y_val.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        media_train=loss_train/len(train_loader)
        media_dev=loss_dev/len(dev_loader)
        all_probs=np.array(all_probs)

        if num_classes == 2:
            dev_prauc=average_precision_score(all_targets, all_probs[:, 1])
        else:
            targets_bin=label_binarize(all_targets, classes=range(num_classes))
            dev_prauc=average_precision_score(targets_bin, all_probs, average='macro')

        train_losses.append(media_train)
        dev_losses.append(media_dev)

        if dev_prauc > best_dev_prauc:
            best_dev_prauc=dev_prauc
            counter=0
            best_model_wts=copy.deepcopy(model.state_dict())
        else:
            counter += 1

        if (epoch + 1) % 5 == 0:
            print(f"Época {epoch+1:02d}/{epochs} | Loss Train: {media_train:.4f} | PR-AUC Dev: {dev_prauc:.4f}")

        if counter >= patience:
            print(f"-> Early Stopping activado en época {epoch+1}.")
            model.load_state_dict(best_model_wts)
            break
            
    print(f"-> Mejor PR-AUC en validación: {best_dev_prauc:.4f}\n")
    plot_learning_curve(train_losses, dev_losses, f"Curva de Aprendizaje - {model_name}")

    # Copia de seguridad y vaciado de memoria GPU
    torch.save(model.state_dict(), model_dir/f"{model_name.replace(' ', '_')}.pth")
    pd.DataFrame({
        'epoch': np.arange(1, len(train_losses)+1),
        'train_loss': train_losses,
        'dev_loss': dev_losses
    }).to_csv(log_dir/f"{model_name.replace(' ', '_')}_learning_curve.csv", index=False)
    
    import gc
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return model, train_losses, dev_losses

# --- FUNCIÓN DE INFERENCIA PARA EVITAR OOM ---
def extract_dl_probs(model, dataloader):
    model.eval()
    all_probs=[]
    with torch.no_grad():
        for x_batch, _ in dataloader:
            x_batch=x_batch.to(dispositivo)
            logits=model(x_batch)
            probs=F.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)

# 3.4.5. Entrenamiento y estudio de ablación
tensor_weights_binary=torch.tensor([dict_weights_bin[0], dict_weights_bin[1]], dtype=torch.float).to(dispositivo)
criteria_binary=nn.CrossEntropyLoss(weight=tensor_weights_binary)
tensor_weights_multi=torch.tensor([dict_weights_multi[c] for c in multiclass_cat], dtype=torch.float).to(dispositivo)
criteria_multi=nn.CrossEntropyLoss(weight=tensor_weights_multi)

# --- 2A. MODELO TEXT-CNN (CON GLOVE FINE-TUNING) ---
print("\n--- ENTRENAMIENTO TEXT-CNN + GLOVE FINE-TUNING BINARIO ---\n")
cnn_bin=TextCNNClassifier(len(vocab), 100, 2, vocab['<PAD>'], pretrained_weights).to(dispositivo)
best_cnn_bin, _, _=train_dl_model(cnn_bin, train_loader_bin, dev_loader_bin, criteria_binary, optim.Adam(cnn_bin.parameters(), lr=0.001), 2, model_name="Text-CNN Binario")

print("\n--- ENTRENAMIENTO TEXT-CNN + GLOVE FINE-TUNING MUTICLASE ---\n")
cnn_multi=TextCNNClassifier(len(vocab), 100, 8, vocab['<PAD>'], pretrained_weights).to(dispositivo)
best_cnn_multi, _, _=train_dl_model(cnn_multi, train_loader_multi, dev_loader_multi, criteria_multi, optim.Adam(cnn_multi.parameters(), lr=0.001), 8, model_name="Text-CNN Multiclase")

# --- 2B. ESTUDIO DE ABLACIÓN BI-LSTM (DESDE CERO VS GLOVE) ---
print("\n--- ENTRENAMIENTO BI-LSTM DESDE CERO/SCRATCH BINARIO ---\n")
bilstm_scratch_bin=BiLSTMClassifier(len(vocab), 100, 64, 2, vocab['<PAD>'], None).to(dispositivo)
best_bilstm_scratch_bin, _, _=train_dl_model(bilstm_scratch_bin, train_loader_bin, dev_loader_bin, criteria_binary, optim.Adam(bilstm_scratch_bin.parameters(), lr=0.001), 2, model_name="Bi-LSTM SCRATCH Binario")

print("\n--- ENTRENAMIENTO BI-LSTM CON GLOVE FIJO BINARIO ---\n")
bilstm_glove_bin=BiLSTMClassifier(len(vocab), 100, 64, 2, vocab['<PAD>'], pretrained_weights).to(dispositivo)
best_bilstm_glove_bin, _, _=train_dl_model(bilstm_glove_bin, train_loader_bin, dev_loader_bin, criteria_binary, optim.Adam(bilstm_glove_bin.parameters(), lr=0.001), 2, model_name="Bi-LSTM FIJO Binario")

print("\n--- ENTRENAMIENTO BI-LSTM DESDE CERO/SCRATCH MULTICLASE ---\n")
bilstm_scratch_multi=BiLSTMClassifier(len(vocab), 100, 64, 8, vocab['<PAD>'], None).to(dispositivo)
best_bilstm_scratch_multi, _, _=train_dl_model(bilstm_scratch_multi, train_loader_multi, dev_loader_multi, criteria_multi, optim.Adam(bilstm_scratch_multi.parameters(), lr=0.001), 8, model_name="Bi-LSTM SCRATCH Multiclase")

print("\n--- ENTRENAMIENTO BI-LSTM CON GLOVE FIJO MULTICLASE ---\n")
bilstm_glove_multi=BiLSTMClassifier(len(vocab), 100, 64, 8, vocab['<PAD>'], pretrained_weights).to(dispositivo)
best_bilstm_glove_multi, _, _=train_dl_model(bilstm_glove_multi, train_loader_multi, dev_loader_multi, criteria_multi, optim.Adam(bilstm_glove_multi.parameters(), lr=0.001), 8, model_name="Bi-LSTM FIJO Multiclase")


# 3.5. Modelo 3: Transformers (Fine-Tuning con BERT)
# --------------------------------------------------
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from torch.optim import AdamW
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import copy
import numpy as np

# 1. Tokenización WordPiece y tensores de atención
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

class BertTweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts=texts
        self.labels=labels
        self.tokenizer=tokenizer
        self.max_len=max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text=str(self.texts[idx])
        label=self.labels[idx]

        encoding=self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

map_multi={cat: i for i, cat in enumerate(multiclass_cat)}
y_train_multi_num=df_train_clean['label_multiclass'].map(map_multi).values
y_dev_multi_num=df_dev_clean['label_multiclass'].map(map_multi).values
y_test_multi_num=df_test_clean['label_multiclass'].map(map_multi).values
y_train_bin_num=df_train_clean['label_binary_num'].values
y_dev_bin_num=df_dev_clean['label_binary_num'].values
y_test_bin_num=df_test_clean['label_binary_num'].values

MAX_LEN=64
BATCH_SIZE=32

# Binario
train_dataset_bert_bin=BertTweetDataset(df_train_clean['tweet_clean'].values, y_train_bin_num, tokenizer, MAX_LEN)
dev_dataset_bert_bin=BertTweetDataset(df_dev_clean['tweet_clean'].values, y_dev_bin_num, tokenizer, MAX_LEN)
test_dataset_bert_bin=BertTweetDataset(df_test_clean['tweet_clean'].values, y_test_bin_num, tokenizer, MAX_LEN)

train_loader_bert_bin=DataLoader(train_dataset_bert_bin, batch_size=BATCH_SIZE, shuffle=True)
train_eval_loader_bert_bin=DataLoader(train_dataset_bert_bin, batch_size=BATCH_SIZE, shuffle=False) 
dev_loader_bert_bin=DataLoader(dev_dataset_bert_bin, batch_size=BATCH_SIZE, shuffle=False)
test_loader_bert_bin=DataLoader(test_dataset_bert_bin, batch_size=BATCH_SIZE, shuffle=False)

# Multiclase
train_dataset_bert_multi=BertTweetDataset(df_train_clean['tweet_clean'].values, y_train_multi_num, tokenizer, MAX_LEN)
dev_dataset_bert_multi=BertTweetDataset(df_dev_clean['tweet_clean'].values, y_dev_multi_num, tokenizer, MAX_LEN)
test_dataset_bert_multi=BertTweetDataset(df_test_clean['tweet_clean'].values, y_test_multi_num, tokenizer, MAX_LEN)

train_loader_bert_multi=DataLoader(train_dataset_bert_multi, batch_size=BATCH_SIZE, shuffle=True)
train_eval_loader_bert_multi=DataLoader(train_dataset_bert_multi, batch_size=BATCH_SIZE, shuffle=False) 
dev_loader_bert_multi=DataLoader(dev_dataset_bert_multi, batch_size=BATCH_SIZE, shuffle=False)
test_loader_bert_multi=DataLoader(test_dataset_bert_multi, batch_size=BATCH_SIZE, shuffle=False)

dispositivo=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 2. Optimizador
def train_bert_model(model, train_loader, dev_loader, class_weights_tensor, num_classes, epochs=4, model_name="Modelo BERT"):
    model=model.to(dispositivo)
    optimizer=AdamW(model.parameters(), lr=2e-5)
    total_steps=len(train_loader) * epochs
    scheduler=get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    loss_fn=nn.CrossEntropyLoss(weight=class_weights_tensor)

    best_dev_prauc=0.0
    best_model_weights=None

    train_losses, dev_losses=[], [] 

    for epoch in range(epochs):
        model.train()
        total_train_loss=0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids=batch['input_ids'].to(dispositivo)
            attention_mask=batch['attention_mask'].to(dispositivo)
            labels=batch['labels'].to(dispositivo)

            outputs=model(input_ids=input_ids, attention_mask=attention_mask)
            loss=loss_fn(outputs.logits, labels)

            total_train_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss=total_train_loss/len(train_loader)

        # Validación y cálculo de PR-AUC
        model.eval()
        total_dev_loss=0
        all_targets=[]
        all_probs=[]
        with torch.no_grad():
            for batch in dev_loader:
                input_ids=batch['input_ids'].to(dispositivo)
                attention_mask=batch['attention_mask'].to(dispositivo)
                labels=batch['labels'].to(dispositivo)

                outputs=model(input_ids=input_ids, attention_mask=attention_mask)
                loss=loss_fn(outputs.logits, labels)
                total_dev_loss += loss.item()

                probs=F.softmax(outputs.logits, dim=1)
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_dev_loss=total_dev_loss/len(dev_loader)
        all_probs=np.array(all_probs)

        if num_classes > 2:
            targets_bin=label_binarize(all_targets, classes=range(num_classes))
            dev_prauc=average_precision_score(targets_bin, all_probs, average='macro')
        else:
            dev_prauc=average_precision_score(all_targets, all_probs[:, 1])

        print(f"Época {epoch+1}/{epochs} | Loss Dev: {avg_dev_loss:.4f} | PR-AUC Dev: {dev_prauc:.4f}")

        # Selección según el PR-AUC
        if dev_prauc > best_dev_prauc:
            best_dev_prauc=dev_prauc
            best_model_weights=copy.deepcopy(model.state_dict())

        train_losses.append(avg_train_loss) 
        dev_losses.append(avg_dev_loss)     

    model.load_state_dict(best_model_weights)
    print(f"\n-> Entrenamiento finalizado. Mejor PR-AUC en validación: {best_dev_prauc:.4f}")

    plot_learning_curve(train_losses, dev_losses, f"Curva de Aprendizaje - {model_name}")

    # Copia de seguridad de los modelos
    torch.save(model.state_dict(), model_dir/f"{model_name.replace(' ', '_')}.pth")
    pd.DataFrame({
        'epoch': np.arange(1, len(train_losses)+1),
        'train_loss': train_losses,
        'dev_loss': dev_losses
    }).to_csv(log_dir/f"{model_name.replace(' ', '_')}_learning_curve.csv", index=False)
    
    # Limpieza de memoria
    import gc
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return model, train_losses, dev_losses


# 3. Fine-Tuning
print("\n--- FINE-TUNING BERT BINARIO ---\n")
bert_bin=BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tensor_weights_binary=torch.tensor([dict_weights_bin[0], dict_weights_bin[1]], dtype=torch.float).to(dispositivo)
best_bert_bin=train_bert_model(bert_bin, train_loader_bert_bin, dev_loader_bert_bin, tensor_weights_binary, num_classes=2, epochs=3, model_name="BERT Binario")

print("\n--- FINE-TUNING BERT MULTICLASE ---\n")
bert_multi=BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=8)
tensor_weights_multi=torch.tensor([dict_weights_multi[c] for c in multiclass_cat], dtype=torch.float).to(dispositivo)
best_bert_multi=train_bert_model(bert_multi, train_loader_bert_multi, dev_loader_bert_multi, tensor_weights_multi, num_classes=8, epochs=4, model_name="BERT Multiclase")


# 3.6. Optimización del espacio latente: BERTweet
# -----------------------------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Carga y tokenización con BERTweet
# Carga del tokenizador específico para Twitter
tokenizer_tweet=AutoTokenizer.from_pretrained('vinai/bertweet-base')

# Regeneración los datasets multiclase con el nuevo tokenizador
train_dataset_tweet_multi=BertTweetDataset(df_train_clean['tweet_clean'].values, y_train_multi_num, tokenizer_tweet, MAX_LEN)
dev_dataset_tweet_multi=BertTweetDataset(df_dev_clean['tweet_clean'].values, y_dev_multi_num, tokenizer_tweet, MAX_LEN)
test_dataset_tweet_multi=BertTweetDataset(df_test_clean['tweet_clean'].values, y_test_multi_num, tokenizer_tweet, MAX_LEN)

train_loader_tweet_multi=DataLoader(train_dataset_tweet_multi, batch_size=BATCH_SIZE, shuffle=True)
train_eval_loader_tweet_multi=DataLoader(train_dataset_tweet_multi, batch_size=BATCH_SIZE, shuffle=False) 
dev_loader_tweet_multi=DataLoader(dev_dataset_tweet_multi, batch_size=BATCH_SIZE, shuffle=False)
test_loader_tweet_multi=DataLoader(test_dataset_tweet_multi, batch_size=BATCH_SIZE, shuffle=False)

# 2. Fine-Tuning de BERTweet Multiclase
# Carga de la arquitectura
bertweet_multi=AutoModelForSequenceClassification.from_pretrained('vinai/bertweet-base', num_labels=8)

# Reutilización de la función de entrenamiento train_bert_model del 3.5
best_bertweet_multi, t_loss_tw, d_loss_tw=train_bert_model(
    model=bertweet_multi,
    train_loader=train_loader_tweet_multi,
    dev_loader=dev_loader_tweet_multi,
    class_weights_tensor=tensor_weights_multi,
    num_classes=8,
    epochs=4,
    model_name="BERTweet Multiclase"
)


# 3.7. BERTweet + FOCAL LOSS: selección por 
#      PR-AUC-Macro en DEV
# ---------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight=weight
        self.gamma=gamma
        self.reduction=reduction

    def forward(self, inputs, targets):
        ce_loss=F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt=torch.exp(-ce_loss)
        focal_loss=((1-pt)**self.gamma)*ce_loss
        if self.reduction=='mean':
            return focal_loss.mean()
        if self.reduction=='sum':
            return focal_loss.sum()
        return focal_loss

def evaluate_multiclass_probs(model, loader):
    model.eval()
    ys, ps=[], []
    with torch.no_grad():
        for batch in loader:
            ids=batch['input_ids'].to(dispositivo)
            mask=batch['attention_mask'].to(dispositivo)
            out=model(input_ids=ids, attention_mask=mask)
            ys.extend(batch['labels'].cpu().numpy())
            ps.extend(F.softmax(out.logits, dim=1).cpu().numpy())
    return np.asarray(ys), np.asarray(ps)

def train_bert_model_focal(model, train_loader, dev_loader, class_weights_tensor,
                           epochs=4, gamma=2.0, lr=2e-5, model_name="BERTweet Focal"):
    model=model.to(dispositivo)
    optimizer=AdamW(model.parameters(), lr=lr)
    total_steps=len(train_loader)*epochs
    scheduler=get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    loss_fn=FocalLoss(weight=class_weights_tensor, gamma=gamma)

    best_dev_prauc=-np.inf
    best_epoch=0
    best_weights=None
    history=[]

    for epoch in range(epochs):
        model.train()
        total_train=0.0
        for batch in train_loader:
            optimizer.zero_grad()
            ids=batch['input_ids'].to(dispositivo)
            mask=batch['attention_mask'].to(dispositivo)
            labels=batch['labels'].to(dispositivo)
            out=model(input_ids=ids, attention_mask=mask)
            loss=loss_fn(out.logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_train += loss.item()

        yv, pv=evaluate_multiclass_probs(model, dev_loader)
        
        # Pérdida de validación
        model.eval()
        total_dev=0.0
        with torch.no_grad():
            for batch in dev_loader:
                ids=batch['input_ids'].to(dispositivo)
                mask=batch['attention_mask'].to(dispositivo)
                labels=batch['labels'].to(dispositivo)
                out=model(input_ids=ids, attention_mask=mask)
                total_dev += loss_fn(out.logits, labels).item()

        train_loss=total_train/len(train_loader)
        dev_loss=total_dev/len(dev_loader)
        yv_bin=label_binarize(yv, classes=np.arange(8))
        dev_prauc=average_precision_score(yv_bin, pv, average='macro')

        history.append({
            'epoch': epoch+1,
            'train_loss': train_loss,
            'dev_loss': dev_loss,
            'dev_pr_auc_macro': dev_prauc,
            'gamma': gamma,
            'lr': lr
        })
        print(f"Época {epoch+1}/{epochs} | Loss Train: {train_loss:.4f} | "
              f"Loss Dev: {dev_loss:.4f} | PR-AUC-Macro Dev: {dev_prauc:.4f}")

        if dev_prauc > best_dev_prauc:
            best_dev_prauc=dev_prauc
            best_epoch=epoch+1
            best_weights=copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    pd.DataFrame(history).to_csv(
        log_dir/f"{model_name.replace(' ','_')}_learning_curve.csv", index=False
    )
    torch.save(model.state_dict(), model_dir/f"{model_name.replace(' ','_')}.pth")
    with open(log_dir/f"{model_name.replace(' ','_')}_selection.json","w",encoding="utf8") as f:
        json.dump({
            'criterion': 'PR-AUC-Macro on DEV',
            'best_epoch': best_epoch,
            'best_dev_pr_auc_macro': float(best_dev_prauc),
            'gamma': gamma,
            'learning_rate': lr,
            'epochs': epochs
        }, f, indent=2, ensure_ascii=False)
        
    # Limpieza de memoria tras el entrenamiento
    import gc
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    return model, pd.DataFrame(history)

print("\n--- FINE-TUNING BERTWEET + FOCAL LOSS MULTICLASE ---\n")
bertweet_focal=AutoModelForSequenceClassification.from_pretrained(
    'vinai/bertweet-base', num_labels=8
)
best_bertweet_focal, focal_history=train_bert_model_focal(
    model=bertweet_focal,
    train_loader=train_loader_tweet_multi,
    dev_loader=dev_loader_tweet_multi,
    class_weights_tensor=tensor_weights_multi,
    epochs=4,
    gamma=2.0,
    lr=2e-5,
    model_name="BERTweet_Multiclase_Focal"
)


# =================================================================================================
# FASE 4: Inferencia y evaluación final
# =================================================================================================

from sklearn.metrics import (
    classification_report, fbeta_score, confusion_matrix,
    average_precision_score, precision_score, recall_score,
    accuracy_score, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from statsmodels.stats.contingency_tables import mcnemar
import gc

print("\n--- INICIANDO FASE 4: INFERENCIA Y EVALUACIÓN FINAL ---")
# Estrategia de gestión de memoria: forzamos limpieza de caché antes de la inferencia masiva
gc.collect()
if torch.cuda.is_available(): 
    torch.cuda.empty_cache()

# Reafirmamos las constantes de índices y clases
binary_negative='not_informative'
binary_positive='informative'
binaryclass_cat=[binary_negative, binary_positive]
BIN_POS_IDX=1
MULTI_N=len(multiclass_cat)

# 4.1. Funciones auxiliares de métricas y guardado
# ---------------------------------------------------------
def binary_metrics(y_true, prob_positive, threshold=0.5):
    y_true=np.asarray(y_true).astype(int)
    prob_positive=np.asarray(prob_positive).astype(float)
    pred=(prob_positive >= threshold).astype(int)
    return {
        'threshold': float(threshold),
        'precision_informative': float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        'recall_informative': float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        'F2_informative': float(fbeta_score(y_true, pred, beta=2, pos_label=1, zero_division=0)),
        'PR_AUC_informative': float(average_precision_score(y_true, prob_positive)),
        'accuracy': float(accuracy_score(y_true, pred))
    }

def multiclass_metrics(y_true, probs):
    y_true=np.asarray(y_true).astype(int)
    probs=np.asarray(probs)
    pred=np.argmax(probs, axis=1)
    y_bin=label_binarize(y_true, classes=np.arange(MULTI_N))
    return {
        'F2_macro': float(fbeta_score(y_true, pred, beta=2, average='macro', zero_division=0)),
        'PR_AUC_macro': float(average_precision_score(y_bin, probs, average='macro')),
        'accuracy': float(accuracy_score(y_true, pred))
    }

def save_json(obj, path):
    def conv(x):
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, np.ndarray): return x.tolist()
        return str(x)
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=conv)

def save_predictions(path, texts, y_true, probs, pred, class_names):
    out=pd.DataFrame({'tweet_clean': texts, 'y_true': y_true, 'y_pred': pred})
    probs=np.asarray(probs)
    if probs.ndim == 1:
        out['prob_informative']=probs
    else:
        for j, c in enumerate(class_names):
            out[f'prob_{c}']=probs[:, j]
    out.to_csv(path, index=False)

def plot_and_save_cm(y_true, y_pred, class_names, title, path):
    cm=confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    row_sums=cm.sum(axis=1, keepdims=True)
    cm_norm=np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    annot=np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j]=f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"
    fig, ax=plt.subplots(figsize=(11, 9))
    sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                cbar_kws={'label': 'Proporción por clase real'})
    ax.set_title(title, fontweight='bold', pad=18, fontsize=14)
    ax.set_ylabel('Etiqueta real', fontweight='bold')
    ax.set_xlabel('Etiqueta predicha', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


# 4.2. Inferencia y evaluación en clasificación binaria
# ---------------------------------------------------------
print("\n--- EJECUTANDO INFERENCIA: CLASIFICACIÓN BINARIA ---")
texts_train=df_train_clean['tweet_clean'].tolist()
texts_dev=df_dev_clean['tweet_clean'].tolist()
texts_test=df_test_clean['tweet_clean'].tolist()

# SVM Binario 
svm_bin_dev_score=svm_bin_final.decision_function(texts_dev)
svm_bin_test_score=svm_bin_final.decision_function(texts_test)
svm_bin_dev_prob=svm_bin_dev_score
svm_bin_test_prob=svm_bin_test_score

# Bi-LSTM Binario (Glove)
lstm_bin_dev_prob=extract_dl_probs(best_bilstm_glove_bin, dev_loader_bin)[:, BIN_POS_IDX]
lstm_bin_test_prob=extract_dl_probs(best_bilstm_glove_bin, test_loader_bin)[:, BIN_POS_IDX]

# BERT Binario
def bert_binary_probs(model, loader):
    model.eval()
    ys=[]
    probs=[]
    with torch.no_grad():
        for batch in loader:
            ids=batch['input_ids'].to(dispositivo)
            mask=batch['attention_mask'].to(dispositivo)
            out=model(input_ids=ids, attention_mask=mask)
            p=F.softmax(out.logits, dim=1)[:, BIN_POS_IDX]
            probs.extend(p.cpu().numpy())
            ys.extend(batch['labels'].cpu().numpy())
    return np.asarray(ys), np.asarray(probs)

y_dev_bert_bin, bert_bin_dev_prob=bert_binary_probs(best_bert_bin, dev_loader_bert_bin)
y_test_bert_bin, bert_bin_test_prob=bert_binary_probs(best_bert_bin, test_loader_bert_bin)

# Selección del modelo binario óptimo en Dev
binary_dev_candidates={
    'SVM': svm_bin_dev_prob,
    'Bi-LSTM': lstm_bin_dev_prob,
    'BERT Base': bert_bin_dev_prob
}
binary_dev_rows=[{'model': name, **binary_metrics(y_dev_bin_num, score, threshold=0.5)} for name, score in binary_dev_candidates.items()]
binary_dev_table=pd.DataFrame(binary_dev_rows).sort_values('PR_AUC_informative', ascending=False)
binary_dev_table.to_csv(bin_dir/'model_selection_dev.csv', index=False)

selected_binary_model=binary_dev_table.iloc[0]['model']
selected_dev_score=binary_dev_candidates[selected_binary_model]

# Búsqueda del umbral óptimo F2 en Dev
threshold_grid=np.linspace(0.05, 0.95, 181)
threshold_rows=[]
optimal_thresholds={}
for name, dev_score in binary_dev_candidates.items():
    model_rows=[]
    for u in threshold_grid:
        pred=(dev_score >= u).astype(int)
        model_rows.append({
            'model': name, 'threshold': float(u),
            'precision_informative': float(precision_score(y_dev_bin_num, pred, pos_label=1, zero_division=0)),
            'recall_informative': float(recall_score(y_dev_bin_num, pred, pos_label=1, zero_division=0)),
            'F2_informative': float(fbeta_score(y_dev_bin_num, pred, beta=2, pos_label=1, zero_division=0))
        })
    model_table=pd.DataFrame(model_rows)
    best_row=model_table.sort_values(['F2_informative', 'recall_informative', 'precision_informative'], ascending=[False, False, False]).iloc[0]
    optimal_thresholds[name]=float(best_row['threshold'])
    threshold_rows.extend(model_rows)

pd.DataFrame(threshold_rows).to_csv(bin_dir/'threshold_search_dev.csv', index=False)
pd.DataFrame([{'model': name, 'optimal_threshold': u} for name, u in optimal_thresholds.items()]).to_csv(bin_dir/'optimal_thresholds.csv', index=False)

optimal_threshold=optimal_thresholds[selected_binary_model]

# Evaluación en Test binario
binary_test_scores={'SVM': svm_bin_test_prob, 'Bi-LSTM': lstm_bin_test_prob, 'BERT Base': bert_bin_test_prob}
binary_test_rows=[]
binary_test_preds={}
for name, score in binary_test_scores.items():
    u=optimal_thresholds[name]
    pred=(score >= u).astype(int)
    binary_test_preds[name]=pred
    binary_test_rows.append({'model': name, **binary_metrics(y_test_bin_num, score, threshold=u), 'selected_model_on_dev': name == selected_binary_model})
    save_predictions(bin_dir/f'predictions_{name.replace(" ", "_").replace("-", "_")}_binary.csv', texts_test, y_test_bin_num, score, pred, binaryclass_cat)

binary_test_table=pd.DataFrame(binary_test_rows)
binary_test_table.to_csv(bin_dir/'test_metrics_binary.csv', index=False)

# Evaluación completa en Train, Dev y Test para el modelo binario seleccionado
if selected_binary_model == 'SVM':
    train_score=svm_bin_final.decision_function(texts_train)
    y_train_eval=y_train_bin_num
    dev_score=svm_bin_dev_prob; test_score=svm_bin_test_prob
elif selected_binary_model == 'Bi-LSTM':
    train_score=extract_dl_probs(best_bilstm_glove_bin, train_eval_loader_bin)[:, BIN_POS_IDX]
    y_train_eval=y_train_bin_num
    dev_score=lstm_bin_dev_prob; test_score=lstm_bin_test_prob
else:
    y_train_eval, train_score=bert_binary_probs(best_bert_bin, train_eval_loader_bert_bin)
    dev_score=bert_bin_dev_prob; test_score=bert_bin_test_prob

pd.DataFrame([
    {'split': 'train', 'model': selected_binary_model, **binary_metrics(y_train_eval, train_score, optimal_threshold)},
    {'split': 'dev', 'model': selected_binary_model, **binary_metrics(y_dev_bin_num, dev_score, optimal_threshold)},
    {'split': 'test', 'model': selected_binary_model, **binary_metrics(y_test_bin_num, test_score, optimal_threshold)}
]).to_csv(bin_dir/'selected_binary_train_dev_test.csv', index=False)

for name, pred in binary_test_preds.items():
    plot_and_save_cm(y_test_bin_num, pred, binaryclass_cat, f'Matriz de confusión - {name} binario', fig_dir/f'cm_{name.replace(" ", "_").replace("-", "_")}_binary.pdf')

save_json({
    'positive_class': binary_positive, 'negative_class': binary_negative,
    'selection_metric': 'PR-AUC on DEV', 'selected_model': selected_binary_model,
    'optimal_threshold_metric': 'F2 on DEV', 'optimal_threshold': optimal_threshold
}, bin_dir/'binary_selection.json')


# 4.3. Inferencia y evaluación en clasificación multiclase
# ---------------------------------------------------------
print("\n--- EJECUTANDO INFERENCIA: CLASIFICACIÓN MULTICLASE ---")
svm_multi_dev_score=svm_multi_final.decision_function(texts_dev)
svm_multi_test_score=svm_multi_final.decision_function(texts_test)

def decision_to_softmax(x):
    e=np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

svm_multi_dev_output=decision_to_softmax(svm_multi_dev_score)
svm_multi_test_output=decision_to_softmax(svm_multi_test_score)

lstm_multi_dev_prob=extract_dl_probs(best_bilstm_glove_multi, dev_loader_multi)
lstm_multi_test_prob=extract_dl_probs(best_bilstm_glove_multi, test_loader_multi)

def bert_multi_probs(model, loader):
    model.eval()
    ys=[]
    ps=[]
    with torch.no_grad():
        for batch in loader:
            ids, mask=batch['input_ids'].to(dispositivo), batch['attention_mask'].to(dispositivo)
            out=model(input_ids=ids, attention_mask=mask)
            ys.extend(batch['labels'].cpu().numpy())
            ps.extend(F.softmax(out.logits, dim=1).cpu().numpy())
    return np.asarray(ys), np.asarray(ps)

_, bert_multi_dev_prob=bert_multi_probs(best_bert_multi, dev_loader_bert_multi)
_, bert_multi_test_prob=bert_multi_probs(best_bert_multi, test_loader_bert_multi)
_, tweet_multi_dev_prob=bert_multi_probs(best_bertweet_focal, dev_loader_tweet_multi)
_, tweet_multi_test_prob=bert_multi_probs(best_bertweet_focal, test_loader_tweet_multi)

multi_dev_probs={
    'SVM': svm_multi_dev_output,
    'Bi-LSTM': lstm_multi_dev_prob,
    'BERT Base': bert_multi_dev_prob,
    'BERTweet + Focal Loss': tweet_multi_dev_prob
}
multi_test_probs={
    'SVM': svm_multi_test_output,
    'Bi-LSTM': lstm_multi_test_prob,
    'BERT Base': bert_multi_test_prob,
    'BERTweet + Focal Loss': tweet_multi_test_prob
}

multi_dev_rows=[{'model': name, **multiclass_metrics(y_dev_multi_num, p)} for name, p in multi_dev_probs.items()]
pd.DataFrame(multi_dev_rows).sort_values('PR_AUC_macro', ascending=False).to_csv(multi_dir/'model_selection_dev_multiclass.csv', index=False)

multi_test_rows=[]
multi_test_preds={}
for name, p in multi_test_probs.items():
    pred=np.argmax(p, axis=1)
    multi_test_preds[name]=pred
    multi_test_rows.append({'model': name, **multiclass_metrics(y_test_multi_num, p)})
    save_predictions(multi_dir/f'predictions_{name.replace(" ", "_").replace("+", "").replace("-", "_")}_multi.csv', texts_test, y_test_multi_num, p, pred, multiclass_cat)
    plot_and_save_cm(y_test_multi_num, pred, multiclass_cat, f'Matriz de confusión - {name}', fig_dir/f'cm_{name.replace(" ", "_").replace("+", "").replace("-", "_")}_multi.pdf')

multi_test_table=pd.DataFrame(multi_test_rows).sort_values('PR_AUC_macro', ascending=False)
multi_test_table.to_csv(multi_dir/'test_metrics_multiclass.csv', index=False)

# Ensemble ponderado de modelos multiclase
best_w=(1/3, 1/3, 1/3)
best_pr=-np.inf
paso=21
for w_tweet in np.linspace(0, 1, paso):
    for w_svm in np.linspace(0, 1 - w_tweet, paso):
        w_lstm=1 - w_tweet - w_svm
        p=w_tweet * tweet_multi_dev_prob + w_svm * svm_multi_dev_output + w_lstm * lstm_multi_dev_prob
        pr=multiclass_metrics(y_dev_multi_num, p)['PR_AUC_macro']
        if pr > best_pr:
            best_pr=pr
            best_w=(w_tweet, w_svm, w_lstm)

w_tweet, w_svm, w_lstm=best_w
ensemble_dev_prob=w_tweet * tweet_multi_dev_prob + w_svm * svm_multi_dev_output + w_lstm * lstm_multi_dev_prob
ensemble_test_prob=w_tweet * tweet_multi_test_prob + w_svm * svm_multi_test_output + w_lstm * lstm_multi_test_prob
ensemble_test_pred=np.argmax(ensemble_test_prob, axis=1)
ensemble_test_metrics=multiclass_metrics(y_test_multi_num, ensemble_test_prob)

pd.DataFrame([{'model': 'Ensemble', 'w_bertweet': w_tweet, 'w_svm': w_svm, 'w_bilstm': w_lstm, 'PR_AUC_macro_dev': best_pr, **ensemble_test_metrics}]).to_csv(multi_dir/'ensemble_test_metrics.csv', index=False)
save_json({'w_bertweet': w_tweet, 'w_svm': w_svm, 'w_bilstm': w_lstm, 'PR_AUC_macro_dev': best_pr}, multi_dir/'ensemble_weights.json')
save_predictions(multi_dir/'predictions_Ensemble_multi.csv', texts_test, y_test_multi_num, ensemble_test_prob, ensemble_test_pred, multiclass_cat)
plot_and_save_cm(y_test_multi_num, ensemble_test_pred, multiclass_cat, 'Matriz de confusión - Ensemble ponderado', fig_dir/'cm_Ensemble_multi.pdf')

# Test de McNemar
def mcnemar_row(y_true, p1, p2, name1, name2):
    c1, c2=np.asarray(p1) == np.asarray(y_true), np.asarray(p2) == np.asarray(y_true)
    n00=int(np.sum((~c1) & (~c2)))
    n01=int(np.sum((~c1) & c2))
    n10=int(np.sum(c1 & (~c2)))
    n11=int(np.sum(c1 & c2))
    result=mcnemar([[n00, n01], [n10, n11]], exact=True)
    return {'model_1': name1, 'model_2': name2, 'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11, 'p_value': float(result.pvalue)}

all_models_test={**multi_test_preds, 'Ensemble': ensemble_test_pred}
mrows=[mcnemar_row(y_test_multi_num, all_models_test[a], all_models_test[b], a, b)
       for a, b in [('BERT Base', 'SVM'), ('BERTweet + Focal Loss', 'BERT Base'), ('Ensemble', 'BERTweet + Focal Loss')]]
pd.DataFrame(mrows).to_csv(multi_dir/'mcnemar_results.csv', index=False)

# Resumen global y resultados finales
summary_binary=binary_test_table.copy(); summary_binary['task']='binary'
summary_multi=multi_test_table.copy(); summary_multi['task']='multiclass'
pd.concat([summary_binary, summary_multi], ignore_index=True, sort=False).to_csv(results_dir/'summary_metrics.csv', index=False)

print('\n' + '='*80)
print('EJECUCIÓN DEL TFG COMPLETA Y PURA FINALIZADA CON ÉXITO')
print('='*80)
print(f'MÉTRICAS TEST BINARIO:\n{binary_test_table.to_string(index=False)}')
print(f'\nMÉTRICAS TEST MULTICLASE:\n{multi_test_table.to_string(index=False)}')
print('\nTodos los artefactos y resultados están guardados en resultados_TFG/.')