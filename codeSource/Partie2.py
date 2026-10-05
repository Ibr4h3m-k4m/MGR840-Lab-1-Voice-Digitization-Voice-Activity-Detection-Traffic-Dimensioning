import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. Chargement des datasets fournis
df_train = pd.read_csv('vad_dataset_train.csv')
df_test = pd.read_csv('vad_dataset_test.csv')

# 2. Séparation des caractéristiques (X) et de la cible (y)
feature_cols = ['energy', 'zcr', 'spectral_flatness', 'spectral_centroid']

X_train = df_train[feature_cols]
y_train = df_train['label']

X_test = df_test[feature_cols]
y_test = df_test['label']

# 3. Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- À COMPLÉTER PAR L'ÉTUDIANT ---
# Entraîner LogisticRegression et RandomForestClassifier
# Comparer les performances sur X_test_scaled et y_test