from xgboost import XGBRegressor

def erlang_b(A, S):
    """ Calcul récursif de la probabilité de blocage Erlang B """
    pb = 1.0
    for k in range(1, S + 1):
        pb = (A * pb) / (k + A * pb)
    return pb

def find_min_channels(A, target_pb=0.01):
    """ Trouve le nombre minimal de canaux S pour garantir Pb <= target_pb """
    S = 1
    while True:
        if erlang_b(A, S) <= target_pb:
            return S
        S += 1

# 1. Calcul Statique
A_peak = 35.0 # Erlangs à l'heure de pointe
S_static = find_min_channels(A_peak, target_pb=0.01)
t1_count_static = int(np.ceil(S_static / 24))

print(f"--- Dimensionnement Statique (Heure de Pointe) ---")
print(f"Trafic de pointe : {A_peak} Erlangs")
print(f"Canaux requis    : {S_static} (soit {t1_count_static} liaisons T1 de 24 canaux)")

# 2. Chargement des données de trafic 24h
df_traffic = pd.read_csv('traffic_24h.csv') # Contient les colonnes 'timestamp' et 'offered_traffic_erlangs'

# --- À COMPLÉTER PAR L'ÉTUDIANT ---
# 1. Générer les colonnes lag_1, lag_2, lag_3, lag_4
# 2. Entraîner XGBRegressor
# 3. Calculer la capacité dynamique S(t) et le gain en ressources sur 24h