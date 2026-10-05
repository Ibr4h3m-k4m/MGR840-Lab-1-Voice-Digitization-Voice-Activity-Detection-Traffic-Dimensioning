# Utilisation des prédictions du meilleur modèle (ex. Random Forest)
y_pred_best = rf_model.predict(X_test_scaled)

# Calcul du Facteur d'Activité Vocale (alpha)
alpha = np.mean(y_pred_best)

r_nominal = 64.0 # kbit/s (Canal DS0)
r_effective = alpha * r_nominal
bandwidth_savings = (1 - alpha) * 100

print(f"--- Bilan d'Économie Réseau ---")
print(f"Trames analysées dans le test set : {len(y_pred_best)}")
print(f"Facteur d'activité vocale (alpha)  : {alpha:.3f}")
print(f"Débit effectif moyen par voie      : {r_effective:.2f} kbit/s")
print(f"Économie de bande passante         : {bandwidth_savings:.2f} %")