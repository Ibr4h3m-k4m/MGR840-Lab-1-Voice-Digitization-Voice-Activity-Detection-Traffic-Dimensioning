import numpy as np
import matplotlib.pyplot as plt

def compress_mu_law(x, mu=255):
    """ Compression logarithmique Loi-mu (ITU-T G.711 North American Standard) """
    return np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)

def decompress_mu_law(y, mu=255):
    """ Décompression (expansion) Loi-mu """
    return np.sign(y) * ((1 + mu)**np.abs(y) - 1) / mu

def quantize_uniform(x, num_bits=8, v_max=1.0):
    """ Quantification uniforme mid-tread sur num_bits """
    levels = 2**num_bits
    q = (2 * v_max) / levels
    x_clipped = np.clip(x, -v_max, v_max)
    indices = np.round((x_clipped + v_max) / q)
    indices = np.clip(indices, 0, levels - 1)
    x_q = indices * q - v_max + (q / 2)
    return x_q

def compute_sqnr(x_original, x_quantized):
    """ Calcul du SQNR empirique en dB """
    p_signal = np.mean(x_original**2)
    p_noise = np.mean((x_original - x_quantized)**2)
    if p_noise == 0:
        return float('inf')
    return 10 * np.log10(p_signal / p_noise)

# Exemple de génération des phonèmes
fs = 8000
t = np.linspace(0, 0.1, int(fs * 0.1), endpoint=False)

# 1. Voyelle (Forte amplitude)
vowel = 0.8 * np.sin(2 * np.pi * 800 * t)

# 2. Consonne (Faible amplitude)
np.random.seed(42)
consonant = 0.04 * np.random.randn(len(t))

# --- À COMPLÉTER PAR L'ÉTUDIANT ---
# Calculer les SQNR Uniforme vs Loi-mu pour voyelle et consonne