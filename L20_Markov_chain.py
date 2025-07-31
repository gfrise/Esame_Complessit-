import numpy as np
import matplotlib.pyplot as plt

# ——————————————————————————————————————————————————
# Funzioni base per catene di Markov
# ——————————————————————————————————————————————————

def step(pi, P):
    """Un passo: pi (1×N) · P (N×N) → nuova distribuzione."""
    return pi @ P

def distribution(pi0, P, n):
    """Applica n passi di Markov a pi0."""
    pi = pi0.copy()
    for _ in range(n):
        pi = step(pi, P)
    return pi

def stationary(P, tol=1e-12, maxiter=1000):
    """Iterazioni di potenze per π = πP; converge se P è ergodica."""
    N = P.shape[0]
    pi = np.ones(N)/N
    for _ in range(maxiter):
        pi_next = step(pi, P)
        if np.abs(pi_next - pi).sum() < tol:
            return pi_next
        pi = pi_next
    raise RuntimeError("Non converge")

def simulate_path(P, start, length):
    """Simula una traiettoria di stati di lunghezza `length`."""
    path = [start]
    for _ in range(length-1):
        path.append(np.random.choice(len(P), p=P[path[-1]]))
    return np.array(path)

# ——————————————————————————————————————————————————
# Esempio d’uso con plot
# ——————————————————————————————————————————————————
if __name__ == "__main__":
    # 1) Matrice di transizione (3 stati)
    P = np.array([[0.9, 0.1, 0.0],
                  [0.2, 0.7, 0.1],
                  [0.0, 0.3, 0.7]])

    # 2) Calcolo distribuzioni a vari passi
    pi0 = np.array([1,0,0])  # inizio certo allo stato 0
    steps = [1, 5, 10, 50]
    dists = [distribution(pi0, P, n) for n in steps]

    # 3) Stazionaria
    pi_stat = stationary(P)

    # 4) Simulazione traiettoria
    path = simulate_path(P, start=0, length=200)

    # — Plot distribuzioni transienti e stazionaria —
    plt.figure(figsize=(8,4))
    for pi, n in zip(dists, steps):
        plt.plot(pi, marker='o', label=f'n={n}')
    plt.plot(pi_stat, marker='s', linestyle='--', color='k', label='stazionaria')
    plt.xlabel('Stato'); plt.ylabel('Probabilità')
    plt.title('Distribuzioni di Markov dopo n passi')
    plt.legend()
    plt.tight_layout()

    # — Plot traiettoria di stati —
    plt.figure(figsize=(8,2))
    plt.step(np.arange(len(path)), path, where='post')
    plt.yticks(range(P.shape[0]))
    plt.xlabel('Passo'); plt.ylabel('Stato')
    plt.title('Traiettoria simulata di Markov')
    plt.tight_layout()

    plt.show()
