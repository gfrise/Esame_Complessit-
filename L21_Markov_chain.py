import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

def transition_matrix(sequence, num_states):
    matrix = np.zeros((num_states, num_states), dtype=float)
    for curr, nxt in zip(sequence[:-1], sequence[1:]):
        matrix[curr, nxt] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums

def simulate_markov(transition_matrix, initial_state, num_steps):
    """
    Simulate a first-order Markov chain given a transition matrix.
    Returns the list of visited states (length = num_steps).
    """
    states = [initial_state]
    for _ in range(num_steps - 1):
        current = states[-1]
        next_state = np.random.choice(len(transition_matrix), p=transition_matrix[current])
        states.append(next_state)
    return states

sequence = [0, 1, 2, 1, 0, 2, 2,1, 2, 3, 3, 0, 2, 3, 1, 0, 1, 2]  # <-- replace with your data
num_states = 4               # Number of distinct states in your data
num_steps = 10000            # Number of steps to simulate
initial_state = sequence[0]  # Use first element of sequence or choose a state

prob_matrix = transition_matrix(sequence, num_states)
plt.figure(figsize=(6, 5))
plt.imshow(prob_matrix, interpolation='nearest')
plt.title('Transition Probability Matrix')
plt.xlabel('Next State')
plt.ylabel('Current State')
plt.colorbar(label='Probability')
plt.show()

# 3) Simulate the Markov chain and plot state visitation histogram
simulated_states = simulate_markov(prob_matrix, initial_state, num_steps)
plt.figure(figsize=(6, 4))
plt.hist(simulated_states, bins=num_states, align='left', rwidth=0.8)
plt.title('Histogram of Simulated State Visits')
plt.xlabel('State')
plt.ylabel('Visit Count')
plt.xticks(range(num_states))
plt.show()

# 4) Plot the trajectory of the first 200 steps
plt.figure(figsize=(8, 3))
plt.plot(simulated_states[:200], marker='o', linestyle='-')
plt.title('State Trajectory (First 200 Steps)')
plt.xlabel('Step')
plt.ylabel('State')
plt.yticks(range(num_states))
plt.tight_layout()
plt.show()

#####################################

P = np.array([[ 0.8, 0.2], 
              [0.1, 0.9]]) 

eigvals = np.linalg.eigvals(P)
eigvals = np.sort(np.abs(eigvals))[::-1]
rate = eigvals[1]
print(f'Il rate di convergenza è: {rate:.2}')

# Evoluzione della distribuzione di probabilitò da pi(0) a pi(t)
# Nel tempo l'evoluzione è pi(t) = pi(0)*P^t

pi0 = ([1.0, 0.0]) #distr iniziale casuale
# Evoluzione per T passi
T = 20
pi_t = np.zeros((T+1, len(pi0)))
pi_t[0] = pi0

for t in range(1, T+1):
    pi_t[t] = pi_t[t-1] @ P # Moltiplicazione tra matrici

plt.plot(pi_t[:,0], label='Stato 0')
plt.plot(pi_t[:,1], label='Stato 1')
plt.axhline(np.linalg.matrix_power(P, 1000)[0,0], color='gray', linestyle='--', label='Stazionario')
plt.xlabel('Tempo')
plt.ylabel('Probabilità')
plt.title('Evoluzione della distribuzione')
plt.grid(True)
plt.show()
# La funzione stazionaria soddisfa anche P.T pi.T = pi.T dove con .T indico la trasposta, ovviamente soddisfa anche l'uguaglianza senza .T
w, v = eig(P.T)
stationary = v[:, np.isclose(w, 1)].real 
print("Distribuzione stazionaria:", stationary/stationary.sum())