import numpy as np
import matplotlib.pyplot as plt

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

