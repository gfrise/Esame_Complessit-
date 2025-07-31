import numpy as np
import matplotlib.pyplot as plt

# ================================================
# Utility functions for Markov chain analysis
# ================================================

def build_transition_matrix(sequence, num_states):
    """
    Build a transition count matrix from the given sequence.
    matrix[i, j] = # of transitions from state i to state j.
    """
    matrix = np.zeros((num_states, num_states), dtype=float)
    for current_state, next_state in zip(sequence[:-1], sequence[1:]):
        matrix[current_state, next_state] += 1
    return matrix


def normalize_matrix(count_matrix):
    """
    Convert a count matrix into a probability transition matrix by row-normalization.
    Each row sums to 1.
    """
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # Prevent division by zero
    return count_matrix / row_sums


def simulate_markov(transition_matrix, initial_state, num_steps, seed=None):
    """
    Simulate a first-order Markov chain given a transition matrix.
    Returns the list of visited states (length = num_steps).
    """
    if seed is not None:
        np.random.seed(seed)
    states = [initial_state]
    for _ in range(num_steps - 1):
        current = states[-1]
        next_state = np.random.choice(len(transition_matrix), p=transition_matrix[current])
        states.append(next_state)
    return states

# ================================================
# Main execution block
# ================================================
if __name__ == '__main__':
    # ---- User parameters ----
    # Provide your own sequence of integer states (e.g., [0,1,2,1,0,...])
    sequence = [0, 1, 2, 1, 0, 2, 2, 3, 1, 0, 1, 2]  # <-- replace with your data
    num_states = 4               # Number of distinct states in your data
    num_steps = 10000            # Number of steps to simulate
    initial_state = sequence[0]  # Use first element of sequence or choose a state
    random_seed = 0              # For reproducibility

    # 1) Build and plot the raw transition count matrix
    count_matrix = build_transition_matrix(sequence, num_states)
    plt.figure(figsize=(6, 5))
    plt.imshow(count_matrix, interpolation='nearest')
    plt.title('Transition Count Matrix')
    plt.xlabel('Next State')
    plt.ylabel('Current State')
    plt.colorbar(label='Counts')
    plt.show()

    # 2) Normalize to obtain the transition probability matrix and plot
    prob_matrix = normalize_matrix(count_matrix)
    plt.figure(figsize=(6, 5))
    plt.imshow(prob_matrix, interpolation='nearest')
    plt.title('Transition Probability Matrix')
    plt.xlabel('Next State')
    plt.ylabel('Current State')
    plt.colorbar(label='Probability')
    plt.show()

    # 3) Simulate the Markov chain and plot state visitation histogram
    simulated_states = simulate_markov(prob_matrix, initial_state, num_steps, seed=random_seed)
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

    # End of program
