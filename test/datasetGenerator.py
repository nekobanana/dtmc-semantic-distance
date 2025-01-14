import numpy as np
import json
from scipy.stats import wasserstein_distance


def generate_markov_matrix(size):
    """Generate a random Markov transition matrix."""
    matrix = np.random.rand(size, size)
    return matrix / matrix.sum(axis=1, keepdims=True)


def compute_stationary_distribution(matrix, tol=1e-8, max_iter=1000):
    """Compute the stationary distribution of a Markov chain using the power method."""
    n = matrix.shape[0]
    dist = np.random.rand(n)
    dist /= dist.sum()  # Normalize to make it a valid probability distribution

    for _ in range(max_iter):
        new_dist = np.dot(dist, matrix)
        if np.linalg.norm(new_dist - dist, ord=1) < tol:
            return new_dist
        dist = new_dist

    return None  # Return None if convergence fails


def total_variation_distance(pi1, pi2):
    """Compute Total Variation Distance."""
    return 0.5 * np.sum(np.abs(pi1 - pi2))


def kl_divergence(pi1, pi2):
    """Compute Kullback-Leibler Divergence."""
    mask = (pi1 > 0) & (pi2 > 0)  # Avoid division by zero or log(0)
    return np.sum(pi1[mask] * np.log(pi1[mask] / pi2[mask]))


def l2_distance(pi1, pi2):
    """Compute L2 Distance."""
    return np.linalg.norm(pi1 - pi2)


def wasserstein_dist(pi1, pi2):
    """Compute Wasserstein Distance between two stationary distributions."""
    return wasserstein_distance(np.arange(len(pi1)), np.arange(len(pi2)), pi1, pi2)


def spectral_distance(m1, m2):
    """Compute Spectral Distance based on eigenvalues of the transition matrices."""
    eigvals_m1 = np.linalg.eigvals(m1)
    eigvals_m2 = np.linalg.eigvals(m2)
    return np.linalg.norm(np.sort(eigvals_m1) - np.sort(eigvals_m2))


def generate_markov_chain_pairs(n_pairs, size_range, n_similar):
    """Generate random Markov chain pairs and compute distances."""
    results = []

    for _ in range(n_similar):
        size = np.random.randint(size_range[0], size_range[1] + 1)  # Random size within range

        # Generate Markov chains
        m1 = generate_markov_matrix(size)
        m2 = m1

        # Compute stationary distributions
        pi1 = compute_stationary_distribution(m1)
        pi2 = compute_stationary_distribution(m2)

        wasserstein = wasserstein_dist(pi1, pi2)
        if (wasserstein < 0.1 * 10**-4):
            wasserstein = 0
        kl = kl_divergence(pi1, pi2)
        if (kl < 0.1 * 10 **(-4)):
            kl = 0
        l2 = l2_distance(pi1, pi2)
        if (l2 < 0.1 * 10 **(-4)):
            l2 = 0
        tv = total_variation_distance(pi1, pi2)
        if (tv < 0.1 * 10 **(-4)):
            tv = 0
        sp = spectral_distance(m1, m2)
        if (sp < 0.1 * 10 **(-4)):
            sp = 0

        if pi1 is None or pi2 is None:
            # If convergence fails, set all distances to -1
            distances = {
                "total_variation": -1,
                "kl_divergence": -1,
                "l2_distance": -1,
                "wasserstein": -1,
                "spectral": -1
            }
        else:
            # Compute distances if stationary distributions are valid
            distances = {
                "total_variation": tv,
                "kl_divergence": kl,
                "l2_distance": l2,
                "wasserstein": wasserstein,
                "spectral": sp
            }

        # Store results
        results.append({
            "m1": m1.flatten().tolist(),
            "m2": m2.flatten().tolist(),
            "pi1": pi1.tolist() if pi1 is not None else None,
            "pi2": pi2.tolist() if pi2 is not None else None,
            "distances": distances
        })

    for _ in range(n_pairs):
        size = np.random.randint(size_range[0], size_range[1] + 1)  # Random size within range

        # Generate Markov chains
        m1 = generate_markov_matrix(size)
        m2 = generate_markov_matrix(size)

        # Compute stationary distributions
        pi1 = compute_stationary_distribution(m1)
        pi2 = compute_stationary_distribution(m2)

        wasserstein = wasserstein_dist(pi1, pi2)
        if(wasserstein < 0.1*10**(-4)):
            wasserstein = 0
        kl = kl_divergence(pi1, pi2)
        if (kl < 0.1 * 10 **(-4)):
            kl = 0
        l2 = l2_distance(pi1, pi2)
        if (l2 < 0.1 * 10 **(-4)):
            l2 = 0
        tv =  total_variation_distance(pi1, pi2)
        if (tv < 0.1 * 10 **(-4)):
            tv = 0
        sp = spectral_distance(m1, m2)
        if (sp < 0.1 * 10 **(-4)):
            sp = 0

        if pi1 is None or pi2 is None:
            # If convergence fails, set all distances to -1
            distances = {
                "total_variation": -1,
                "kl_divergence": -1,
                "l2_distance": -1,
                "wasserstein": -1,
                "spectral": -1
            }
        else:
            # Compute distances if stationary distributions are valid
            distances = {
                "total_variation": tv,
                "kl_divergence": kl,
                "l2_distance": l2,
                "wasserstein": wasserstein,
                "spectral": sp
            }

        # Store results
        results.append({
            "m1": m1.flatten().tolist(),
            "m2": m2.flatten().tolist(),
            "pi1": pi1.tolist() if pi1 is not None else None,
            "pi2": pi2.tolist() if pi2 is not None else None,
            "distances": distances
        })

    return results


def save_to_json(filename, data):
    """Save results to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    # Parameters
    n_pairs = 800  # Number of Markov chain pairs
    n_similar = 200
    size_range = (5, 50)  # Range for the number of states (min, max)
    output_file = "markov_chain_results.json"

    # Generate Markov chain pairs and compute distances
    results = generate_markov_chain_pairs(n_pairs, size_range, n_similar)

    # Save results to a JSON file
    save_to_json(output_file, results)

    print(f"Results saved to {output_file}")