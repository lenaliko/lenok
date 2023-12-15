import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    vec = np.ones(data.shape[1])
    for i in range(num_steps):
        vec_prev = vec
        vec = (np.dot(data, vec)) / (np.linalg.norm(np.dot(data, vec)))
        lambda_ = (np.dot(vec_prev,np.dot(data, vec))) / (np.dot(vec_prev, vec_prev))
    return float(lambda_), vec