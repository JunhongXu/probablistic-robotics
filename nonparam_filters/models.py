"""
This module implements different motion and measurement models.
"""

import numpy as np


def multivariate_gaussian(x, mean, cov):
    """
    Compute the density of the given variable x.
    """
    dim = x.shape[0]
    # reshape x to a column vector
    x = x.reshape(-1, 1)
    mean = mean.reshape(-1, 1)
    return  np.power(2 * np.pi, -dim / 2.) * np.power(np.linalg.det(cov), -0.5) * np.exp(-0.5 * (x - mean).T @ np.linalg.inv(cov) @ (x - mean))[0, 0]

def gaussian_pdf(x, mean, variance):
    return 1 / np.sqrt(2 * np.pi * variance) * np.exp(-(mean - x) ** 2 / (2. * variance))


class NonLinearMotionModel(object):
    """
    Defines the nonlinear motion [x, y, \theta] = [x + cos\theta, y+sin\theta, \theta].
    It returns the probability density p(x' | u', x) 
    """
    def __init__(self):
        self.cov = np.array([[0.001, 0.0, 0.0], [0.0, 0.001, 0.0], [0.0, 0.0, 0.001]])
    
    def __call__(self, curr_state, u, prev_state):
        mean_state = prev_state.flatten()
        mean_state[0] = mean_state[0] + u * np.cos(mean_state[2])
        mean_state[1] = mean_state[1] + u * np.sin(mean_state[2])
        prob_density = multivariate_gaussian(curr_state, mean_state, self.cov)
        return prob_density
    
    def sample(self, u, x):
        mean_state = x.flatten()
        mean_state[0] = mean_state[0] + u * np.cos(mean_state[2])
        mean_state[1] = mean_state[1] + u * np.sin(mean_state[2])
        # sample the next state
        return np.random.multivariate_normal(mean_state, self.cov)


class MeasurementModel(object):
    def __init__(self, C, Q):
        """
        Implements the measurement model for a noisy GPS.
        """
        self.C = C
        self.Q = Q
    
    def __call__(self, measurement, state):
        """
        Gives the probability density of p(z | x)
        """
        measurement = measurement.flatten()
        state = state.reshape(-1, 1)
        mean = self.C @ state
        return gaussian_pdf(measurement, mean[0, 0], self.Q)
    
    def sample(self, state):
        """
        Samples one measurement from the measurement model.
        """
        state = state.reshape(-1, 1)
        mean = self.C @ state
        return np.random.normal(loc=mean, scale=self.Q)


class MotionModel(object):
    def __init__(self, A, B, R):
        self.A = A
        self.B = B
        self.R = R
    
    def mean(self, x, u):
        x = x.reshape(-1, 1)
        u = np.array([[u]])
        mean = self.A @ x + self.B @ u 
        return mean 
    
    def __call__(self, curr_state, u, prev_state):
        """
        Output a a value represents the probability of transitioning to the next state
        """
        prev_state = prev_state.reshape(-1, 1)
        u = np.array([[u]])
        mean = self.A @ prev_state + self.B @ u 
        return multivariate_gaussian(curr_state, mean, self.R)

    def sample(self, u, x):
        x = x.reshape(-1, 1)
        u = np.array([[u]])
        mean = self.A @ x + self.B @ u
        return np.random.multivariate_normal(mean.flatten(), self.R)
