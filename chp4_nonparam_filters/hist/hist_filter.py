import matplotlib.cm as cm           # import colormap stuff!
import numpy as np
from kf.kf import run_pred, KalmanFilter 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from hist.state_tb import StateTable


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
        state = state.reshape(-1, 1)
        mean = self.C @ state
        return gaussian_pdf(measurement, mean[0, 0], self.Q)

class MotionModel(object):
    def __init__(self, A, B, R):
        self.A = A
        self.B = B
        self.R = R
    
    def __call__(self, curr_state, u, prev_state):
        """
        Output a a value represents the probability of transitioning to the next state
        """
        prev_state = prev_state.reshape(-1, 1)
        u = np.array([[u]])
        mean = self.A @ prev_state + self.B @ u 
        return multivariate_gaussian(curr_state, mean, self.R)
        

class HistFilter(object):
    """
    An implementation of the histogram filter. In the histogram filter, the belief is
    represented by a probability value for each of the discretized state.
    """

    def __init__(self, init_bel, tb, motion_model, measurement_model):
        """
        init_bel is a list that represents the initial belief. 
        state_tb represents the discretized state space.
        The motion and measurement models.
        """
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.state_tb = tb
        # the number of states should be the same as the state table
        self.n_states = tb.n_state
        # initialize the belief
        self.bel = init_bel
        self.init_bel = init_bel

    def predict(self, u):
        """
        The prediction step of the histogram filter. This step updates the 
        belief probability.
        """
        bel_cp = np.zeros_like(self.bel)
        # we iterate over each state to get the summation \bar{bel(x_t)} = \sum p(x_t | u, x_{t-1}) bel(x_{t-1})
        for s_curr in range(self.n_states):
            # iterate over the previous states
            s_curr_val = self.state_tb.value(s_curr)
            prob = 0.0
            for s_prev in range(self.n_states):
                s_prev_val = self.state_tb.value(s_prev)
                transition_prob = self.motion_model(s_curr_val, u, s_prev_val)
                # weighted average
                prob += transition_prob * self.bel[s_prev]
            # print("Probability of", s_curr_val, "is", prob)
            bel_cp[s_curr] = prob
        normalizer = 1 / np.sum(bel_cp)        
        self.bel = bel_cp * normalizer
    
    def update(self, measurement):
        """
        Update the belief distribution given the measurement.
        """
        for s in range(self.n_states):
            # maps the discrete state to the continuous state
            value = self.state_tb.value(s)
            # p(z_t | x_{t, k})
            meas_prob = self.measurement_model(measurement, value)
            # p_{t, k} = p(z_t | x_{t, k}) * \Bar{p}_{t, k} 
            self.bel[s] = self.bel[s] * meas_prob 
        # normalizing 
        self.bel = self.bel / np.sum(self.bel)

    def reset(self):
        self.bel = self.init_bel 