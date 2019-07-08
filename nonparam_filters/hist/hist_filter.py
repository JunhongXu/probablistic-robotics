import matplotlib.cm as cm           # import colormap stuff!
import numpy as np
# from kf.kf import run_pred, KalmanFilter 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from nonparam_filters.hist.state_tb import StateTable
from nonparam_filters.models import multivariate_gaussian


def init_belief(hist_filter, init_mean, init_cov):
    """
    Initialize the belief for the histogram filter based on the mean and covariance matrix
    """

    for k in range(hist_filter.n_states):
        state_value = hist_filter.state_tb.value(k)
        hist_filter.bel[k] = multivariate_gaussian(state_value, init_mean, init_cov)
    # normalize
    hist_filter.bel = hist_filter.bel / np.sum(hist_filter.bel)

        

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
        init_bel = init_bel if init_bel is not None else np.full(self.n_states, 1./self.n_states)
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