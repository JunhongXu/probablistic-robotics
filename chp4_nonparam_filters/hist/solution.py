import numpy as np
import matplotlib.pyplot as plt
from hist.hist_filter import HistFilter, MotionModel, MeasurementModel
from hist.state_tb import StateTable
from kf.kf import KalmanFilter, run_pred


def run_hist_pred(hist_filter):
    # we only apply 0 control
    hist_filter.predict(0.0)
    plot_dist(hist_filter)


def plot_dist(hist_filter, color='gray'):
    bel = hist_filter.bel
    # sample 1000 points from the state space
    for i in range(300):
        # first we draw a state from the belief
        s = np.random.choice(np.arange(0, hist_filter.state_tb.n_state), p=bel)
        # draw random values from the state region
        value_range = hist_filter.state_tb.s_range(s) 
        s_value = []
        for _range in value_range:
            s_value.append(np.random.uniform(low=_range[0], high=_range[1]))
        plt.scatter(s_value[0], s_value[1], c=color, s=1)


def plot_measurement_update(init_state, hist_filter):
    # map the discrete state to the continuous state
    true_state = hist_filter.state_tb.value(init_state).reshape(-1, 1)
    A = hist_filter.motion_model.A
    B = hist_filter.motion_model.B
    for i in range(5):
        # the control has random noise
        true_state = A @ true_state + B @ [[np.random.randn()]]
    
    # measurement has random noise on the poistion of the robot
    x = true_state.flatten()[0]
    measurement = x + np.random.randn() * np.sqrt(10.0)
    # plot the distribution before measurement update
    plot_dist(hist_filter)
    # measurement update
    hist_filter.update(measurement)
    plot_dist(hist_filter, 'red')


def run_sol_one():
    """
    The solution of exercise 4.1
    """
    ####### Parameters of the histogram filter ######
    pos_interval = [-15.0, 15.0]
    vel_interval = [-15.0, 15.0]
    # max - min - 1
    num_grids = [29, 29]
    names = ['pos', 'vel']
    tb = StateTable((pos_interval, vel_interval), num_grids, names)
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    B = np.array([[0.5], [1.0]])
    R = np.zeros((2, 2))
    R[0, 0] = 0.25
    R[1, 1] = 1
    # motion model
    motion = MotionModel(A, B, R) 

    # measurement model
    C = np.array([[1.0, 0.0]])
    Q = 10
    measurement_model = MeasurementModel(C, Q)

    # initial state
    init_state = tb.s_id(np.array([0.0, 0.0]))
    init_bel = np.zeros(tb.n_state)
    init_bel[init_state] = 1.0

    hist_filter = HistFilter(init_bel, tb, motion, measurement_model)

    ###### Parameters of the Kalman Filter ######
    C = np.array([[1.0, 0.0]])
    Q = np.array([[10.0]]) 
    mean = np.array([[0.0], [0.0]])
    cov = np.array([[0.0, 0.0], [0.0, 0.0]])

    kalman_filter = KalmanFilter(A, B, C, R, Q, mean, cov)

    ###### Question 4.6.1 (a) ######
    for i in range(1, 6):
        run_hist_pred(hist_filter)
        run_pred(kalman_filter, n_times=i)
        plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.xlim(-15, 15)
        plt.ylim(-15, 15)
        plt.show()
    
    ###### Question 4.6.1 (b) ######
    plot_measurement_update(init_state, hist_filter)
    plt.show()


if __name__ == "__main__":
    run_sol_one()