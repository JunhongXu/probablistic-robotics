import numpy as np
import matplotlib.pyplot as plt
from nonparam_filters.hist.hist_filter import HistFilter, MotionModel, MeasurementModel, NonLinearMotionModel, init_belief
from nonparam_filters.hist.state_tb import StateTable
from gaussian_filters.kf import KalmanFilter, run_pred


def run_hist_pred(hist_filter):
    # we only apply 0 control
    hist_filter.predict(0.0)
    plot_hist_dist(hist_filter)


def plot_hist_dist(hist_filter, use_angle=False, **kwargs):
    if use_angle:
        n = hist_filter.state_tb.num_grids[2]
    else:
        n = 1

    for angle in range(n):
        bel = hist_filter.bel
        if use_angle:
            _num_state = hist_filter.state_tb.num_grids[0] * hist_filter.state_tb.num_grids[1]
            state_slice = [angle + _i * n for _i in range(_num_state)]
        else:
            state_slice = np.arange(0, hist_filter.state_tb.n_state)
        values = []
        for i in range(1000):
            # first we draw a state from the belief
            s = np.random.choice(np.arange(0, hist_filter.n_states), p=bel)
            # draw random values from the state region if it is inside the state slice
            if s in state_slice:
                value_range = hist_filter.state_tb.s_range(s) 
                s_value = []
                for _range in value_range:
                    s_value.append(np.random.uniform(low=_range[0], high=_range[1]))
                values.append(s_value)
        values = np.array(values)
        if values.shape[0] != 0:
            if use_angle:
                angle_value = hist_filter.state_tb.s_range(state_slice[0])[2]
                legend = "%.2f - %.2f" %(angle_value[0], angle_value[1]) 
            else:
                legend = ""
            plt.scatter(values[:, 0], values[:, 1], s=1, label=legend, **kwargs)


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
    plot_hist_dist(hist_filter, use_angle=False, color='gray')
    # measurement update
    hist_filter.update(measurement)
    plot_hist_dist(hist_filter, use_angle=False, color='red')




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


def run_sol_two():
    """
    The solution of the exercise 4.6.2 (a) and 4.6.2 (b)
    """

    # parameters of the histogram filter
    x_range = [-1.8, 1.8]
    y_range = [-1.8, 1.8]
    theta_range = [0.0, 2 * np.pi]
    interval = (x_range, y_range, theta_range)
    ngrid = [15, 15, 8]
    names = ['x', 'y', 'theta']
    tb = StateTable(interval, ngrid, names)
    motion_model = NonLinearMotionModel()
    # C = [1.0, 0.0, 0.0] and the variance=0.01
    measurement_model = MeasurementModel(np.array([[1.0, 0.0, 0.0]]), 0.01) 

    hist_filter = HistFilter(None, tb, motion_model, measurement_model)

    # initialize the belief 
    init_mean = np.array([0.0, 0.0, 0.0])
    init_cov = np.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 10000]])
    init_belief(hist_filter, init_mean, init_cov)

    print("Plotting the solution for 4.6.2 (a)")
    hist_filter.predict(1.0)
    plot_hist_dist(hist_filter, use_angle=True)
    plt.legend()
    plt.show()

    print("Plotting the solution for 4.6.2 (b)")
    # sample the state from the motion model 
    next_state = motion_model.sample(1.0, init_mean)
    # we can not directly observe the state, we only observe a noisy measurement around that state
    measurement = measurement_model.sample(next_state)
    print("The sampled measurement is", measurement)
    # measurement update
    hist_filter.update(measurement)
    plot_hist_dist(hist_filter, use_angle=True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # run_sol_one()
    run_sol_two()