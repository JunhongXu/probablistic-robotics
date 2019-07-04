import numpy as np
from gaussian_filters.utils import plot_cov_ellipse
import matplotlib.pyplot as plt


class ExtendedKalmanFiter(object):
    """
    A simple implementation of the EKF algorithm. It is for completing the exercise 3.8.4.
    G and H are the functions for computing the Jacobian matrices of the motion and measurement models.
    R and Q are the covariance matrices of the motion and measurement models.
    mean and cov are initial mean and covariance of the belief. 
    motion_fn is the motion model.
    measurement_fn is the measurement model.
    """

    def __init__(self, G, H, R, Q, mean, cov, motion_fn, measurement_fn):
        self.G = G
        self.H = H
        self.R = R
        self.Q = Q

        self.init_mean = mean
        self.init_cov = cov

        self.mean = mean
        self.cov = cov

        self.motion_fn = motion_fn
        self.measurement_fn = measurement_fn

    def predict(self, u):
        """
        The prediction step, u is the control.
        """
        jacobian = self.G(self.mean.flatten()[2], u)
        self.mean = self.motion_fn(self.mean, u)
        self.cov = jacobian @ self.cov @ jacobian.T + self.R

    def update(self, measurement):
        gain = self.cov @ self.H.T @ np.linalg.inv(self.H @ self.cov @ self.H.T + self.Q)
        self.mean = self.mean + gain @ (measurement - measurement_fn(self.mean))
        self.cov = (np.identity(3) - gain @ self.H) @ self.cov

    def reset(self):
        self.cov = self.init_cov
        self.mean = self.init_mean


def motion_fn(state, control):
    """
    The non-linear motion function that takes the previous state and current control
    to produce the next state. Here, we do not assume there is any noise in the control.
    x_{t+1} = x_t + u_t * theta
    y_{t+1} = y_t + u_t * theta
    theta_{t+1} = theta_t
    """
    state = state.flatten()
    x = state[0] + control * np.cos(state[2])
    y = state[1] + control * np.sin(state[2])
    theta = state[2]
    next_state = np.array([x, y, theta])
    # to a column vector
    return next_state.reshape(3, -1)


def measurement_fn(state):
    """
    Very simple measurement function: z_t = [1, 0, 0] @ x_t + \epsilon
    """ 
    C = np.array([[1.0, 0.0, 0.0]])
    return C @ state


def sampale_starting_state():
    x = np.random.randn() * 0.2
    y = np.random.randn() * 0.2
    theta = np.random.normal(0, 100)
    return np.array([x, y, theta]).reshape(3, 1)


def draw_update_posterior(starting_state):
    """
    Plot the measurement update for the intuitive posterior.
    The figure will plot (z, y), where z is the sampled noisy measurement and 
    y is the y coordinate sampled from the posterior.
    The starting state is used to generate the noisy measurement after
    taking the control.
    """
    next_state = motion_fn(starting_state, 1)

    zs = []
    ys = [] 
    true_state = motion_fn(starting_state, 1)
    # simulate 300 starting states and noisy measurements.
    for i in range(1000):
        # sample measurement
        measurement = measurement_fn(next_state) + np.random.randn() * 0.1 
        y = np.random.randn() * 0.2
        theta = np.random.normal(0, 100)
        y = y + np.sin(np.deg2rad(theta))
        zs.append(measurement)
        ys.append(y)
    ax = plt.gca()
    plt.scatter(true_state[0, 0], true_state[1, 0], s=5, label="True state")
    plt.scatter(zs, ys, s=2, label="Meas. update for intuitive posterior")


def draw_posterior():
    # simulate 1000 samples
    xs = []
    ys = []
    thetas = []
    for i in range(0, 5000):
        x = np.random.randn() * 0.1 
        y = np.random.randn() * 0.1 
        _x = x

        theta = np.random.normal(0, 100)
        # motion model
        x = x + np.cos(np.deg2rad(theta))
        y = y + np.sin(np.deg2rad(theta))
        xs.append(x)
        ys.append(y)
        thetas.append(theta)
    plt.scatter(xs, ys, s=2, label="Inuitive posterior")


def get_motion_jacobian(theta, control):
    motion_jacobian = np.array([[1.0, 0.0, -control * np.sin(theta)],
                                [0.0, 1.0, control * np.cos(theta)],
                                [0.0, 0.0, 1.0]])
    return motion_jacobian


def plot_prediction_step(ekf):
    ekf.reset()
    ekf.predict(1.0)
    ax = plt.gca()
    plt.plot(ekf.mean[0, 0], ekf.mean[1, 0])
    plot_cov_ellipse(ekf.cov[:2, :2], ekf.mean[:2].flatten(), ax=ax, 
                     color='red', alpha=0.5, label='EKF prediction step')
    return ax

def plot_update_step(ekf, starting_state):
    # get next pose
    next_x = motion_fn(starting_state, 1)
    # plot the true state
    plt.scatter(next_x[0, 0], next_x[1, 0], s=5, label="True state")

    # add noise to the measurement
    measurement = measurement_fn(next_x) + np.random.randn() * 0.1

    # EKF update
    ekf.update(measurement)

    # plot
    ax = plt.gca()
    plt.plot(ekf.mean[0, 0], ekf.mean[1, 0])
    plot_cov_ellipse(ekf.cov[:2, :2], ekf.mean[:2].flatten(), ax=ax, 
                     color='green', alpha=0.8, label='EKF measurement update')
    return ax


if __name__ == "__main__":
    starting_state = sampale_starting_state()
    init_cov = np.array([[0.01, 0.0, 0.0],
                         [0.0, 0.01, 0.0],
                         [0.0, 0.0, 10000]])
    init_mean = np.array([[0.0], [0.0], [0.0]])

    # the control is flawless, there is little uncertainty
    R = np.array([[0.01, 0.0, 0.0], 
                  [0.0, 0.01, 0.0], 
                  [0.0, 0.0, 0.01]])
    
    G = get_motion_jacobian

    Q = 0.01
    H = np.array([[1.0, 0.0, 0.0]])
    ekf = ExtendedKalmanFiter(G, H, R, 0.01, init_mean, init_cov, 
                              motion_fn, measurement_fn)



    print("Plotting for question 3.8.4 (a)")
    draw_posterior()
    # draw_update_posterior(starting_state)
    plt.legend()
    plt.show()

    print("Plotting the intuitive and EKF posteriors for question 3.8.4 (b)")
    draw_posterior()
    plot_prediction_step(ekf)
    print("EKF mean and covariance before incorperating the measurement")
    print(ekf.mean)
    print(ekf.cov)
    plt.legend()
    plt.show()

    print("Plotting before and after the measurement update of EKF")
    ax = plot_prediction_step(ekf)
    ax = plot_update_step(ekf, starting_state)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-500, 500)
    plt.legend()
    plt.show()

    print("Plotting the posteriors after the measurement.")
    draw_update_posterior(starting_state)
    # we first need to reset the EKF
    ekf.reset()
    ekf.predict(1.0)
    ax = plot_update_step(ekf, starting_state)
    print("EKF mean and covariance after incorperating the measurement")
    print(ekf.mean)
    print(ekf.cov)
    ax.set_ylim(-5, 5)
    ax.set_xlim(-5, 5)
    plt.legend()
    plt.show()