import numpy as np
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_cov_ellipse

# 1 second interval for two time points
delta_t = 1.0
# variance for the acceleration control
sigma = 1
# mean of the control
mean_u = 0.0

# random color for the plot
random_color = ['r', 'g', 'b', 'gray', 'yellow']


class KalmanFilter(object):
    """
    An implementation of the Kalman Filter algorithm. This implementation aims at solving 
    Exercise 3.8.1 and 3.8.2 of probablistic robotics book.
    A, B are the linear system matrices. R is the covariance matrix for the linear system. 
    C is the matrix for the measurement. Q is the covariance matrix for the measurement. 
    mean and cov parameterize the initial belief.
    """

    def __init__(self, A, B, C, R, Q, mean, cov):
        self.A = A
        self.B = B
        self.C = C
        self.R = R
        self.Q = Q
        self.init_mean = mean
        self.init_cov = cov 
        self.mean = mean
        self.cov = cov
    
    def predict(self, u):
        """
        The prediction step that takes the previous belief parameterized by
        prev_mean and prev_cov and the current control u to calculate the
        posterior probability over the next state given the control and the
        previous belief.
        """
        # if u is a scalar, need to wrap the control
        self.mean = self.A @ self.mean + self.B @ [[u]]
        self.cov = self.A @ self.cov @ self.A.T + self.R
    
    def measurement_update(self, measurement):
        """
        The measurement update step that incorperates the sensor measurement. 
        """
        measurement = np.array([[measurement]])
        gain = self.cov @ self.C.T @ np.linalg.inv(self.C @ self.cov @ self.C.T + self.Q)
        self.mean = self.mean + gain @ (measurement - self.C @ self.mean)
        self.cov = (np.identity(2) - gain @ self.C) @ self.cov
    
    def update(self, u, measurement):
        self.predict(u)
        self.measurement_update(measurement)
    
    def reset(self):
        self.cov = self.init_cov 
        self.mean = self.init_mean


def run_pred(kalman_filter, n_times=6):
    """
    Run n times of the Kalman filter prediction step.
    """
    kalman_filter.reset()
    for t in range(1, n_times + 1):
        ax = plt.gca()
        kalman_filter.predict(mean_u)
        print('Kalman filter covariance at time', i, 'is', kalman_filter.cov)
    cov = kalman_filter.cov
    mean = kalman_filter.mean
    plot_cov_ellipse(cov, mean, ax=ax, nstd=1, alpha=0.3, lw=2, facecolor=random_color[t-1], edgecolor='black')
    plt.scatter(mean[0, 0], mean[1,0], s=3)


def run_kf(kalman_filter):
    """
    Run the kalman filter with predict step up to the 5th time points then incorperate
    sensor measurement. Plot the confidence ellipse before and after incorperating the
    sensor measurement.
    In addition, plot the confidence ellipse and actual state for every time point in a sequence.
    """
    state = np.array([[0.0], [0.0]])
    kalman_filter.reset()

    for t in range(1, 6):
        ax = plt.gca()
        u = np.random.randn() * np.sqrt(sigma)
        # the true motion of the car
        state = kalman_filter.A @ state + kalman_filter.B @ [[u]]
        # prediction
        kalman_filter.predict(mean_u)

    print("The covariance before measurement update", kalman_filter.cov)
    # measurement update, the measurement is corrupted with noise
    measurement = state[0, 0] + np.random.randn() * np.sqrt(10.0) 

    # plot the motion of the car
    plt.scatter(state[0, 0], state[1, 0], s=10, label='true state, t=%i' %t, c='blue')

    # plot the covariance
    plot_cov_ellipse(kalman_filter.cov, kalman_filter.mean, ax=ax, nstd=1, alpha=0.3, lw=2, facecolor='gray', edgecolor='black')
    plt.scatter(kalman_filter.mean[0, 0], kalman_filter.mean[1, 0], s=10, label='mean before update, t=%i' %t, c='gray')

    # the velocity is not measured
    plt.scatter(measurement, state[1, 0], s=10, label='measurement, t=%i' %t, c='green')
    kalman_filter.measurement_update(measurement)

    print("The covariance after measurement update", kalman_filter.cov)

    plot_cov_ellipse(kalman_filter.cov, kalman_filter.mean, ax=ax, nstd=1, alpha=0.3, lw=2, facecolor='red', edgecolor='black')
    plt.scatter(kalman_filter.mean[0, 0], kalman_filter.mean[1, 0], s=10, label='mean after update, t=%i' %t, c='red')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    A = np.array([[1.0, delta_t], [0.0, 1.0]])
    B = np.array([[0.5 * delta_t**2], [delta_t]])
    R = np.zeros((2, 2))
    R[0, 0] = 0.25
    R[1, 1] = 1.0
    # R = np.dot(B, B.T)
    C = np.array([[1.0, 0.0]])
    Q = np.array([[10.0]]) 
    mean = np.array([[0.0], [0.0]])
    cov = np.array([[1e-4, 0.0], [0.0, 1e-4]])
    kalman_filter = KalmanFilter(A, B, C, R, Q, mean, cov)
    for i in range(1, 6):
        run_pred(kalman_filter, n_times=i)
    plt.show()
    run_kf(kalman_filter)

