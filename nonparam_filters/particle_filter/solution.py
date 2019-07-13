import numpy as np
import matplotlib.pyplot as plt
from nonparam_filters.models import MotionModel, MeasurementModel, NonLinearMotionModel
from nonparam_filters.particle_filter.pf import ParticleFilter


def run_sol_four():
    # motion model
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    B = np.array([[0.5], [1.0]])
    R = np.zeros((2, 2))
    R[0, 0] = 0.25
    R[1, 1] = 1
    motion = MotionModel(A, B, R) 

    # measurement model
    C = np.array([[1.0, 0.0]])
    Q = 10
    measurement_model = MeasurementModel(C, Q)

    # particle filter
    n_particles = 1000
    # sample the initial particles according to the initial belief, the same as the KF exercise
    init_cov = np.identity(2) * 1e-4 
    init_mean = np.zeros(2)
    init_particles = np.random.multivariate_normal(init_mean, init_cov, size=n_particles)
    pf = ParticleFilter(n_particles, motion, measurement_model, 2, [(-3, 3.), (-3., 3)], init_particles)

    print("Plotting the posterior given the control for t=0...5.")
    state = init_mean
    for i in range(0, 5):
        weights = pf.sampling_step(0.0, None)
        particles = pf.particles
        plt.scatter(particles[:, 0], particles[:, 1], s=1, alpha=0.5, c='red') 
        plt.xlim(-30, 30)
        plt.ylim(-30, 30)
        plt.show()
        state = motion.mean(state, np.random.randn()) 
    
    print("Plotting the posterior after seeing the measurement at t=6.")
    measurement = measurement_model.sample(state) 
    print("The measurement is at", measurement)
    weights = pf.sampling_step(0.0, measurement)
    plt.scatter(pf.particles[:, 0], pf.particles[:, 1], s=1, label='Before resampling')
    pf.resampling_step(weights)
    plt.scatter(pf.particles[:, 0], pf.particles[:, 1], s=1, label='After resampling')
    plt.legend()
    plt.show()
    

def run_sol_five():
    initial_mean = np.array([0.0, 0.0, 0.0])
    initial_cov = np.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 10000]])

    motion_model = NonLinearMotionModel()

    C = np.array([1.0, 0.0, 0.0]).reshape(1, -1)
    Q = 0.01
    measurement_model = MeasurementModel(C, Q)
    n_particles = 5000
    # initialize the belief
    init_particles = np.random.multivariate_normal(initial_mean, initial_cov, size=n_particles)
    pf = ParticleFilter(n_particles, motion_model, measurement_model, None, None, init_particles)

    # move the robot forward
    control = 1.0
    next_state = motion_model.sample(control, initial_mean)
    # observe the measurement
    measurement = measurement_model.sample(next_state)

    # run the sampling step with the control and the measurement
    print("Plotting the solution for 4.6.4, the sampling step.")
    weights = pf.sampling_step(control, measurement)
    plt.scatter(pf.particles[:, 0], pf.particles[:, 1], s=1.0, label="Particles of t=1, \nsampling step.")
    plt.legend()
    plt.show()

    # run the resampling step to update the belief
    print("Plotting the solution for 4.6.4, the resampling step.")
    pf.resampling_step(weights)
    plt.scatter(pf.particles[:, 0], pf.particles[:, 1], s=1.0, label="Particles of t=1, \nresampling step.")
    plt.scatter(measurement, next_state[1], label="Measurement on x=%.2f" %measurement)
    plt.scatter(next_state[0], next_state[1], label="True state.")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # run_sol_four()
    run_sol_five()