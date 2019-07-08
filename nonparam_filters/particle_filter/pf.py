import numpy as np
from nonparam_filters.models import MotionModel, MeasurementModel


class ParticleFilter(object):
    """
    This class implements the particle filter algorithm for exercise 4.3.4 - 4.3.5
    """
    def __init__(self, num_particles, motion_model, measurement_model, state_dim, state_range, init_particles=None):
        self.num_particles = num_particles 
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        # represent a set of particles
        self.particles = np.array([np.random.uniform(state_range[i][0], high=state_range[i][1], size=self.num_particles) for i in range(state_dim)]).T if init_particles is None else init_particles 

    def sampling_step(self, u, z=None):
        """
        The sampling step does two things:
        1. Generate hypotheticle samples from the motion model given the state and control.
        2. Adding the importance factor to each of the particle. 
        """
        # initialize two arrays one for the particle one for the weight
        new_particles = np.zeros_like(self.particles) 
        weights = np.zeros(self.num_particles)

        for i in range(self.num_particles):
            # sample one particle using the motion model
            sample = self.motion_model.sample(u, self.particles[i])
            # append this to the new particle set
            new_particles[i] = sample

            # get the weight of the particle from the measurement model
            if z is not None:
                p = self.measurement_model(z, sample)
                # append the weight to the weight array
                weights[i] = p
        # finally, we normalize the weights
        if z is not None:
            weights = weights / np.sum(weights)

        self.particles = new_particles
        return weights
    
    def resampling_step(self, weights):
        # draw num_particles from the new particles according to the weights
        sample_indices = np.random.choice(np.arange(0, self.num_particles), size=self.num_particles, replace=True, p=weights)
        self.particles = self.particles[sample_indices]