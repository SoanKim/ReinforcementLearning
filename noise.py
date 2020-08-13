# OrnsteinUhlenbeckProcess

import numpy as np

class QUActionsNoise():
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=13-2, x0=None): # mu: mean for the noise, sigma: standard deviation, theta, timeparameter dt, x0 for starting value)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        # noise = QUActionNoise()
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)

        # current_noise = noise()
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)