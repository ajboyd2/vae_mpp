import numpy as np
from numpy import random as rand
from abc import ABC, abstractmethod


class SoftPlus:

    def __init__(self, K):
        self.K = K
        self.weights = np.ones(K)  # (rand.rand(K) / 4) + 0.75

    def __call__(self, t):
        s = self.weights
        return s * np.log(1 + np.exp(t / s))


class PointProcess(ABC):

    def __init__(self, conditional=False):
        self.conditional = conditional

    @abstractmethod
    def intensity(self, t, batch):
        pass

    @abstractmethod
    def update(self, t, k, batch):
        pass

    def generate_point_pattern(self, base_intensity, right_limit, batch_size):
        '''
        Generates a temporal point pattern for a given intensity using a thinning process.

        Arguments:
            base_intensity (float) - Defines a single intensity rate that dominates the entire intensity
                function for process being simulated across the time window
            right_limit (float) - The function generates points within the time window [0, right_limit)
            batch_size (int) - The size of the batch of independent samples being generated

        Returns:
            A tuple of tuples containing three arrays, each length n.
            The first element in the nested tuple contains the generated arrival event times.
            The second element contains the associated mark id.
            The third element contains the intensity values at the associated time.
            Each nested tuple is a sample and there will be 'batch_size' of them.
        '''

        initial_point_count = base_intensity * right_limit
        candidate_points = []
        max_sum = np.zeros(batch_size)
        while True:
            random_quants = rand.rand(batch_size, initial_point_count)  # uniform [0,1) draws
            random_homo_pp = -1 * np.log(1 - random_quants) / base_intensity  # quantile for exp dist
            candidate_points.append(random_homo_pp)

            # check if we have reached the right_limit yet for all samples
            # (could speed up for checking individual samples)
            max_sum += np.sum(random_homo_pp, axis=1)
            if np.all(max_sum > right_limit):
                break

        candidate_points = np.concatenate(candidate_points, axis=1)  # shape = batch_size, samples
        candidate_arrival_times = np.cumsum(candidate_points, axis=1)  # calculate the actual event times

        if self.conditional:
            marks = []
            arrival_times = []
            intensities = []
            batch_size, sample_size = candidate_arrival_times.shape
            generated_rvs = rand.rand(batch_size, sample_size)
            for b in range(batch_size):
                for s in range(sample_size):
                    t = candidate_arrival_times[b, s]
                    rv = generated_rvs[b, s]
                    if t < right_limit:
                        intensity_vector = self.intensity(t, b)
                        prob = intensity_vector.sum() / base_intensity
                        if rv < prob:
                            arrival_times.append(t)
                            intensities.append(intensity_vector)
                            intensity_prob = intensity_vector / intensity_vector.sum()
                            mark = rand.choice(intensity_vector.shape[0], p=intensity_prob)
                            marks.append(mark)
                            self.update(t, mark, b)
            marks = np.array(marks)
            arrival_times = np.array(arrival_times)
            intensity_values = np.array(intensities)
        else:
            intensity_values = self.intensity(candidate_arrival_times)  # shape = sample_size, samples, marks
            K = intensity_values.shape[2]
            probs = intensity_values.sum(axis=2) / base_intensity

            # thinning
            accepted_point_indices = (rand.rand(*candidate_arrival_times.shape) < probs) & (
                    candidate_arrival_times < right_limit)

            arrival_times = candidate_arrival_times[accepted_point_indices]
            intensity_values = intensity_values[accepted_point_indices]

            intensity_probs = (
                        intensity_values / intensity_values.sum(axis=1)[:, np.newaxis])  # normalize at each point
            marks = np.apply_along_axis(lambda x: rand.choice(K, p=x), axis=1, arr=intensity_probs)

        split_indices = np.where(np.diff(arrival_times) < 0)[0] + 1

        return (np.split(arrival_times, split_indices),
                np.split(marks, split_indices),
                np.split(intensity_values, split_indices))

class HomogenousPoissonProcess(PointProcess):

    def __init__(self, K, scale):
        super().__init__(conditional=False)
        self.K = K
        self.scale = rand.rand(K) * scale
        
    def intensity(self, t, batch=None):
        scale = self.scale
        x = np.repeat(t[:, :, np.newaxis], self.K, axis=2)
        return scale * ((x*0) + 1)  # just to get the same size

    def update(self, t, k, batch):
        return

    def __repr__(self):
        return "K={}\nScale={}".format(self.K, self.scale)

class InhomogenousPoissonProcess(PointProcess):

    def __init__(self, K, scale, right_limit):
        super().__init__(conditional=False)
        self.K = K
        self.scale = scale
        self.mu = rand.rand(K) * right_limit
        self.sigma = 3 + rand.rand(K) * 10

    def intensity(self, t, batch=None):
        sigma = self.sigma
        mu = self.mu
        scale = self.scale
        x = np.repeat(t[:, :, np.newaxis], self.K, axis=2)
        return scale / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

    def update(self, t, k, batch):
        return

    def __repr__(self):
        return "K={}\nScale={}\nMu={}\nSigma={}".format(self.K, self.scale, self.mu, self.sigma)


class SelfCorrectingProcess(PointProcess):

    def __init__(self, K):
        super().__init__(conditional=True)
        self.K = K
        self.func = SoftPlus(K)
        self.eta = (rand.rand(K) - 0.1) * 3 / K
        self.gamma = (rand.rand(K, K) - 0.2) / K  # influence of events to row, from col
        self.history_times = {}
        self.history_marks = {}

    def clear(self):
        self.history_times = {}
        self.history_marks = {}

    def intensity(self, t, batch):
        # t: single value
        # intensity output shape: K

        if batch in self.history_times:
            times = self.history_times[batch]
            marks = self.history_marks[batch]

            valid_obs = times < t
            gammas = self.gamma[:, marks]
            vals = np.exp((self.eta * t) - np.sum(gammas * valid_obs, axis=1))
        else:
            vals = np.exp(self.eta * t)
        return self.func(vals)

    def update(self, t, k, batch):
        if batch in self.history_times:
            self.history_times[batch] = np.append(self.history_times[batch], t)
            self.history_marks[batch].append(k)
        else:
            self.history_times[batch] = np.array([t])
            self.history_marks[batch] = [k]

    def __repr__(self):
        return str(self.__dict__)


class SelfExcitingProcess(PointProcess):

    def __init__(self, K):
        super().__init__(conditional=True)
        self.K = K
        self.func = SoftPlus(K)
        self.mu = (rand.rand(K) - 0.5) * 3 / K
        self.alpha = (rand.rand(K, K) - 0.5) / K  # influence of events to row, from col
        self.delta = rand.rand(K, K) * 3 * K
        self.history_times = {}
        self.history_marks = {}

    def clear(self):
        self.history_times = {}
        self.history_marks = {}

    def intensity(self, t, batch):
        # t: single value
        # intensity output shape: K

        if batch in self.history_times:
            times = self.history_times[batch]
            marks = self.history_marks[batch]

            valid_obs = times < t
            diffs = t - times
            alphas = self.alpha[:, marks]
            deltas = self.delta[:, marks]
            vals = self.mu + np.sum(valid_obs * alphas * np.exp(-1 * deltas * diffs), axis=1)
        else:
            vals = self.mu
        return self.func(vals)

    def update(self, t, k, batch):
        if batch in self.history_times:
            self.history_times[batch] = np.append(self.history_times[batch], t)
            self.history_marks[batch].append(k)
        else:
            self.history_times[batch] = np.array([t])
            self.history_marks[batch] = [k]

    def __repr__(self):
        return str(self.__dict__)


class SelfModulatingProcess(PointProcess):

    def __init__(self, K):
        super().__init__(conditional=True)
        self.K = K
        self.func = SoftPlus(K)
        self.mu = (rand.rand(K) - 0.5) * 3 / K
        self.alpha = (rand.rand(K, K) - 0.5) / K  # influence of events to row, from col
        self.delta = rand.rand(K, K) * 3 * K
        self.eta = (rand.rand(K) - 0.1) * 3 / K
        self.gamma = (rand.rand(K, K) - 0.2) / K  # influence of events to row, from col
        self.proportion = rand.rand(1)[0]
        self.history_times = {}
        self.history_marks = {}

    def clear(self):
        self.history_times = {}
        self.history_marks = {}

    def intensity(self, t, batch):
        # t: single value
        # intensity output shape: K

        if batch in self.history_times:
            times = self.history_times[batch]
            marks = self.history_marks[batch]

            valid_obs = times < t
            diffs = t - times
            alphas = self.alpha[:, marks]
            deltas = self.delta[:, marks]
            se_vals = self.mu + np.sum(valid_obs * alphas * np.exp(-1 * deltas * diffs), axis=1)
            gammas = self.gamma[:, marks]
            sc_vals = np.exp((self.eta * t) - np.sum(gammas * valid_obs, axis=1))
        else:
            se_vals = self.mu
            sc_vals = np.exp(self.eta * t)
        p = self.proportion
        return self.func(p * se_vals + (1 - p) * sc_vals)

    def update(self, t, k, batch):
        if batch in self.history_times:
            self.history_times[batch] = np.append(self.history_times[batch], t)
            self.history_marks[batch].append(k)
        else:
            self.history_times[batch] = np.array([t])
            self.history_marks[batch] = [k]

    def __repr__(self):
        return str(self.__dict__)

def PointProcessFactory(pp_args):

    pp_obj = SelfExcitingProcess(pp_args["K"])

    for k,v in pp_args.items():
        pp_obj.__dict__[k] = v

    return pp_obj

