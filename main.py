import os
import numpy as np

class Bayes:
    def __init__(self, hypothesis, priors, observations, likelihood_array):
        self.hypothesis = hypothesis
        self.observations = observations
        self.priors = priors
        self.likelihood_array = likelihood_array

    def likelihood(self, observations, hypothesis):
        obs_idx = self.observations.index(observations)
        hypo_idx = self.hypothesis.index(hypothesis)
        return self.likelihood_array[hypo_idx][obs_idx]

    def norm_constant(self, observation):
        total = 0
        for i, hypothesis in enumerate(self.hypothesis):
            total += self.priors[i] * self.likelihood(observation, hypothesis)
        return total

    def single_posterior_update(self, observation, prior):
        posterior = []
        for i, hypothesis in enumerate(self.hypothesis):
            posterior.append(prior[i] * self.likelihood(observation, hypothesis) / self.norm_constant(observation))
        return posterior

    def compute_posterior(self, observations_list):
        for observation in observations_list:
            self.priors = self.single_posterior_update(observation, self.priors)
        assert np.isclose(np.sum(self.priors), 1.), "The sum of the posteriors is not equal to 1"
        return self.priors

if __name__ == "__main__":
    hypothesis = ["bowl1", "bowl2"]
    observations = ["chocolate", "vanilla"]
    likelihood = [[15/50, 35/50],[30/50, 20/50]]
    priors = [0.5, 0.5]

    b = Bayes(hypothesis, priors, observations, likelihood)
    l = b.likelihood("chocolate", "bowl1")
    print("likelihood(chocolate, bowl1) = %s " % l)

    n_c = b.norm_constant("vanilla")
    print("normalizing constant for vanilla: %s" % n_c)

    p_1 = b.single_posterior_update("vanilla", [0.5, 0.5])
    print("vanilla - posterior: %s" % p_1)

    p_2 = b.compute_posterior(["chocolate", "vanilla"])
    print("chocolate, vanilla - posterior: %s" % p_2)

    ## Fill in questions
    f = open("questions.txt", "w+")
    f.write("%.3f \n" % p_1[0])
    f.write("%.3f \n" % p_2[1])
    f.close()

    hypos = ["Beginner", "Intermediate", "Advanced", "Expert"]
    priors = [0.25, 0.25, 0.25, 0.25]
    obs = ["Yellow", "Red", "Blue", "Black", "White"]
    likelihood = [[0.05, 0.1, 0.4, 0.25, 0.2], [0.1, 0.2, 0.4, 0.2, 0.1], [0.2, 0.4, 0.25, 0.1, 0.05], [0.3, 0.5, 0.125, 0.05, 0.025]]

    b = Bayes(hypos, priors, obs, likelihood)

    p_2 = b.compute_posterior(["Yellow", "White", "Blue", "Red", "Red", "Blue"])
    print("Yellow, White, Blue, Red, Red, Blue - posterior: %s" % p_2)


    ## Fill in questions
    f = open("questions.txt", "a+")
    f.write("%.3f \n" % p_2[1])
    f.write("%s \n" % hypos[np.argmax(p_2)])
    f.close()