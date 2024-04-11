from scipy.stats import beta, beta_gen
import numpy as np
import pandas as pd

class SimpleRateExperiment:
    """
    A class to conduct Bayesian A/B testing for categorical experiment outcomes based on rate metrics such as click-through rates.
    
    This class calculates the posterior distributions for conversion rates using a Beta distribution, 
    samples from these distributions to estimate performance differences (lifts), and then makes decisions 
    based on expected losses and specified thresholds of caring.
    """

    def __init__(self, counts: str, totals: str, group: str, a_prior: int = 1, b_prior: int = 1, size: int = 1000):
        """
        Constructs all the necessary attributes for the SimpleRateExperiment object.
        
        Parameters:
            counts (str): The column name for the count of successes (e.g. clicks).
            totals (str): The column name for the total attempts (e.g. impressions).
            group (str): The column name to group data by, typically representing different experiment variants.
            a_prior (int): The alpha parameter of the prior Beta distribution (default 1).
            b_prior (int): The beta parameter of the prior Beta distribution (default 1).
            size (int): The number of samples to draw from each posterior (default 1000).
        """
        self.counts, self.totals = counts, totals
        self.group = group
        self.a_prior = a_prior
        self.b_prior = b_prior
        self.size = size
        self.posterior: dict[str, beta_gen] = {}
        self.samples: dict[str, np.ndarray] = {}
        self.expected_losses: dict[str, float] = {}

    def _compute_posterior(self, data: pd.DataFrame):
        """
        Computes the posterior distribution of each group using the Beta distribution based on the provided data.

        Parameters:
            data (DataFrame): Pandas DataFrame containing experiment data with columns for counts and totals.
        """
        for name, group in data.groupby(self.group):
            self.posterior[name] = beta(self.a_prior + group[self.counts], self.b_prior + group[self.totals] - group[self.counts])
    
    def _sample_posterior(self):
        """
        Samples from the posterior distributions of each group.
        """
        for name, post in self.posterior.items():
            self.samples[name] = post.rvs(self.size)

    def _compute_lift(self, current: str, alternative: str) -> np.ndarray:
        """
        Computes the lift of an alternative group over the current group by calculating the difference in samples from their posterior distributions.

        Parameters:
            current (str): The current group name.
            alternative (str): The alternative group name to compare against.

        Returns:
            np.ndarray: Differences in sample values representing the lift.
        """
        return self.samples[alternative] - self.samples[current]

    def decide(self, thres_caring: float = 0.01):
        """
        Decides the best group based on the expected loss being below a certain threshold.

        Parameters:
            thres_caring (float): The threshold below which an alternative's expected loss is considered acceptable.
        """
        # Compute expected loss
        for current, post in self.posterior.items():
            lifts = np.asarray([self._compute_lift(current, alt) for alt in self.posterior.keys()]) 
            self.expected_losses[current] = -(-lifts).min(axis=0).mean()
        
        # Check to see which alternatives if any meet threshold of caring
        threshold_met = False
        for alt, expected_loss in self.expected_losses.items():
            if expected_loss < thres_caring:
                print(f"{alt} is acceptable")
                threshold_met = True
        if threshold_met:
            self.decision = min(self.expected_losses, key=self.expected_losses.get)
            print(f"We choose {self.decision}.")
        else:
            print("No alternatives met the threshold of caring.")
    
    def run_test(self, data: pd.DataFrame, thres_caring: float = None):
        """
        Executes the test by computing posterior distributions, sampling them, and possibly making a decision based on a threshold.

        Parameters:
            data (DataFrame): Pandas DataFrame containing experiment data.
            thres_caring (float, optional): The threshold of caring to use when making a decision. If None, no decision is made.
        """
        self._compute_posterior(data)
        self._sample_posterior()

        if thres_caring is not None:
            self.decide(thres_caring)