import numpy as np
from scipy.stats import norm

class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma):
        """
        Initialize the Black-Scholes model.
        
        Parameters:
        - S: Current stock price (S_0)
        - K: Strike price
        - T: Time to maturity (in years)
        - r: Risk-free interest rate
        - sigma: Volatility of the stock price
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def compute_d1_d2(self):
        """
        Compute d1 and d2 for the Black-Scholes formula.
        """
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def call_price(self):
        """
        Calculate the price of a European call option using the Black-Scholes formula.
        """
        d1, d2 = self.compute_d1_d2()
        call_price = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        return call_price

    def put_price(self):
        """
        Calculate the price of a European put option using the Black-Scholes formula.
        """
        d1, d2 = self.compute_d1_d2()
        put_price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        return put_price

    def vega(self):
        d1, _ = self.compute_d1_d2()
        return self.S * norm.pdf(d1) * np.sqrt(self.T)

    def newton_raphson_method(self, C_market, sigma_guess=0.2):
        sigma = sigma_guess
        for _ in range(100):  # Maximum iterations
            price = self.call_price()
            v = self.vega()
            price_diff = C_market - price
            if abs(price_diff) < 1e-8:  # Convergence criteria
                self.sigma = sigma  # Update instance's sigma to found value
                return sigma
            sigma += price_diff / v  # Newton-Raphson update
        self.sigma = sigma  # Update instance's sigma to last estimate
        return sigma