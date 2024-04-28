import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import ipywidgets as widgets
from IPython.display import display

class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma, dividend_yield):
        """
        Initialize the Black-Scholes model.
        
        Parameters:
        - S: Current stock price (S_0)
        - K: Strike price
        - T: Time to maturity (in years)
        - r: Risk-free interest rate
        - sigma: Volatility of the stock price
        - dividend_yield: Annual dividend yield
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.dividend_yield = dividend_yield  


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
    
    def plot_option_prices(S, K, T, r, sigma):
        """
        Plot Black-Scholes prices for European call and put options.

        This function displays a bar chart of the prices for call and put options based on provided
        parameters for stock price (S), strike price (K), time to maturity (T),
        risk-free rate (r), and volatility (sigma). Assumes a dividend yield of 0.0.

        Parameters:
        - S: Current stock price
        - K: Option strike price
        - T: Time until option expiration in years
        - r: Annual risk-free rate
        - sigma: Stock price volatility
        """

        model = BlackScholesModel(S, K, T, r, sigma, dividend_yield=0.0)
        call_price = model.call_price()
        put_price = model.put_price()
        
        plt.figure(figsize=(10, 5))
        bar = plt.bar(['Call Option', 'Put Option'], [call_price, put_price], color=['purple', 'pink'])
        plt.title('Black-Scholes Option Prices')
        plt.ylabel('Option Price')
        plt.ylim(0, max(call_price, put_price) + 10)
        for rect in bar:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')
        plt.show()


    def binomial_tree_option_price(self, N=100, option_type='call'):
        """
        Calculate the price of a European option using a binomial tree model.
        
        Parameters:
        - N: Number of steps in the binomial tree
        - option_type: Type of option ('call' or 'put')
        
        Returns:
        - float: Price of the option
        """
        dt = self.T / N  # Time step
        u = np.exp(self.sigma * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp(self.r * dt) - d) / (u - d)  # Probability of an up move

        # Initialize the prices at maturity
        prices = self.S * u**np.arange(N, -1, -1) * d**np.arange(0, N + 1)

        # Calculate option value at maturity
        if option_type == 'call':
            values = np.maximum(prices - self.K, 0)
        else:
            values = np.maximum(self.K - prices, 0)

        # Backward induction for option price
        for i in range(N - 1, -1, -1):
            values = (p * values[:-1] + (1 - p) * values[1:]) * np.exp(-self.r * dt)

        return values[0]


    def plot_binomial_tree_prices(self, S, K, T, r, sigma, K_range):
        """
        Plots the option prices for different strike prices using the binomial tree model.
        
        Parameters:
        - S: Current stock price (S_0)
        - K: Strike price
        - T: Time to maturity (in years)
        - r: Risk-free interest rate
        - sigma: Volatility of the stock price
        - K_range: Range of strike prices
        """
        model = BlackScholesModel(S, K, T, r, sigma, dividend_yield=0.0)
        call_prices = []
        put_prices = []
        for K_val in K_range:
            model.K = K_val
            call_price = model.binomial_tree_option_price(N=100, option_type='call')
            put_price = model.binomial_tree_option_price(N=100, option_type='put')
            call_prices.append(call_price)
            put_prices.append(put_price)
        
        plt.figure(figsize=(10, 5))
        plt.plot(K_range, call_prices, label='Call Option Price', color="purple")
        plt.plot(K_range, put_prices, label='Put Option Price', color="pink")
        plt.title('Option Prices for Different Strike Prices using Binomial Tree Model')
        plt.xlabel('Strike Price')
        plt.ylabel('Option Price')
        plt.legend()
        plt.grid(True)
        plt.show()

    def binomial_tree_American_option_price(self, N=100, option_type='call'):
        """
        Calculate the price of an American option using a binomial tree model, incorporating dividends.
        
        Parameters:
        - N: Number of steps in the binomial tree
        - option_type: Type of option ('call' or 'put')
        
        Returns:
        - float: Price of the option
        """
        dt = self.T / N  # Time step
        u = np.exp(self.sigma * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp((self.r - self.dividend_yield) * dt) - d) / (u - d)  # Risk-neutral probability, adjusted for dividends

        # Initialize the prices at maturity, adjusted for dividends
        prices = self.S * np.exp(-self.dividend_yield * dt) * u**np.arange(N, -1, -1) * d**np.arange(0, N + 1)

        # Initialize option values at maturity
        if option_type == 'call':
            option_values = np.maximum(prices - self.K, 0)
        else:
            option_values = np.maximum(self.K - prices, 0)

        # Backward induction for option price
        for i in range(N - 1, -1, -1):
            # Calculate the option value at the node, considering the possibility of early exercise
            option_values = (p * option_values[:-1] + (1 - p) * option_values[1:]) * np.exp(-self.r * dt)
            
            # Early exercise value, adjusting asset prices for dividends
            asset_prices = self.S * np.exp(-self.dividend_yield * dt * i) * (u**np.arange(i, -1, -1) * d**np.arange(0, i + 1))
            if option_type == 'call':
                exercise_values = np.maximum(asset_prices - self.K, 0)
            else:
                exercise_values = np.maximum(self.K - asset_prices, 0)
            
            # Choose the maximum of exercising now or holding the option
            option_values = np.maximum(option_values, exercise_values)

        return option_values[0]