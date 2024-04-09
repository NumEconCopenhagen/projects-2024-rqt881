import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class DowJonesDataAnalyzer:
    def __init__(self, start_date="2000-01-01", end_date="2023-12-01"):
        self.start_date = start_date
        self.end_date = end_date
        self.djia_tickers = [
            'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
            'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
            'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
        ]
        self.dow_prices = None
        self.excluded_tickers = []
        self.monthly_returns = None


    def fetch_and_filter_dow_data(self):
        """Fetches historical Dow Jones stock data and filters out tickers with incomplete data."""

        self.dow_prices = yf.download(self.djia_tickers, start=self.start_date, end=self.end_date)['Adj Close']
        self.dow_prices.dropna(axis=1, how='any', inplace=True)
        self.excluded_tickers = list(set(self.djia_tickers) - set(self.dow_prices.columns))
        print("Excluded tickers due to missing data:", self.excluded_tickers)

    def calculate_metrics(self):
        """Calculates date-related metrics and monthly returns for the Dow Jones data."""
        
        self.dow_prices_monthly = self.dow_prices.resample('ME').last()
        self.monthly_returns = self.dow_prices_monthly.pct_change().dropna()
        return {
            'monthly_returns': self.monthly_returns
        }

    def calculate_financial_metrics(self):
        """Calculates financial metrics including mean, variance-covariance matrix, standard deviation, and Sharpe ratios."""

        # Convert returns to percentages
        returns_matrix = self.monthly_returns * 100
        
        # Compute annualized mean, covariance matrix, and standard deviation
        mu = returns_matrix.mean() * 12  # Annualized mean, stored as an attribute
        Sigma = returns_matrix.cov() * 12  # Annualized covariance matrix, stored as an attribute
        SD = np.sqrt(np.diag(Sigma))  # Annualized standard deviation, stored as an attribute
        
        # Risk-free rate
        rf = 0
        
        # Calculate Sharpe Ratios
        sharpe_ratios = (mu - rf) / SD
        
        # Combine into DataFrame
        mu_table = pd.DataFrame({"Sample mean": mu, "Sharpe ratio": sharpe_ratios})
        
        # Find tickers with max and min Sharpe ratio
        ticker_max_sharpe = mu_table['Sharpe ratio'].idxmax()
        ticker_min_sharpe = mu_table['Sharpe ratio'].idxmin()
        
        # Set attributes
        self.mu = mu
        self.Sigma = Sigma
        self.SD = SD
        
        # Optionally print or return this table
        print(mu_table)
        print("Ticker with max Sharpe ratio:", ticker_max_sharpe)
        print("Ticker with min Sharpe ratio:", ticker_min_sharpe)
        
        # Print the variance-covariance matrix
        print("\nVariance-Covariance Matrix (Annualized):")
        print(Sigma)

    
    def compute_efficient_frontier(self, mu, Sigma, simulated=False):
        """
        Computes the efficient frontier for custom mu and Sigma.
        
        :param mu: Expected annual returns for each asset.
        :param Sigma: Annual covariance matrix of returns.
        :param simulated: indicating whether the computation is simulated or theoretical.
        """
        # Initialize variables for efficient frontier calculation
        N = len(mu)  # Number of assets
        iota = np.ones(N)  # Vector of ones
        Sigma_inv = np.linalg.inv(Sigma)  # Inverse of the covariance matrix
        
        # Compute weights for the Minimum Variance Portfolio (MVP)
        mvp_weights = Sigma_inv @ iota / (iota @ Sigma_inv @ iota)
        mvp_mu = mvp_weights.T @ mu  # Expected return for MVP
        mvp_sd = np.sqrt(mvp_weights.T @ Sigma @ mvp_weights)  # Standard deviation for MVP
            
        # Define parameters for the efficient frontier
        mu_bar = 2 * (mvp_weights @ mu)  # Adjusted mean return
        C = iota @ Sigma_inv @ iota  # Scalar
        D = iota @ Sigma_inv @ mu  # Dot product
        E = mu @ Sigma_inv @ mu  # Scalar
        
        # Compute the adjustment factor lambda_tilde
        lambda_tilde = 2 * (mu_bar - D / C) / (E - D**2 / C)
        
        # Calculate the weights for the Efficient Frontier Portfolio (EFP)
        efp_weights = mvp_weights + lambda_tilde / 2 * (Sigma_inv @ mu - D * mvp_weights)
        
        # Correctly generate efficient frontier points across a range of 'c' values
        start, end, num_points = (-4, 3, 150)  # Correctly defined range for 'c'
        c_values = np.linspace(start, end, num_points)  # Generate a sequence of 'c' values
        ef_points = pd.DataFrame(index=c_values, columns=['mu', 'sd'])
        
        for c in c_values:
            w = (1 - c) * efp_weights + c * mvp_weights
            portfolio_mu = w.T @ mu
            portfolio_sd = np.sqrt(w.T @ Sigma @ w)
            ef_points.loc[c, 'mu'] = portfolio_mu   
            ef_points.loc[c, 'sd'] = portfolio_sd    

        # Compute efficient tangent portfolio weights
        wtgc = np.linalg.solve(Sigma, mu)
        wtgc /= np.sum(wtgc)

        # Calculate portfolio metrics
        expected_return = np.dot(wtgc, mu)
        standard_deviation = np.sqrt(np.dot(wtgc.T, np.dot(Sigma, wtgc)))
        sharpe_ratio = expected_return / standard_deviation

        # Save the efficient frontier points with a distinction for simulated or theoretical
        if simulated:
            self.ef_points_simulated = ef_points
            self.min_simulated = {'mu': mvp_mu, 'sd': mvp_sd}  
            self.tan_simulated= {'mu': expected_return, 'sd': standard_deviation}  
            self.sharpe_ratio_simulated= sharpe_ratio
            self.wtgc=wtgc

        else:
            self.ef_points_theoretical = ef_points
            self.min_theoretical = {'mu': mvp_mu, 'sd': mvp_sd}  
            self.tan_theoretical= {'mu': expected_return, 'sd': standard_deviation}  
            self.sharpe_ratio_theoretical= sharpe_ratio
            self.wtgc=wtgc


    def simulate_returns(self, periods=100, expected_returns=None, covariance_matrix=None):

        """    Simulates asset returns based on provided mean returns and covariance matrix."""

        return np.random.multivariate_normal(mean=expected_returns, cov=covariance_matrix, size=periods)