import numpy as np
import pandas as pd
from scipy.optimize import fsolve, minimize, minimize_scalar
from types import SimpleNamespace
import matplotlib.pyplot as plt 

class Problem1:
    def __init__(self):
        """
        Initialize the problem class with default parameters.

        Attributes:
        par (SimpleNamespace): Namespace containing all the parameters for the model.
            - A (float): Total factor productivity.
            - gamma (float): Output elasticity of labor.
            - alpha (float): Share of income spent on good 1.
            - nu (float): Disutility of labor parameter.
            - epsilon (float): Elasticity of labor supply.
            - tau (float): Initial tax rate.
            - T (float): Initial transfer.
            - kappa (float): Social welfare weight on good 2.
            - w (float): Wage rate (numeraire).
        """

        self.par = SimpleNamespace()
        self.par.A = 1.0  # Total factor productivity
        self.par.gamma = 0.5  # Output elasticity of labor
        self.par.alpha = 0.3  # Share of income spent on good 1
        self.par.nu = 1.0  # Disutility of labor parameter
        self.par.epsilon = 2.0  # Elasticity of labor supply
        self.par.tau = 0.0  # Initial tax rate
        self.par.T = 0.0  # Initial transfer
        self.par.kappa = 0.1  # Social welfare weight on good 2
        self.par.w = 1.0  # Wage rate (numeraire)

    # Functions for firm behavior
    def optimal_labor(self, w, p, A, gamma):
        """ Calculate the optimal labor demand given parameters """
        return (p * A * gamma / w) ** (1 / (1 - gamma))

    def optimal_output(self, A, labor, gamma):
        """ Calculate the optimal output given labor input """
        return A * labor ** gamma

    def optimal_profits(self, w, p, A, gamma):
        """ Calculate the optimal profits given parameters """
        return (1 - gamma) / gamma * w * (p * A * gamma / w) ** (1 / (1 - gamma))

    # Consumer behavior
    def consumption1(self, w, ell, T, pi1, pi2, p1, alpha):
        """ Calculate the consumption of good 1 """
        return alpha * (w * ell + T + pi1 + pi2) / p1

    def consumption2(self, w, ell, T, pi1, pi2, p2, tau, alpha):
        """ Calculate the consumption of good 2 """
        return (1 - alpha) * (w * ell + T + pi1 + pi2) / (p2 + tau)

    # Utility maximization
    def utility_maximization(self, p1, p2, w, A, gamma, alpha, nu, epsilon, tau, T):
        """ Maximize utility given parameters and return optimal labor and consumption """
        pi1 = self.optimal_profits(w, p1, A, gamma)
        pi2 = self.optimal_profits(w, p2, A, gamma)

        def utility(ell):
            """ Utility function based on labor supply """
            c1 = self.consumption1(w, ell, T, pi1, pi2, p1, alpha)
            c2 = self.consumption2(w, ell, T, pi1, pi2, p2, tau, alpha)
            return np.log(c1 ** alpha * c2 ** (1 - alpha)) - nu * ell ** (1 + epsilon) / (1 + epsilon)

        res = minimize_scalar(lambda ell: -utility(ell), bounds=(0, 10), method='bounded')
        ell_star = res.x
        c1_star = self.consumption1(w, ell_star, T, pi1, pi2, p1, alpha)
        c2_star = self.consumption2(w, ell_star, T, pi1, pi2, p2, tau, alpha)

        return ell_star, c1_star, c2_star
    

    # Define the system of equations for market clearing
    def market_clearing(self, prices):
        """ Check market clearing conditions for given prices """
        p1, p2 = prices
        ell_star, c1_star, c2_star = self.utility_maximization(p1, p2, self.par.w, self.par.A, self.par.gamma, self.par.alpha, self.par.nu, self.par.epsilon, self.par.tau, self.par.T)
        ell1_star = self.optimal_labor(self.par.w, p1, self.par.A, self.par.gamma)
        ell2_star = self.optimal_labor(self.par.w, p2, self.par.A, self.par.gamma)
        y1_star = self.optimal_output(self.par.A, ell1_star, self.par.gamma)
        y2_star = self.optimal_output(self.par.A, ell2_star, self.par.gamma)

        labor_market = ell_star - (ell1_star + ell2_star)
        goods_market1 = c1_star - y1_star
        # We do not need to check goods market 2 because of Walras' law

        return [labor_market, goods_market1]

    def check_market_clearing_conditions(self):
        """ Check market clearing conditions for a grid of p1 and p2 values """
        p1_values = np.linspace(0.1, 2.0, 10)
        p2_values = np.linspace(0.1, 2.0, 10)
        results = []

        for p1 in p1_values:
            for p2 in p2_values:
                labor_market, goods_market1 = self.market_clearing([p1, p2])
                results.append((p1, p2, labor_market, goods_market1))

        return results
    
    def find_equilibrium_prices(self):
        """ Find equilibrium prices that clear the markets """
        initial_guess = [1.0, 1.0]
        equilibrium_prices = fsolve(self.market_clearing, initial_guess)
        return equilibrium_prices

    def check_equilibrium_conditions(self, equilibrium_prices):
        """ Check if the equilibrium conditions hold for given prices """
        p1_star, p2_star = equilibrium_prices
        ell_star, c1_star, c2_star = self.utility_maximization(p1_star, p2_star, self.par.w, self.par.A, self.par.gamma, self.par.alpha, self.par.nu, self.par.epsilon, self.par.tau, self.par.T)
        ell1_star = self.optimal_labor(self.par.w, p1_star, self.par.A, self.par.gamma)
        ell2_star = self.optimal_labor(self.par.w, p2_star, self.par.A, self.par.gamma)
        y1_star = self.optimal_output(self.par.A, ell1_star, self.par.gamma)
        y2_star = self.optimal_output(self.par.A, ell2_star, self.par.gamma)

        labor_market_clearing = np.isclose(ell_star, ell1_star + ell2_star)
        goods_market1_clearing = np.isclose(c1_star, y1_star)
        goods_market2_clearing = np.isclose(c2_star, y2_star)

        return labor_market_clearing, goods_market1_clearing, goods_market2_clearing

    # Define the social welfare function
    def social_welfare(self, tau, p1, p2, w, A, gamma, alpha, nu, epsilon, kappa):
        """ Calculate the social welfare given a tax rate tau """
        T = tau * self.consumption2(w, 1, 0, self.optimal_profits(w, p1, A, gamma), self.optimal_profits(w, p2, A, gamma), p2, tau, alpha)  # Initial T=0
        ell_star, c1_star, c2_star = self.utility_maximization(p1, p2, w, A, gamma, alpha, nu, epsilon, tau, T)
        y2_star = self.optimal_output(A, self.optimal_labor(w, p2, A, gamma), gamma)
        utility_value = np.log(c1_star ** alpha * c2_star ** (1 - alpha)) - nu * ell_star ** (1 + epsilon) / (1 + epsilon)
        swf = utility_value - kappa * y2_star
        return -swf  # Negative for minimization

    def find_optimal_tau(self, equilibrium_prices):
        """ Find the optimal tax rate tau that maximizes social welfare """
        p1_star, p2_star = equilibrium_prices
        initial_tau = 0.1
        result = minimize(self.social_welfare, initial_tau, args=(p1_star, p2_star, self.par.w, self.par.A, self.par.gamma, self.par.alpha, self.par.nu, self.par.epsilon, self.par.kappa), bounds=[(0, 1)])
        optimal_tau = result.x[0]
        optimal_T = optimal_tau * self.consumption2(self.par.w, 1, 0, self.optimal_profits(self.par.w, p1_star, self.par.A, self.par.gamma), self.optimal_profits(self.par.w, p2_star, self.par.A, self.par.gamma), p2_star, optimal_tau, self.par.alpha)
        return optimal_tau, optimal_T
    

class Problem2:
    def __init__(self, J, N, K, v, sigma, F, c):
        """
        Initialize the Problem2 instance.

        Parameters:
        J (int): Number of career tracks.
        N (int): Number of graduates.
        K (int): Number of simulations to run.
        v (array-like): Array of known utility values for each career track.
        sigma (float): Standard deviation of the normally distributed random variable.
        F (array-like): Array of number of friends for each graduate.
        c (float): Switching cost for changing careers after the first year.
        """
        self.J = J
        self.N = N
        self.K = K
        self.v = np.array(v)
        self.sigma = sigma
        self.F = np.array(F)
        self.c = c

    def simulate(self):
        """
        Simulate the expected and average realized utility for each career track.

        Returns:
        expected_utility (numpy array): The expected utility for each career track.
        average_realized_utility (numpy array): The average realized utility for each career track.
        """
        # Simulate epsilon, which is a normally distributed random variable
        epsilon = np.random.normal(0, self.sigma, (self.K, self.J))
        
        # Calculate the expected utility by adding the mean of epsilon to v
        expected_utility = self.v + np.mean(epsilon, axis=0)
        
        # Calculate the realized utility for each simulation
        realized_utility = self.v + epsilon
        
        # Calculate the average realized utility over all simulations
        average_realized_utility = np.mean(realized_utility, axis=0)
        
        return expected_utility, average_realized_utility

    def simulate_alternative_scenario(self):
        """
        Simulate the scenario where graduates base their decisions on friends' information.

        Returns:
        choices (numpy array): The chosen career for each graduate in each simulation.
        expected_utilities (numpy array): The prior expected utility for each chosen career.
        realized_utilities (numpy array): The realized utility for each chosen career.
        """
        choices = np.zeros((self.N, self.K), dtype=int)
        expected_utilities = np.zeros((self.N, self.K))
        realized_utilities = np.zeros((self.N, self.K))

        for k in range(self.K):
            for i in range(self.N):
                Fi = self.F[i]
                friend_epsilons = np.random.normal(0, self.sigma, (Fi, self.J))
                prior_expected_utilities = self.v + np.mean(friend_epsilons, axis=0)

                own_epsilons = np.random.normal(0, self.sigma, self.J)
                realized_utilities_i = self.v + own_epsilons

                chosen_career = np.argmax(prior_expected_utilities)
                choices[i, k] = chosen_career
                expected_utilities[i, k] = prior_expected_utilities[chosen_career]
                realized_utilities[i, k] = realized_utilities_i[chosen_career]

        return choices, expected_utilities, realized_utilities

    def analyze_results(self, choices, expected_utilities, realized_utilities):
        """
        Analyze the results of the alternative scenario simulation.

        Returns:
        career_shares (numpy array): The share of graduates choosing each career.
        average_expected_utilities (numpy array): The average subjective expected utility for each graduate.
        average_realized_utilities (numpy array): The average realized utility for each graduate.
        """
        career_shares = np.zeros((self.N, self.J))
        average_expected_utilities = np.zeros(self.N)
        average_realized_utilities = np.zeros(self.N)

        for i in range(self.N):
            for j in range(self.J):
                career_shares[i, j] = np.mean(choices[i, :] == j)
            average_expected_utilities[i] = np.mean(expected_utilities[i, :])
            average_realized_utilities[i] = np.mean(realized_utilities[i, :])

        return career_shares, average_expected_utilities, average_realized_utilities

    def simulate_with_switching(self):
        """
        Simulate the scenario where graduates can switch careers after one year.

        Returns:
        initial_choices (numpy array): Initial career choices for each graduate in each simulation.
        switch_choices (numpy array): New career choices after switching.
        switch_expected_utilities (numpy array): Expected utility for the new career choices.
        switch_realized_utilities (numpy array): Realized utility for the new career choices.
        switched (numpy array): Indicator if the graduate switched careers.
        """
        # Simulate initial choices and utilities
        initial_choices, initial_expected_utilities, initial_realized_utilities = self.simulate_alternative_scenario()
        
        # Initialize arrays for storing results after switching
        switch_choices = np.zeros((self.N, self.K), dtype=int)
        switch_expected_utilities = np.zeros((self.N, self.K))
        switch_realized_utilities = np.zeros((self.N, self.K))
        switched = np.zeros((self.N, self.K), dtype=bool)
        
        for k in range(self.K):
            for i in range(self.N):
                initial_choice = initial_choices[i, k]
                initial_utility = initial_realized_utilities[i, k]
                
                Fi = self.F[i]
                friend_epsilons = np.random.normal(0, self.sigma, (Fi, self.J))
                prior_expected_utilities = self.v + np.mean(friend_epsilons, axis=0)
                
                # Adjust expected utilities based on switching cost
                for j in range(self.J):
                    if j != initial_choice:
                        prior_expected_utilities[j] -= self.c
                    else:
                        prior_expected_utilities[j] = initial_utility

                # Choose new career based on adjusted expected utilities
                new_choice = np.argmax(prior_expected_utilities)
                switch_choices[i, k] = new_choice
                switch_expected_utilities[i, k] = prior_expected_utilities[new_choice]

                own_epsilons = np.random.normal(0, self.sigma, self.J)
                new_realized_utilities = self.v + own_epsilons
                if new_choice != initial_choice:
                    switch_realized_utilities[i, k] = new_realized_utilities[new_choice] - self.c
                    switched[i, k] = True
                else:
                    switch_realized_utilities[i, k] = initial_utility

        return initial_choices, switch_choices, switch_expected_utilities, switch_realized_utilities, switched

    def analyze_switching_results(self, initial_choices, switch_choices, switch_expected_utilities, switch_realized_utilities, switched):
        """
        Analyze the results of the switching scenario simulation.

        Returns:
        career_shares_after_switching (numpy array): The share of graduates choosing each career after switching.
        average_switch_expected_utilities (numpy array): The average subjective expected utility for each graduate after switching.
        average_switch_realized_utilities (numpy array): The average realized utility for each graduate after switching.
        switch_rate (numpy array): The rate of graduates switching careers based on initial career choice.
        """
        career_shares_after_switching = np.zeros((self.N, self.J))
        average_switch_expected_utilities = np.zeros(self.N)
        average_switch_realized_utilities = np.zeros(self.N)
        switch_rate = np.zeros((self.N, self.J))

        for i in range(self.N):
            for j in range(self.J):
                career_shares_after_switching[i, j] = np.mean(switch_choices[i, :] == j)
                switch_rate[i, j] = np.mean(switched[i, :] & (initial_choices[i, :] == j))
            average_switch_expected_utilities[i] = np.mean(switch_expected_utilities[i, :])
            average_switch_realized_utilities[i] = np.mean(switch_realized_utilities[i, :])

        return career_shares_after_switching, average_switch_expected_utilities, average_switch_realized_utilities, switch_rate


class Problem3:
    def __init__(self, rng_seed=2024):
        """
        Initialize Problem3.
        
        Parameters:
        rng_seed (int): Seed for the random number generator.
        """
        self.rng = np.random.default_rng(rng_seed)
        self.X = self.rng.uniform(size=(50, 2))
        self.y = self.rng.uniform(size=(2,))
    
    @staticmethod
    def f(x1, x2):
        """
        Dummy function to evaluate at given points.

        Parameters:
        x1, x2 (float): Coordinates of the point.

        Returns:
        float: Function value at the given point.
        """
        return np.sin(np.pi * x1) * np.cos(np.pi * x2)

    def calculate_function_values(self):
        """
        Calculate the function values for the random points in the unit square.
        """
        self.F = np.array([self.f(x1, x2) for x1, x2 in self.X])

    @staticmethod
    def find_closest_point(X, y, condition):
        """
        Find the closest point in X that satisfies the given condition.

        Parameters:
        X (numpy array): Array of points.
        y (numpy array): Target point.
        condition (function): Condition to satisfy.

        Returns:
        numpy array: Closest point that satisfies the condition.
        """
        filtered_points = np.array([point for point in X if condition(point)])
        if len(filtered_points) == 0:
            return np.array([np.nan, np.nan])
        closest_point = filtered_points[np.argmin(np.linalg.norm(filtered_points - y, axis=1))]
        return closest_point

    def find_points(self):
        """
        Find points A, B, C, and D based on their conditions relative to y.
        """
        conditions = [
            lambda p: p[0] > self.y[0] and p[1] > self.y[1],  # A
            lambda p: p[0] > self.y[0] and p[1] < self.y[1],  # B
            lambda p: p[0] < self.y[0] and p[1] < self.y[1],  # C
            lambda p: p[0] < self.y[0] and p[1] > self.y[1]   # D
        ]
        self.A, self.B, self.C, self.D = [self.find_closest_point(self.X, self.y, condition) for condition in conditions]

    @staticmethod
    def barycentric_coords(A, B, C, y):
        """
        Compute barycentric coordinates for point y with respect to triangle ABC.

        Parameters:
        A, B, C (numpy array): Vertices of the triangle.
        y (numpy array): Target point.

        Returns:
        tuple: Barycentric coordinates (r1, r2, r3).
        """
        denom = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        r1 = ((B[1] - C[1]) * (y[0] - C[0]) + (C[0] - B[0]) * (y[1] - C[1])) / denom
        r2 = ((C[1] - A[1]) * (y[0] - C[0]) + (A[0] - C[0]) * (y[1] - C[1])) / denom
        r3 = 1 - r1 - r2
        return r1, r2, r3

    def approximate_function(self):
        """
        Approximate the function value at y using barycentric coordinates.

        Returns:
        float: Approximated function value at y.
        """
        if np.isnan(self.A).any() or np.isnan(self.B).any() or np.isnan(self.C).any() or np.isnan(self.D).any():
            return np.nan
        
        r1_ABC, r2_ABC, r3_ABC = self.barycentric_coords(self.A, self.B, self.C, self.y)
        if 0 <= r1_ABC <= 1 and 0 <= r2_ABC <= 1 and 0 <= r3_ABC <= 1:
            return r1_ABC * self.f(*self.A) + r2_ABC * self.f(*self.B) + r3_ABC * self.f(*self.C)
        
        r1_CDA, r2_CDA, r3_CDA = self.barycentric_coords(self.C, self.D, self.A, self.y)
        if 0 <= r1_CDA <= 1 and 0 <= r2_CDA <= 1 and 0 <= r3_CDA <= 1:
            return r1_CDA * self.f(*self.C) + r2_CDA * self.f(*self.D) + r3_CDA * self.f(*self.A)
        
        return np.nan

    def plot_points_and_triangles(self):
        """
        Plot the points, target point y, and triangles ABC and CDA with smaller points and feminine colors.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(self.X[:, 0], self.X[:, 1], label='Random Points', color='lightpink', s=10)
        plt.scatter(self.y[0], self.y[1], color='red', label='y', s=50)
        plt.scatter([self.A[0], self.B[0], self.C[0], self.D[0]], [self.A[1], self.B[1], self.C[1], self.D[1]], color='mediumpurple', label='A, B, C, D', s=50)

        triangle_ABC = plt.Polygon([self.A, self.B, self.C], fill=None, edgecolor='darkviolet', linestyle='--', linewidth=1.5, label='Triangle ABC')
        triangle_CDA = plt.Polygon([self.C, self.D, self.A], fill=None, edgecolor='deeppink', linestyle='--', linewidth=1.5, label='Triangle CDA')
        plt.gca().add_patch(triangle_ABC)
        plt.gca().add_patch(triangle_CDA)

        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('Points and Triangles')
        plt.grid(True)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        plt.show()

    def approximate_function_new(self, f):
        """
        Approximate the function value at y using barycentric coordinates for a new function.

        Parameters:
        f (function): The new function to evaluate.

        Returns:
        float: Approximated function value at y.
        """
        if np.isnan(self.A).any() or np.isnan(self.B).any() or np.isnan(self.C).any() or np.isnan(self.D).any():
            return np.nan
        
        r1_ABC, r2_ABC, r3_ABC = self.barycentric_coords(self.A, self.B, self.C, self.y)
        if 0 <= r1_ABC <= 1 and 0 <= r2_ABC <= 1 and 0 <= r3_ABC <= 1:
            return r1_ABC * f(self.A) + r2_ABC * f(self.B) + r3_ABC * f(self.C)
        
        r1_CDA, r2_CDA, r3_CDA = self.barycentric_coords(self.C, self.D, self.A, self.y)
        if 0 <= r1_CDA <= 1 and 0 <= r2_CDA <= 1 and 0 <= r3_CDA <= 1:
            return r1_CDA * f(self.C) + r2_CDA * f(self.D) + r3_CDA * f(self.A)
        
        return np.nan
