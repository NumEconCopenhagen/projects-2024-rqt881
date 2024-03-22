class ExchangeEconomyClass:
    
    # Utility function for A
    def utility_A(self, x1, x2, alpha):
        """Calculates the utility of consumer A given goods x1 and x2 and preference parameter alpha."""
        return (x1**alpha) * (x2**(1-alpha))

    # Utility function for B
    def utility_B(self, x1, x2, beta):
        """Calculates the utility of consumer B given goods x1 and x2 and preference parameter beta."""
        return (x1**beta) * (x2**(1-beta))

    # Generates Pareto improvements for consumers A and B.
    def pareto_improvements(self, omega_A1, omega_A2, alpha, beta, N=75):
        """Generates Pareto improvements for consumers A and B."""
        from numpy import linspace, array
        
        initial_utility_A = self.utility_A(omega_A1, omega_A2, alpha)
        initial_utility_B = self.utility_B(1-omega_A1, 1-omega_A2, beta)
        
        improvements = []
        
        for xA1 in linspace(0, 1, N+1):
            for xA2 in linspace(0, 1, N+1):
                xB1 = 1 - xA1
                xB2 = 1 - xA2
                if self.utility_A(xA1, xA2, alpha) >= initial_utility_A and self.utility_B(xB1, xB2, beta) >= initial_utility_B:
                    improvements.append((xA1, xA2))
                    
        return array(improvements)

    # Define demand functions for consumers A and B
    def demand_A(self, p1, p2, omega_A1, omega_A2, alpha):
        budget = p1*omega_A1 + p2*omega_A2
        xA1_star = alpha * budget / p1
        xA2_star = (1 - alpha) * budget / p2
        return xA1_star, xA2_star

    def demand_B(self, p1, p2, omega_B1, omega_B2, beta):
        budget = p1*omega_B1 + p2*omega_B2
        xB1_star = beta * budget / p1
        xB2_star = (1 - beta) * budget / p2
        return xB1_star, xB2_star

    # Function to calculate market clearing error
    def market_clearing_error(self, p1, p2, omega_A, omega_B, alpha, beta):
        xA1_star, xA2_star = self.demand_A(p1, p2, omega_A[0], omega_A[1], alpha)
        xB1_star, xB2_star = self.demand_B(p1, p2, omega_B[0], omega_B[1], beta)
        
        epsilon1 = xA1_star - omega_A[0] + xB1_star - omega_B[0]
        epsilon2 = xA2_star - omega_A[1] + xB2_star - omega_B[1]
        
        return epsilon1, epsilon2

# Define the objective function to maximize consumer A's utility
def objective_function(p1, omega_B1, omega_B2):
    # Calculate the optimal consumption choices of consumer B
    xB1_star = demand_B(p1, omega_B1, omega_B2)[0]
    xB2_star = demand_B(p1, omega_B1, omega_B2)[1]
    
    # Calculate consumer A's utility
    uA = utility_A(1 - xB1_star, 1 - xB2_star)
    
    return -uA  # Negative sign for maximization
