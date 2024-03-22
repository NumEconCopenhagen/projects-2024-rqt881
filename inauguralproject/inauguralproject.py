def utility_A(x1, x2, alpha=1/3):
    """Calculates the utility of consumer A given goods x1 and x2 and preference parameter alpha."""
    return (x1**alpha) * (x2**(1-alpha))

def utility_B(x1, x2, beta=2/3):
    """Calculates the utility of consumer B given goods x1 and x2 and preference parameter beta."""
    return (x1**beta) * (x2**(1-beta))

def pareto_improvements(omega_A1, omega_A2, alpha, beta, N=75):
    """Generates Pareto improvements for consumers A and B."""
    from numpy import linspace, array
    
    initial_utility_A = utility_A(omega_A1, omega_A2, alpha)
    initial_utility_B = utility_B(1-omega_A1, 1-omega_A2, beta)
    
    improvements = []
    
    for xA1 in linspace(0, 1, N+1):
        for xA2 in linspace(0, 1, N+1):
            xB1 = 1 - xA1
            xB2 = 1 - xA2
            if utility_A(xA1, xA2, alpha) >= initial_utility_A and utility_B(xB1, xB2, beta) >= initial_utility_B:
                improvements.append((xA1, xA2))
                
    return array(improvements)