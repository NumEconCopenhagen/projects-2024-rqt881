from types import SimpleNamespace
from scipy import optimize
import numpy as np  
from scipy.optimize import minimize


class ExchangeEconomyClass:
    
    def __init__(self):
        self.par = SimpleNamespace()

        # a. preferences
        self.par.alpha = 1/3
        self.par.beta = 2/3

        # b. endowments for A
        self.par.w1A = 0.8
        self.par.w2A = 0.3

        # c. calculating and setting endowments for B based on total endowment being 1
        self.par.w1B = 1 - self.par.w1A
        self.par.w2B = 1 - self.par.w2A
    
    def utility_A(self,x1A,x2A):
        """
        Returns the utility of agent A. Takes x1A and x2A as arguments.
        """
        par= self.par
        return  x1A**par.alpha*x2A**(1-par.alpha)

    def utility_B(self,x1B,x2B):
        """
        Returns the utility of agent B. Takes x1A and x2A as arguments.
        """
        par=self.par
        return x1B**par.beta*x2B**(1-par.beta)

    def demand_A(self,p1):
        """
        Returns the demand of agent A. Takes the prices of good 1 and good 2 respectively as arguments.
        """
        p2 = 1 #p2 is numeraire
        par = self.par
        x1A = par.alpha*(par.w1A*p1+par.w2A*p2)/p1
        x2A = (1-par.alpha)*(par.w1A*p1+par.w2A*p2)/p2
        return x1A, x2A

    def demand_B(self,p1):
        """
        Returns the demand of agent B. Takes the prices of good 1 and good 2 respectively as arguments.
        """
        p2 = 1 #p2 is numeraire
        par = self.par
        x1B = par.beta*(par.w1B*p1+par.w2B*p2)/p1
        x2B = (1-par.beta)*(par.w1B*p1+par.w2B*p2)/p2
        return x1B, x2B
    
    
    def pareto_improve(self, x1A, x2A):
        """
        Checks for each combination of (x1A, x2A) if it's a Pareto improvement.
        x1A and x2A should be NumPy arrays of the same length.
        """
        par = self.par
        pareto_improvements = []

        init_utilityA = self.utility_A(par.w1A, par.w2A)
        init_utilityB = self.utility_B(par.w1B, par.w2B)

        # Iterate over each combination of points in x1A and x2A
        for i in range(len(x1A)):
            for j in range(len(x2A)):
                c = x1A[i]
                d = x2A[j]
                if self.utility_A(c, d) > init_utilityA and self.utility_B(1-c, 1-d) > init_utilityB:
                    pareto_improvements.append((c, d))
        
        return pareto_improvements
  
    # Market clearing error
    def check_market_clearing(self,p1):
        """
        Calculates the market error on the market for good 1 and good 2. Takes p1 as argument.
        """

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2

    
    def find_market_equilibrium_price(self):
        """
        Optimizes agent A's allocation to maximize their utility while ensuring 
        agent B's utility is not less than with their initial endowment.
        """
        # Objective function to maximize A's utility (minimize the negative of it)
        obj = lambda x: -self.utility_A(x[0], x[1])
        
        # Constraint ensuring B's utility is at least as high as with their initial endowment
        constraints = {'type': 'ineq', 'fun': lambda x: self.utility_B(1 - x[0], 1 - x[1]) - self.utility_B(self.par.w1B, self.par.w2B)}
        
        # Bounds ensuring allocations for A are within [0,1]
        bounds = ((1e-8, 1), (1e-8, 1))
        
        # Initial guess (could be A's initial endowment for simplicity)
        x0 = [self.par.w1A, self.par.w2A]
        
        # Optimize using scipy's minimize method
        result = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            x1A_opt, x2A_opt = result.x
            print(f'Optimal allocation for A: x1 = {x1A_opt:.3f}, x2 = {x2A_opt:.3f}, Utility = {self.utility_A(x1A_opt, x2A_opt):.3f}')
        else:
            print("Optimization failed.")
            x1A_opt, x2A_opt = np.nan, np.nan
        
        return x1A_opt, x2A_opt

  

    def find_optimal_allocation(self):
        # Objective function: We want to maximize agent A's utility (thus, we minimize the negative of it)
        obj = lambda x: -self.utility_A(x[0], x[1])
        
        # Constraint: Agent B's utility with the allocation must be at least as high as with their initial endowment
        constraints = ({
            'type': 'ineq', 
            'fun': lambda x: self.utility_B(1 - x[0], 1 - x[1]) - self.utility_B(self.par.w1B, self.par.w2B)
        })
        
        # Bounds for xA1 and xA2, ensuring they are within [0,1]
        bounds = ((0, 1), (0, 1))
        
        # Initial guess for the optimization
        x0 = [0.5, 0.5]
        
        # Perform the optimization
        result = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            x1A_opt, x2A_opt = result.x
            print(f"Optimal allocation for A: x1A = {x1A_opt:.4f}, x2A = {x2A_opt:.4f}")
            print(f"Utility for A: {self.utility_A(x1A_opt, x2A_opt):.4f}")
            print(f"Utility for B: {self.utility_B(1 - x1A_opt, 1 - x2A_opt):.4f}")
            return x1A_opt, x2A_opt
        else:
            print("Optimization failed.")
            return None


    def solve_social_planner(self):
        '''
        Solves the social planner problem by maximizing the sum of utilities of agent A and B.

        Returns:
        - p1opt (float): Optimal price of good 1 for agent A.
        - utilityA_opt (float): Optimal utility of agent A.
        '''

        # par = self.par
        # sol = model.sol    
        
        # a. objective function (to minimize) 
        obj = lambda xA: -(self.utility_A(xA[0],xA[1]) + self.utility_B(1 - xA[0],1 - xA[1])) # minimize -> negative of utility
            
        # b. constraints and bounds
        # budget_constraint = lambda x: par.m-par.p1*x[0]-par.p2*x[1] # violated if negative
        # constraints = ({'type':'ineq','fun':budget_constraint})
        bounds = ((1e-8,1),(1e-8,1))
                
        # c. call solver
        x0 = [0.2,0.6]
        print(x0)
        result = optimize.minimize(obj,x0,method='SLSQP',bounds=bounds)
            
        # d. save
        x1Aopt, x2Aopt = result.x
        # utilityA = self.utility_A(x1Aopt,x2Aopt)
        return x1Aopt, x2Aopt
    
    
    def walras(self, p1, eps=1e-8, maxiter=500):
        """
        Returns the market clearing price based on the lowest market error. takes p1 as argument.
        """
        t = 0
        while True:

            # i. excess demand
            excess = self.check_market_clearing(p1)

            # ii: ensures that the break conditions hold, i.e. that the excess demand for good 1 is not smaller then epsilon
            # and that the number of iterations (t) isn't higher than 500 
            if np.abs(excess[0]) < eps or t >= maxiter:
                print(f'{t:3d}: p1 = {p1:12.8f} -> excess demand -> {excess[0]:14.8f}')
                break

            # iii. updates p1
            p1 += excess[0]

            # iv. return and a lot of formatting for printing
            if t < 5 or t % 25 == 0:
                print(f'{t:3d}: p1 = {p1:12.8f} -> excess demand -> {excess[0]:14.8f}')
            elif t == 5:
                print('   ...')

            # v. update t (interation counter)
            t += 1

        return p1