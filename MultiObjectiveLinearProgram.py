import pulp
import numpy as np

"""
This script defines the classes MultiObjectiveLinearProgram() and Solution() to handle multi-objective linear programs.
"""
class MultiObjectiveLinearProgram():
    def __init__(self,prob,objective_functions, objective_functions_names):
        self.prob = prob
        self.objective_functions = objective_functions
        self.objective_functions_names = objective_functions_names
        self.num_objectives = len(objective_functions)
        self.num_variables = len(prob.variables())

        # Attributes to be filled during the optimization process (using the class methods)
        self.individual_optima = None # List of individual optima (solution object)
        self.payoff_table = None # Payoff table (num_objectives x num_objectives numpy array)
        self.ideal_point = None # Ideal point (numpy array)
        self.nadir_point = None # Nadir point (numpy array)

    def compute_individual_optima(self):
        self.individual_optima = []
        for f in self.objective_functions:
            self.prob.setObjective(f)
            self.prob.solve(pulp.PULP_CBC_CMD(msg=0))
            solution = Solution(self.prob, self.objective_functions_names)
            self.individual_optima.append(solution)
        return self.individual_optima

    def compute_payoff_table(self):
        # Compute individual optima if not already computed
        if self.individual_optima is None:
            self.compute_individual_optima()
        # Compute payoff table
        self.payoff_table = np.zeros((self.num_objectives, self.num_objectives))
        for i in range(self.num_objectives):
            for j in range(self.num_objectives):
                self.payoff_table[i, j] = self.individual_optima[i].objective_values[j]
        return self.payoff_table

    def compute_ideal_point(self):
        # Compute payoff table if not already computed
        if self.payoff_table is None:
            self.compute_payoff_table()
        # Compute ideal point
        self.ideal_point = np.min(self.payoff_table,axis=0)
        return self.ideal_point
    
    def compute_nadir_point(self):
        # Compute payoff table if not already computed
        if self.payoff_table is None:
            self.compute_payoff_table()
        # Compute nadir point
        self.nadir_point = np.max(self.payoff_table,axis=0)
        return self.nadir_point

    def normalize_objective_values(self,objective_values):
        # Compute ideal and nadir points if not already computed
        if self.ideal_point is None:
            self.compute_ideal_point()
        if self.nadir_point is None:
            self.compute_nadir_point()
        # Normalize objective values
        return (objective_values - self.ideal_point) / (self.nadir_point - self.ideal_point)

class Solution():
    def __init__(self,prob=None, objective_names=None):
        self.id = None # Unique identifier
        self.prob = prob
        self.objective_names = objective_names

        self.objective_values = None
        self.decision_values = None
        self.execution_time = None

        if prob is not None:
            self.execution_time = prob.solutionTime

        if (prob is not None) and (objective_names is not None):
            self.objective_values = np.array([v.varValue for v in self.prob.variables() if v.name in self.objective_names])
            self.objective_dict = {v.name: v.varValue for v in self.prob.variables() if v.name in self.objective_names}
            self.decision_values = np.array([v.varValue for v in self.prob.variables() if not v.name in self.objective_names])
            self.decision_dict = {v.name: v.varValue for v in self.prob.variables() if not v.name in self.objective_names}
        
    # Set an arbitrary point of the decision space (given by a dictionary)
    def set_decision_dict(self,arbitrary_decision_dict):
        self.decision_dict = arbitrary_decision_dict
        self.decision_values = np.array([v for k,v in arbitrary_decision_dict.items])

    # Set an arbitrary point of the objective space (given by a dictionary)
    def set_objective_dict(self,arbitrary_objective_dict):
        self.objective_dict = arbitrary_objective_dict
        self.objective_values = np.array([v for k,v in arbitrary_objective_dict.items])