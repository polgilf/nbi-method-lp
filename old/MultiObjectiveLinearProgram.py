import pulp
import numpy as np
from copy import deepcopy, copy
"""
This script defines the classes MultiObjectiveLinearProgram() and Solution() to handle multi-objective linear programs.
"""
class Solution():
    def __init__(self,prob=None, objective_names=None):
        self.id = None # Unique identifier
        self.objective_names = objective_names # List of objective names (strings, same used to define Pulp)

        self.status = None
        self.execution_time = None
        self.objective_dict = None # Dictionary with objective values
        self.variable_dict = None # Dictionary with variable values


        if self.prob is not None and objetive_names is not None:
            self.prob = prob
            self.prob.solve(pulp.PULP_CBC_CMD(msg=0))
            self.status = pulp.LpStatus[self.prob.status] # Status of the optimization process
            self.execution_time = self.prob.solutionTime # Execution time of the optimization process

            self.objective_dict = 


            self.objective_dict = {v.name: v.varValue for v in self.prob.variables() if v.name in self.objective_names}
            self.variable_dict = {v.name: v.varValue for v in self.prob.variables() if not v.name in self.objective_names}

    def objective_values(self):
        if self.objective_dict is None:
            return None
        return np.array(list(self.objective_dict.values()))
    
    def variable_values(self):
        if self.variable_dict is None:
            return None
        return np.array(list(self.variable_dict.values()))
    
    def variable_names(self):
        if self.variable_dict is None:
            return None
        return list(self.variable_dict.keys())

    def remove_variable(self, variable_to_remove):
        self.variable_dict = {k: v for k, v in self.variable_dict.items() if k not in variable_to_remove}

    def print(self):
        '''
        Solution
        self.objective_dict
        self.variable_dict
        '''
        print('Solution:')
        print('Objective values:')
        print(self.objective_dict)
        print('Variable values:')
        print(self.variable_dict)
        print('Problem:')
        print(self.prob)
        

class MultiObjectiveLinearProgram():
    def __init__(self,prob,objective_functions, objective_names):
        self.prob = prob
        self.objective_functions = objective_functions
        self.objective_names = objective_names

        # Attributes to be filled during the optimization process (using the class methods)
        self.individual_optima = None # List of individual optima (solution object)
        self.payoff_table = None # Payoff table (num_objectives x num_objectives numpy array)
        self.ideal_point = None # Ideal point (numpy array)
        self.nadir_point = None # Nadir point (numpy array)

    def num_objectives(self):
        return len(self.objective_functions)
    
    def num_variables(self):
        return len(self.prob.variables())
    
    def print(self):
        print('Multi Objective Linear Program:')
        print('Objective functions:')
        print(self.objective_functions)
        print('PULP model:')
        print(self.prob)

    def compute_individual_optima(self):
        self.individual_optima = []
        for i, f in enumerate(self.objective_functions):
            sub_problem = deepcopy(self.prob)
            # Create the aggregated objective function
            aggregated_objective = f + 0.000001 * sum(self.objective_functions[j] for j in range(self.num_objectives()) if j != i)
            sub_problem.setObjective(aggregated_objective)           
            sub_problem.solve(pulp.PULP_CBC_CMD(msg=0))
            solution = Solution(sub_problem, self.objective_names)
            self.individual_optima.append(solution)
        return self.individual_optima

    def compute_payoff_table(self):
        # Compute individual optima if not already computed
        if self.individual_optima is None:
            self.compute_individual_optima()
        # Compute payoff table
        self.payoff_table = np.zeros((self.num_objectives(), self.num_objectives()))
        for i in range(self.num_objectives()):
            for j in range(self.num_objectives()):
                self.payoff_table[i,j] = self.individual_optima[i].objective_values()[j]
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
    
    def individual_optima_objective_array(self):
        if self.individual_optima is None:
            self.compute_individual_optima()
        return np.array([sol.objective_values() for sol in self.individual_optima])
    
    def individual_optima_variable_array(self):
        if self.individual_optima is None:
            self.compute_individual_optima()
        return np.array([sol.variable_values() for sol in self.individual_optima])