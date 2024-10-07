import pulp
import numpy as np
from copy import deepcopy, copy

class MOLP:
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

def distribute_line_points(A, B, K):
    A, B = np.array(A), np.array(B)
    t = np.linspace(0, 1, K)
    return np.array([(1 - ti) * A + ti * B for ti in t])

def distribute_triangle_points(A, B, C, K):
    A, B, C = np.array(A), np.array(B), np.array(C)
    points = []
    for i in range(K):
        for j in range(K - i):
            a = i / (K - 1)
            b = j / (K - 1)
            c = 1 - a - b
            point = a * A + b * B + c * C
            points.append(point)
    return np.array(points)
    
class NormalBoundaryIntersection:
    def __init__(self, MOLPproblem, num_reference_points=10):
        self.MOLPproblem = MOLPproblem # MOLP object
        self.prob = MOLPproblem.prob # pulp.LpProblem object
        self.num_objectives = MOLPproblem.num_objectives
        self.num_variables = MOLPproblem.num_variables
        self.num_reference_points = num_reference_points

        # Attributes from the problem, compute if not already computed
        # ... individual optima
        if MOLPproblem.individual_optima is not None:
            self.individual_optima = MOLPproblem.individual_optima
        else:
            self.individual_optima = MOLPproblem.compute_individual_optima()
        # ... payoff table
        if MOLPproblem.payoff_table is not None:
            self.payoff_table = MOLPproblem.payoff_table
        else: 
            self.payoff_table = MOLPproblem.compute_payoff_table()
        # ... ideal point
        if MOLPproblem.ideal_point is not None:    
            self.ideal_point = MOLPproblem.ideal_point
        else:
            self.ideal_point = MOLPproblem.compute_ideal_point()
        # ... nadir point
        if MOLPproblem.nadir_point is not None:
            self.nadir_point = MOLPproblem.nadir_point
        else:
            self.nadir_point = MOLPproblem.compute_nadir_point()

        # Attributes to be filled during the NBI method (using the class methods)
        self.reference_points = None
        self.normal_vector = None # Normal vector to the hyperplane defined by the individual optima
        self.representative_pareto_front = None # List of solutions
        self.reference_point_to_solution = None # Dictionary that maps reference points to solutions

    def compute_reference_points(self):
        # For bi-objective problems, the reference points are distributed along the line connecting the individual optima
        if self.num_objectives == 2:
            A = self.individual_optima[0].objective_values
            B = self.individual_optima[1].objective_values
            # Both inplace and return:
            self.reference_points = distribute_line_points(A, B, self.num_reference_points)
            return self.reference_points
        
        # For tri-objective problems, the reference points are distributed in the triangle formed by the individual optima
        elif self.num_objectives == 3:
            A = self.individual_optima[0].objective_values
            B = self.individual_optima[1].objective_values
            C = self.individual_optima[2].objective_values
            # Both inplace and return:
            self.reference_points = distribute_triangle_points(A, B, C, self.num_reference_points)
            return self.reference_points
        
    def compute_normal_vector(self):
        # Compute the normal vector to the hyperplane defined by the individual optima
        # This vector should point to the half-space where the ideal point is located
        if self.num_objectives == 2:
            A = self.individual_optima[0].objective_values
            B = self.individual_optima[1].objective_values
            direction_vector = B - A
            normal_vector_1 = np.array([-direction_vector[1], direction_vector[0]]) 
            normal_vector_2 = -normal_vector_1
            # Determine which normal vector points to the half-space where the ideal point is located (TO BE CHECKED)
            if np.dot(normal_vector_1, self.ideal_point - A) > 0:
                self.normal_vector = normal_vector_1 / np.linalg.norm(normal_vector_1)
            else:
                self.normal_vector = normal_vector_2 / np.linalg.norm(normal_vector_1)
        return self.normal_vector
    
    def solve_NBI_subprolbem(self, reference_point):
        # Solve the NBI subproblem for a given reference point
        # Create a new problem to define the max t NBI subproblem
        nbi_subproblem = deepcopy(self.prob)
        normal_vector = self.compute_normal_vector()
        # Set new variable t, objective function max t, and constraints reference_point + t * normal_vector == f
        t = pulp.LpVariable("t",0,None)
        #nbi_subproblem.setObjective(t)
        for i in range(self.num_objectives):
            nbi_subproblem += reference_point[i] + t * normal_vector[i] == self.MOLPproblem.objective_functions[i]
        nbi_subproblem.solve(pulp.PULP_CBC_CMD(msg=0))
        #print('Status:', [v.varValue for v in nbi_subproblem.variables() if v.name in objective_function_names])
        # Return the solution
        nbi_solution = Solution(nbi_subproblem, objective_names=self.MOLPproblem.objective_functions_names)
        #print('SubNBI: ', nbi_solution.objective_dict)

        return nbi_solution

    def compute_representative_pareto_front(self):
        # If not already computed, compute reference points
        if self.reference_points is None:
            self.reference_points = self.compute_reference_points()

        # Compute the representative Pareto front
        self.representative_pareto_front = []
        self.reference_point_to_solution = {}

        # Solve NBI subproblems for each reference point
        for reference_point in self.reference_points:
            nbi_solution = self.solve_NBI_subprolbem(reference_point)
            # Add the solution to the representative Pareto front
            self.representative_pareto_front.append(nbi_solution)
            self.reference_point_to_solution[tuple(reference_point)] = nbi_solution
        return self.representative_pareto_front