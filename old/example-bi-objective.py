import pulp
import numpy as np
from copy import deepcopy, copy

from old.MultiObjectiveLinearProgram import MultiObjectiveLinearProgram, Solution
from old.NormalBoundaryIntersection import NormalBoundaryIntersection #, distribute_line_points, distribute_triangle_points

"""
This script is an example of how to use NBI method with a bi-objective linear program.
"""

###################################################################################
# DEFINE THE PROBLEM
###################################################################################

# Define the problem
prob = pulp.LpProblem("BasicLP",pulp.LpMinimize)

# Variables
x1 = pulp.LpVariable("x1",0,None)
x2 = pulp.LpVariable("x2",0,None)

# Objectives (defined as variables)
f1 = pulp.LpVariable("f1",None,None)
f2 = pulp.LpVariable("f2",None,None)

# Objective function (defined as constraints, to be minimized)
prob += f1 == 10*x1 + 1*x2
prob += f2 == 1*x1 + 10*x2

# Constraints
prob += 4*x1 + 1*x2 >= 8
prob += 1*x1 + 1*x2 >= 4
prob += 1*x1 + 8*x2 >= 8
prob += x1 + x2 <= 10

# List of objective functions (does not belong to prob object)
objective_names = ["f1","f2"] 
objective_functions = [f1,f2]

###################################################################################
# COMPUTE k representative solutions
k = 20

###################################################################################
# Automatic instructions
###################################################################################

original_problem = MultiObjectiveLinearProgram(prob,objective_functions, objective_names)
"""
individual_optima = original_problem.compute_individual_optima() # List of individual optima (solution object)
payoff_table = original_problem.compute_payoff_table() # Payoff table (num_objectives x num_objectives numpy array)
ideal_point = original_problem.compute_ideal_point() # Ideal point (numpy array)
nadir_point = original_problem.compute_nadir_point() # Nadir point (numpy array)
"""

NBI_method = NormalBoundaryIntersection(original_problem, num_reference_points=k)
NBI_method.compute_subset_pareto_front()
representative_set = NBI_method.subset_pareto_front
representative_sol_values = NBI_method.subset_pareto_front_values

#NBI_method.plot_solutions_in_objective_space()

print("Reference points: ", NBI_method.reference_points)
print("Representative set: ", NBI_method.reference_points_dict)
print("Representative set: ", NBI_method.dict_objective_name_to_value(self)
NBI_method.plot_NBI_method()
"""
print("Individual optima: ", [sol.objective_values for sol in individual_optima])
print("Payoff table: ", payoff_table)
print("Ideal point: ", ideal_point)
print("Nadir point: ", nadir_point)
print("Reference points: ", NBI_method.reference_points)
print("Representative set: ", representative_obj_values)

print(type(ideal_point))
print(type(payoff_table))
print(type(representative_obj_values))
"""