import pulp
import numpy as np
from copy import deepcopy, copy

from MultiObjectiveLinearProgram import MultiObjectiveLinearProgram, Solution
from NormalBoundaryIntersection import NormalBoundaryIntersection #, distribute_line_points, distribute_triangle_points

"""
This script is an example of how to use NBI method with a tri-objective linear program.
"""

###################################################################################
# DEFINE THE PROBLEM
###################################################################################

# Define the problem
prob = pulp.LpProblem("BasicLP",pulp.LpMinimize)

# Variables
x1 = pulp.LpVariable("x1",0,None)
x2 = pulp.LpVariable("x2",0,None)
x3 = pulp.LpVariable("x3",0,None)

# Objectives (defined as variables)
f1 = pulp.LpVariable("f1",None,None)
f2 = pulp.LpVariable("f2",None,None)
f3 = pulp.LpVariable("f3",None,None)

# Objective function (defined as constraints, to be minimized)
prob += f1 == x1 + 3*x2 + 3* x3
prob += f2 == 4*x1 + x2 + 4*x3
prob += f3 == 5*x1 + 5*x2 + x3 

# Constraints
prob += x1 + x2 + x3 >= 8 # P1
prob += x1 + x2 + 2*x3 >= 10 # P2
prob += x1 + 2*x2 + x3 >= 10 # P3
prob += 3*x1 + x2 + x3 >= 12 # P4
prob += x1 + x2 + x3 <= 15 # P5

# List of objective functions (does not belong to prob object)
objective_names = ["f1","f2","f3"] 
objective_functions = [f1,f2,f3]

k = 5 # Number of reference points

# 2 -> 3
# 3 -> 6
# 4 -> 10
# 5 -> 15
# 6 -> 21

# In general, the number of reference points is given by the formula k(k+1)/2
###################################################################################
###################################################################################
# Automatic instructions
###################################################################################

original_problem = MultiObjectiveLinearProgram(prob,objective_functions, objective_names)
individual_optima = original_problem.compute_individual_optima() # List of individual optima (solution object)

payoff_table = original_problem.compute_payoff_table() # Payoff table (num_objectives x num_objectives numpy array)
ideal_point = original_problem.compute_ideal_point() # Ideal point (numpy array)
nadir_point = original_problem.compute_nadir_point() # Nadir point (numpy array)

print("Objective functions: ", objective_functions)
print("Individual optima objective values: ", [sol.objective_values for sol in individual_optima])   
print("Individual optima decision values: ", [sol.decision_values for sol in individual_optima])
NBI_method = NormalBoundaryIntersection(original_problem, num_reference_points=k)
NBI_method.compute_normal_vector()
NBI_method.compute_reference_points()
NBI_method.compute_representative_pareto_front()
representative_sol_values = NBI_method.representative_pareto_front_values
for reference_name, reference_point in NBI_method.reference_points_dict.items():
    print('Iteration: ')
    print(reference_name)
    print('Reference point: ', reference_point)
    print('Solution point: ', NBI_method.reference_point_to_solution_dict[reference_name].objective_values)
#print("Reference points: ", NBI_method.reference_points)

NBI_method.plot_NBI_method()

