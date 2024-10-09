import pulp
import numpy as np
from src.MOLP import MOLP, Solution
from src.NBI import NBI, plot_NBI_2D, plot_NBI_3D
#----------------------------------------------
# Define the problem
#----------------------------------------------
prob = pulp.LpProblem("DummyTriObjectiveLP",pulp.LpMinimize)

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
objectives = [f1, f2, f3]
variables = [x1, x2, x3]

#----------------------------------------------
# Parameters
#----------------------------------------------
num_ref_points = 20
#----------------------------------------------
# Run the NBI algorithm
#----------------------------------------------
# Create the MOLP object
molp = MOLP(prob, objectives, variables)
# Compute the individual optima for all objectives
sol = molp.compute_all_individual_optima()
# Object can call to these methods now
print('Pay-off matrix:', list(molp.payoff_matrix()))
print('Ideal point: ', molp.ideal_point())
print('Nadir point: ', molp.nadir_point())

# Create the NBI object (inherits from MOLP and adds the NBI algorithm)
nbi = NBI(prob, objectives, variables)
# Compute NBI algorithm
nbi.NBI_algorithm(num_ref_points)

solutions = nbi.solutions_dict
ref_points = nbi.ref_points_dict
variable_values = nbi.solutions_variable_values()
print('\nRESULTS  \n')
print('Reference points: \n')
print(nbi.ref_points_values())
print('\nSolutions: \n')
print(nbi.solutions_values())
print('\nVariable values: \n')
print(nbi.solutions_variable_values())
print('\nVariable dict: \n')
for solution in solutions:
    print(solutions[solution].variable_dict())

plot_NBI_3D(nbi)