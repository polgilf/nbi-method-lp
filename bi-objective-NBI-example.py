import pulp
import numpy as np
from src.MOLP import MOLP, Solution
from src.NBI import NBI, plot_NBI_2D, plot_NBI_3D
#----------------------------------------------
# Define the problem
#----------------------------------------------
prob = pulp.LpProblem("DummyBiObjectiveLP",pulp.LpMinimize)

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

objectives = [f1,f2]
variables = [x1, x2]

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
print(ref_points)
print('\nSolutions: \n')
print(nbi.solutions_ref_to_values())
print('\nVariable values: \n')
print(nbi.solutions_variable_values())
print('\nVariable dict: \n')
for solution in solutions:
    print(solutions[solution].variable_dict())

print(' ')

plot_NBI_2D(nbi)
