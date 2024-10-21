import os
import sys
import pulp
import numpy as np

# Adding src directory to module search path (sys.path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Importing from the src directory
from MOLP import MOLP, Solution
from nNBI import nNBI , plot_NBI_3D, plot_NBI_3D_to_2D


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
# Create the NBI object (inherits from MOLP and adds the NBI algorithm)
nnbi = nNBI(prob, objectives, variables)

nnbi.compute_all_individual_optima()

nnbi.normalize_objectives_and_individual_optima()

nnbi.normalized_NBI_algorithm(num_ref_points)

nnbi.denormalize_solutions()

print('\n Solutions dict where key is ref_id  (method nbi.solutions_ref_to_values()): \n')
print(nnbi.solutions_ref_to_values())

plot_NBI_3D(nnbi, normalize_scale=False)
plot_NBI_3D(nnbi, normalize_scale=True)
plot_NBI_3D_to_2D(nnbi, objectives_to_use=[1,1,0], swap_axes=False, normalize_scale=True)
plot_NBI_3D_to_2D(nnbi, objectives_to_use=[0,1,1], swap_axes=False, normalize_scale=True)