import os
import sys

import pulp
import numpy as np

# Adding src directory to module search path (sys.path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Importing from the src directory
from MOLP import MOLP, Solution
from nNBI import nNBI, plot_NBI_2D

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
prob += f2 == 1*x1 + 4*x2

# Constraints
prob += 4*x1 + 1*x2 >= 6
prob += 1*x1 + 1*x2 >= 4
prob += 1*x1 + 8*x2 >= 2
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
# Create the NBI object (inherits from MOLP and adds the NBI algorithm)
nnbi = nNBI(prob, objectives, variables)

nnbi.compute_all_individual_optima()

nnbi.normalize_objectives_and_individual_optima()

nnbi.normalized_NBI_algorithm(num_ref_points)

nnbi.denormalize_solutions()

print('\n Solutions dict where key is ref_id  (method nbi.solutions_ref_to_values()): \n')
print(nnbi.solutions_ref_to_values())

plot_NBI_2D(nnbi, normalize_scale=False)
plot_NBI_2D(nnbi, normalize_scale=True)