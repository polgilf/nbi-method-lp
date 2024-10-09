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
# Create the MOLP object
molp = MOLP(prob, objectives, variables)
# Compute the individual optima for all objectives
sol = molp.compute_all_individual_optima()
# Create the NBI object (inherits from MOLP and adds the NBI algorithm)
nbi = NBI(prob, objectives, variables)
# Compute NBI algorithm
nbi.NBI_algorithm(num_ref_points)

#----------------------------------------------
# Prints and plot
#----------------------------------------------
# Create the MOLP object
# Object can call to these methods now
print('\n############################################')
print('\n#---------------- RESULTS ----------------*  \n')
print('############################################\n')
print('\n------------------------------------------')
print('\n#---- MOLP class attributes and methods----#\n')
print('------------------------------------------\n')
print('\n# molp.individual_optima atribute is a  list  that contains solution objects\n')
print('\n ...Using solution.objective_dict() and solution.variable dict: \n')
print('\n Individual optima (objective_dict()): \n')
print([sol.objective_dict() for sol in molp.individual_optima])
print('\n Individual optima (variable_dict()): \n')
print([sol.variable_dict() for sol in molp.individual_optima])
print('\n ...Using solution.objective_values() and solution.variable_values(): \n')
print('\n Individual optima (objective_values): \n')
print(np.array([sol.objective_values() for sol in molp.individual_optima]))
print('\n Individual optima (variable_values): \n')
print(np.array([sol.variable_values() for sol in molp.individual_optima]),  '\n')
print('\n# Methods from molp object (accessible after computing compute_all_individual_optima()\n')
print('\n Pay-off matrix (molp.payoff_matrix()): \n')
print(list(molp.payoff_matrix()))
print('\n Ideal point: (molp.ideal_point()): \n')
print(molp.ideal_point())
print('\n Nadir point: (molp.nadir_point())\n')
print(molp.nadir_point())

print('\n------------------------------------------')
print('\n#---- NBI class attributes and methods----#\n')
print('------------------------------------------\n')
solutions = nbi.solutions_dict
ref_points = nbi.ref_points_dict
variable_values = nbi.solutions_variable_values()
print('\n Reference points dict where key is ref_id (attribute nbi.ref_points dict): \n')
print(nbi.ref_points_dict)
print('\n Solutions dict where key is ref_id  (method nbi.solutions_ref_to_values()): \n')
print(nbi.solutions_ref_to_values())
print('\n Variable dicts (for each solution) (method solution.variable_dict() for each solution in nbi.solutions_dict): \n')
for solution in solutions:
    print(solutions[solution].variable_dict())
print('\n Solution dicts (for each solution) (method solution.objective_dict() for each solution in nbi.solutions_dict): \n')
for solution in solutions:
    print(solutions[solution].objective_dict())
print('\n Solution values as an array (method nbi.solutions_ref_to_values()): \n')
print(nbi.solutions_values())
print('\n Variable values as an array (method nbi.solutions_variable_values()): \n')
print(nbi.solutions_variable_values())

print(' ')

plot_NBI_2D(nbi)
