import pulp

#----------------------------------------------
# Simplest bi-objective problem (original for development)
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
# Simplest tri-objective problem (original for development)
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
