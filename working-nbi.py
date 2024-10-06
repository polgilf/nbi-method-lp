import numpy as np
from pulp import *

class MultiObjectiveLP:
    def __init__(self, num_variables, num_objectives):
        self.num_variables = num_variables
        self.num_objectives = num_objectives
        self.variables = LpVariable.dicts("x", range(num_variables))

    def objective_function(self, index):
        """
        Should return the objective function for the given index.
        To be implemented by the user.
        """
        raise NotImplementedError("Objective function must be implemented")

    def constraints(self):
        """
        Should return a list of constraints.
        To be implemented by the user.
        """
        raise NotImplementedError("Constraints must be implemented")

class NBISolver:
    def __init__(self, molp):
        self.molp = molp

    def solve_single_objective(self, objective_index):
        prob = LpProblem(f"Single Objective Optimization {objective_index}", LpMaximize)
        
        # Set objective
        prob += self.molp.objective_function(objective_index)
        
        # Add constraints
        for constraint in self.molp.constraints():
            prob += constraint

        # Solve the problem
        prob.solve()
        
        return [value(v) for v in self.molp.variables.values()]

    def get_payoff_matrix(self):
        payoff_matrix = np.zeros((self.molp.num_objectives, self.molp.num_objectives))
        
        for i in range(self.molp.num_objectives):
            solution = self.solve_single_objective(i)
            for j in range(self.molp.num_objectives):
                payoff_matrix[i, j] = value(self.molp.objective_function(j))
        
        return payoff_matrix

    def solve(self, num_points):
        payoff_matrix = self.get_payoff_matrix()
        utopia_point = np.max(payoff_matrix, axis=0)
        nadir_point = np.min(payoff_matrix, axis=0)

        weight_vectors = self.generate_weight_vectors(num_points)
        results = []

        for weights in weight_vectors:
            prob = LpProblem("NBI Subproblem", LpMaximize)
            
            t = LpVariable("t")
            prob += t

            # NBI constraint
            for i in range(self.molp.num_objectives):
                normalized_obj = self.normalize_objective(self.molp.objective_function(i), payoff_matrix)
                prob += normalized_obj - t * weights[i] >= 0

            # Original constraints
            for constraint in self.molp.constraints():
                prob += constraint

            # Solve the problem
            prob.solve()

            # Collect results
            solution = [value(v) for v in self.molp.variables.values()]
            objective_values = [value(self.molp.objective_function(i)) for i in range(self.molp.num_objectives)]
            results.append((solution, objective_values))

        return results

    @staticmethod
    def normalize_objective(obj, payoff_matrix):
        f_max = np.max(payoff_matrix, axis=0)
        f_min = np.min(payoff_matrix, axis=0)
        return (obj - f_min) / (f_max - f_min)

    @staticmethod
    def generate_weight_vectors(num_objectives, num_points):
        weights = []
        step = 1.0 / (num_points - 1)
        
        def generate_weights_recursive(current_weights, remaining_sum, index):
            if index == num_objectives - 1:
                current_weights.append(remaining_sum)
                weights.append(current_weights.copy())
                current_weights.pop()
                return

            for w in np.arange(0, remaining_sum + step, step):
                current_weights.append(w)
                generate_weights_recursive(current_weights, remaining_sum - w, index + 1)
                current_weights.pop()

        generate_weights_recursive([], 1.0, 0)
        return weights

# Example usage
class MyMOLP(MultiObjectiveLP):
    def __init__(self):
        super().__init__(num_variables=2, num_objectives=2)

    def objective_function(self, index):
        if index == 0:
            return 2 * self.variables[0] + self.variables[1]
        elif index == 1:
            return self.variables[0] + 3 * self.variables[1]

    def constraints(self):
        return [
            lpSum(self.variables.values()) <= 10,
            self.variables[0] + 2 * self.variables[1] <= 16
        ]

if __name__ == "__main__":
    molp = MyMOLP()
    solver = NBISolver(molp)
    results = solver.solve(num_points=5)

    # Print results
    for i, (solution, objective_values) in enumerate(results):
        print(f"Solution {i + 1}:")
        print(f"  Variables: {solution}")
        print(f"  Objective values: {objective_values}")
        print()