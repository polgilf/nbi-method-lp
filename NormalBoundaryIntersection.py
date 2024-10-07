import pulp
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy, copy

from MultiObjectiveLinearProgram import MultiObjectiveLinearProgram, Solution

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
        self.reference_points_dict = None # {'q1' : np.array()}
        self.normal_vector = None # Normal vector to the hyperplane defined by the individual optima
        self.representative_pareto_front = None # List of solutions
        self.representative_pareto_front_values = None # Numpy array of objective values sorted!
        self.reference_point_to_solution_dict = {} # Dictionary that maps reference points to solutions

    def compute_reference_points(self):
        # For bi-objective problems, the reference points are distributed along the line connecting the individual optima
        if self.num_objectives == 2:
            A = self.individual_optima[0].objective_values
            B = self.individual_optima[1].objective_values
            # Both inplace and return:
            self.reference_points = distribute_line_points(A, B, self.num_reference_points)
        
        # For tri-objective problems, the reference points are distributed in the triangle formed by the individual optima
        elif self.num_objectives == 3:
            A = self.individual_optima[0].objective_values
            B = self.individual_optima[1].objective_values
            C = self.individual_optima[2].objective_values
            # Both inplace and return:
            self.reference_points = distribute_triangle_points(A, B, C, self.num_reference_points)
        self.reference_points_dict = {f'q{i+1}': self.reference_points[i] for i in range(self.reference_points.shape[0])}
        return self.reference_points
        
    def compute_normal_vector(self):
        # Compute the normal vector to the hyperplane defined by the individual optima
        # This vector should point to the half-space where the ideal point is located
        # Compute normal vector in 2D (bi-objective problems)
        if self.num_objectives == 2:
            A = self.individual_optima[0].objective_values
            B = self.individual_optima[1].objective_values
            direction_vector = B - A
            normal_vector_1 = np.array([-direction_vector[1], direction_vector[0]]) 
            normal_vector_2 = -normal_vector_1
            # Determine which normal vector points to the half-space where the ideal point is located
            if np.dot(normal_vector_1, self.ideal_point - A) > 0:
                self.normal_vector = normal_vector_1 / np.linalg.norm(normal_vector_1)
            else:
                self.normal_vector = normal_vector_2 / np.linalg.norm(normal_vector_1)

        # Compute normal vector in 3D (tri-objective problems)        
        if self.num_objectives == 3:
            A = self.individual_optima[0].objective_values
            B = self.individual_optima[1].objective_values
            C = self.individual_optima[2].objective_values
            direction_vector_1 = B - A
            direction_vector_2 = C - A
            normal_vector_1 = np.cross(direction_vector_1, direction_vector_2)
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
        nbi_subproblem.sense = pulp.LpMaximize
        normal_vector = self.compute_normal_vector()
        # Set new variable t, objective function max t, and constraints reference_point + t * normal_vector == f
        t = pulp.LpVariable("t",0,None)
        nbi_subproblem.setObjective(t)
        for i in range(self.num_objectives):
            nbi_subproblem += reference_point[i] + t * normal_vector[i] == self.MOLPproblem.objective_functions[i]
        # Solve the NBI subproblem
        nbi_subproblem.solve(pulp.PULP_CBC_CMD(msg=0))
        # Return the solution
        nbi_solution = Solution(nbi_subproblem, objective_names=self.MOLPproblem.objective_functions_names)
        # If t = 0 (i.e., the reference point is on the Pareto front), set the objective values to the reference point (to avoid multiple-solution problem)
        tvalue = copy(t.varValue)
        if t.varValue == 0.0:
            variable_values_dict = {}
            for objective in nbi_solution.objective_names:
                variable_values_dict[objective] = float(reference_point[nbi_solution.objective_names.index(objective)])
            nbi_solution.replace_solution_giving_objective_dict(variable_values_dict)
        # Remove variable t form nbi_solution
        nbi_solution.decision_dict.pop('t', None)
        nbi_solution.decision_values = np.array([value for value in nbi_solution.decision_values if value != tvalue])
        """
        print('** New iteration **')
        print('t value: ', tvalue)
        print('insolver reference point: ', reference_point)
        print('insolver normal vector: ', normal_vector)
        print('insolver objective values: ', nbi_solution.objective_values)
        print('insolver decision values: ', nbi_solution.decision_values)
        """
        return nbi_solution

    def compute_representative_pareto_front(self):
        # If not already computed, compute reference points
        if self.reference_points is None:
            self.reference_points = self.compute_reference_points()
        # Compute the representative Pareto front
        self.representative_pareto_front = []
        self.reference_point_to_solution_dict = {}
        # Solve NBI subproblems for each reference point
        for reference_name, reference_point in self.reference_points_dict.items():
            nbi_solution = self.solve_NBI_subprolbem(reference_point)
            # Add the solution to the representative Pareto front
            self.reference_point_to_solution_dict[reference_name] = nbi_solution
            self.representative_pareto_front.append(nbi_solution)
        # Sort the representative Pareto front based on the first objective value
        self.representative_pareto_front_values = np.array(
            sorted((sol.objective_values for sol in self.representative_pareto_front), key=lambda x: x[0])
        )
        return self.representative_pareto_front
    
    def plot_solutions_in_objective_space(self):
        if self.num_objectives == 2:
            # Plot the representative solutions in the objective space
            solutions = self.representative_pareto_front
            # Extract objective values from solutions
            objective_values = [solution.objective_values for solution in solutions]
            # Convert to numpy array for easier plotting
            objective_values = np.array(objective_values)
            # Plot the Pareto front
            plt.figure(figsize=(6, 6))
            plt.scatter(objective_values[:, 0], objective_values[:, 1], color='red', zorder=5)
            plt.xlabel(self.MOLPproblem.objective_functions_names[0])
            plt.ylabel(self.MOLPproblem.objective_functions_names[1])
            plt.title('Representative Pareto Front, NBI method with ' + str(self.num_reference_points) + ' reference points')
            plt.grid(True)
            plt.show()
        elif self.num_objectives == 3:
            # Plot the representative solutions in the objective space
            solutions = self.representative_pareto_front
            # Extract objective values from solutions
            objective_values = [solution.objective_values for solution in solutions]
            # Convert to numpy array for easier plotting
            objective_values = np.array(objective_values)
            # Plot the Pareto front
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(objective_values[:, 0], objective_values[:, 1], objective_values[:, 2], color='red', s=50, zorder=5)
            ax.set_xlabel(self.MOLPproblem.objective_functions_names[0])
            ax.set_ylabel(self.MOLPproblem.objective_functions_names[1])
            ax.set_zlabel(self.MOLPproblem.objective_functions_names[2])
            ax.set_title('Representative Pareto Front, NBI method with ' + str(self.num_reference_points) + ' reference points')
            plt.show()
        else:
            print("Plotting is only available for bi-objective and tri-objective problems")


    def plot_NBI_method(self):
        if self.num_objectives == 2:
            # Plot the representative solutions in the objective space
            solutions = self.representative_pareto_front         
            # Extract objective values from solutions
            objective_values = self.representative_pareto_front_values
            # Convert to numpy array for easier plotting
            #objective_values = np.array(objective_values)
            # Reference points
            reference_points = np.array(self.reference_points)
            # Plot the Pareto front
            plt.figure(figsize=(6, 6))
            # Plot solution points
            plt.scatter(objective_values[:, 0], objective_values[:, 1], color='red', zorder=5)
            # Plot reference points
            plt.scatter(reference_points[:, 0], reference_points[:, 1], color='blue', zorder=5)
            # Plot the lines connecting the reference points
            for i in range(self.num_reference_points - 1):
                plt.plot([reference_points[i, 0], reference_points[i + 1, 0]], [reference_points[i, 1], reference_points[i + 1, 1]], color='gray', linestyle='--', zorder=1)
            # Plot normal vectors as arrows, from reference points to solutions (first reference point to first solution)
            for i in range(self.num_reference_points):
                plt.arrow(reference_points[i, 0], reference_points[i, 1], objective_values[i, 0] - reference_points[i, 0], objective_values[i, 1] - reference_points[i, 1], 
                          head_width=0.5, head_length=0.5, fc='gray', ec='gray', zorder=3)
            plt.xlabel(self.MOLPproblem.objective_functions_names[0])
            plt.ylabel(self.MOLPproblem.objective_functions_names[1])
            plt.title('NBI method, NBI method with ' + str(self.num_reference_points) + ' reference points')
            plt.grid(True)
            plt.show()

        elif self.num_objectives == 3:
            # Plot the representative solutions in the objective space
            solutions = self.representative_pareto_front
            # Extract objective values from solutions
            objective_values = self.representative_pareto_front_values
            # Reference points
            reference_points = np.array(self.reference_points)
            # Plot the Pareto front
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            # Plot solution points
            ax.scatter(objective_values[:, 0], objective_values[:, 1], objective_values[:, 2], color='red', s=50, zorder=5)
            # Plot reference points
            ax.scatter(reference_points[:, 0], reference_points[:, 1], reference_points[:, 2], color='blue', s=50, zorder=5)
            # Plot normal vectors as arrows, from reference points to solutions (first reference point to first solution)
            # Use reference_points_dict and reference_point_to_solution_dict matching names (keys) to map reference points to solutions
            ax.set_xlabel(self.MOLPproblem.objective_functions_names[0])
            ax.set_ylabel(self.MOLPproblem.objective_functions_names[1])
            ax.set_zlabel(self.MOLPproblem.objective_functions_names[2])
            ax.set_title('NBI method, NBI method with ' + str(self.num_reference_points) + ' reference points')
            plt.show()

    def plot_3D_reference_points(self):
        '''
        Plot the reference points and the individual optima in 3D
        '''
        if self.num_objectives != 3:
            print("Plotting is only available for tri-objective problems")
        else:
            # Plot the reference points and the individual optima in 3D
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            # Plot the reference points
            reference_points = np.array(self.reference_points)
            ax.scatter(reference_points[:, 0], reference_points[:, 1], reference_points[:, 2], color='red', s=50, zorder=5)
            # Plot the individual optima
            individual_optima = np.array([sol.objective_values for sol in self.individual_optima])
            ax.scatter(individual_optima[:, 0], individual_optima[:, 1], individual_optima[:, 2], color='gray', s=50, zorder=5)
            # Set labels
            ax.set_xlabel(self.MOLPproblem.objective_functions_names[0])
            ax.set_ylabel(self.MOLPproblem.objective_functions_names[1])
            ax.set_zlabel(self.MOLPproblem.objective_functions_names[2])
            ax.set_title('Reference points and individual optima in 3D')
            plt.show()