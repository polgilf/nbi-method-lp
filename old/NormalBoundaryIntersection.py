import pulp
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy, copy

from old.MultiObjectiveLinearProgram import MultiObjectiveLinearProgram, Solution

def distribute_line_points(A, B, K):
    '''
    Distribute K points along the line connecting points A and B
    Input: A, B (numpy arrays), K (int)
    Output: numpy array with K points (dimension K x 2)
    '''
    A, B = np.array(A), np.array(B)
    t = np.linspace(0, 1, K)
    return np.array([(1 - ti) * A + ti * B for ti in t])


def distribute_triangle_points(A, B, C, K):
    '''
    Distribute K points in the triangle formed by points A, B, and C
    Input: A, B, C (numpy arrays), K (int)
    Output: numpy array with K points (dimension K x 3)
    '''
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
    def __init__(self, MOLPproblem, num_ref_points=10):
        self.MOLPproblem = MOLPproblem # MOLP object
        self.prob = MOLPproblem.prob # pulp.LpProblem object
        self.num_ref_points = num_ref_points

        # Attributes from the problem, compute if not already computed
        self.individual_optima = MOLPproblem.individual_optima or MOLPproblem.compute_individual_optima()
        self.payoff_table = MOLPproblem.payoff_table or MOLPproblem.compute_payoff_table()
        self.ideal_point = MOLPproblem.ideal_point or MOLPproblem.compute_ideal_point()
        self.nadir_point = MOLPproblem.nadir_point or MOLPproblem.compute_nadir_point()
        
        # Attributes to be filled during the NBI method (using NBI method)
        self.ref_points_dict = None
        self.normal_vector = None
        self.solutions_dict = None # Dictionary with the representative Pareto front solutions (key: reference point name, value: solution object)

        # NBI method
        self.compute_ref_points()
        self.compute_normal_vector()
        #self.compute_subset_pareto_front()
    
    # Attributes to access data and results
    def num_objectives(self):
        return self.MOLPproblem.num_objectives()
    
    def num_variables(self):
        return self.MOLPproblem.num_variables()
    
    def objective_functions(self):
        return self.MOLPproblem.objective_functions
    
    def objective_names(self):
        return self.MOLPproblem.objective_names
    
    def ref_points_values(self):
        if self.ref_points_dict is None:
            raise ValueError("Reference points have not been computed yet.")
        return self.ref_points_dict.values()
    
    def ref_points_id(self):
        if self.ref_points_dict is None:
            raise ValueError("Reference points have not been computed yet.")
        return self.ref_points_dict.keys()
    
    def solutions_objective_values_array(self):
        if self.solutions_dict is None:
            raise ValueError("Subset Pareto front has not been computed yet.")
        return np.array([sol.objective_values() for sol in self.solutions_dict.values()])
    
    def solutions_variable_values_array(self):
        if self.solutions_dict is None:
            raise ValueError("Subset Pareto front has not been computed yet.")
        return np.array([sol.variable_values() for sol in self.solutions_dict.values()])
    
    def solutions_ref_point_to_obj_values(self):
        if self.solutions_dict is None:
            raise ValueError("Subset Pareto front has not been computed yet.")
        return {key: value.objective_values() for key, value in self.solutions_dict.items()}
    
    def subset_solution_list(self):
        if self.solutions_dict is None:
            raise ValueError("Subset Pareto front has not been computed yet.")
        return list(self.solutions_dict.values())
    
    # Methods to compute the NBI algorithm and intermediate steps
    def compute_ref_points(self):
        # For bi-objective problems, the reference points are distributed along the line connecting the individual optima
        if self.num_objectives() == 2:
            A = self.individual_optima[0].objective_values()
            B = self.individual_optima[1].objective_values()
            # Both inplace and return:
            ref_points = distribute_line_points(A, B, self.num_ref_points)
            self.ref_points_dict = {f'q{i+1}': ref_points[i] for i in range(ref_points.shape[0])}
            return self.ref_points_dict
        
        # For tri-objective problems, the reference points are distributed in the triangle formed by the individual optima
        elif self.num_objectives() == 3:
            A = self.individual_optima[0].objective_values()
            B = self.individual_optima[1].objective_values()
            C = self.individual_optima[2].objective_values()
            # Both inplace and return:
            ref_points_values = distribute_triangle_points(A, B, C, self.num_ref_points)
            self.ref_points_dict = {f'q{i+1}': ref_points_values[i] for i in range(ref_points_values.shape[0])}
        else:
            raise ValueError("The NBI method is only defined for bi-objective and tri-objective problems")
        return self.ref_points_dict
    
    def compute_normal_vector(self):
        # Compute normal vector in 2D (bi-objective problems)
        if self.num_objectives() == 2:
            A = self.individual_optima[0].objective_values()
            B = self.individual_optima[1].objective_values()
            direction_vector = B - A
            normal_vector_1 = np.array([-direction_vector[1], direction_vector[0]]) 
            normal_vector_2 = -normal_vector_1
            # Determine which normal vector points to the half-space where the ideal point is located
            if np.dot(normal_vector_1, self.ideal_point - A) > 0:
                self.normal_vector = normal_vector_1 / np.linalg.norm(normal_vector_1)
            else:
                self.normal_vector = normal_vector_2 / np.linalg.norm(normal_vector_1)
        # Compute normal vector in 3D (tri-objective problems)        
        elif self.num_objectives() == 3:
            A = self.individual_optima[0].objective_values()
            B = self.individual_optima[1].objective_values()
            C = self.individual_optima[2].objective_values()
            direction_vector_1 = B - A
            direction_vector_2 = C - A
            normal_vector_1 = np.cross(direction_vector_1, direction_vector_2)
            normal_vector_2 = -normal_vector_1
            # Determine which normal vector points to the half-space where the ideal point is located (TO BE CHECKED)
            if np.dot(normal_vector_1, self.ideal_point - A) > 0:
                self.normal_vector = normal_vector_1 / np.linalg.norm(normal_vector_1)
            else:
                self.normal_vector = normal_vector_2 / np.linalg.norm(normal_vector_1)
        else:
            raise ValueError("The NBI method is only defined for bi-objective and tri-objective problems")
        return self.normal_vector
    
    def solve_NBI_subproblem(self, ref_id):
        # Solve the NBI subproblem for a given reference point
        # Create a new problem to define the max t NBI subproblem
        NBI_subproblem = deepcopy(self.prob)

        # Set new variable t, objective function max t, and constraints reference_point + t * normal_vector == f
        t = pulp.LpVariable("t", 0, None)
        NBI_subproblem += t
        NBI_subproblem.sense = pulp.LpMaximize

        for i in range(self.num_objectives()):
            NBI_subproblem += self.ref_points_dict[ref_id][i] + t * self.normal_vector[i] == self.objective_functions()[i]

        # Solve the NBI subproblem
        NBI_subproblem.solve(pulp.PULP_CBC_CMD(msg=0))
        NBI_solution = Solution(NBI_subproblem, self.objective_names())
        NBI_solution.remove_variable("t")

        NBI_subproblem.solve(pulp.PULP_CBC_CMD(msg=0))

        print('** New iteration **')
        print('insolver reference point: ', ref_id, ' ', self.ref_points_dict[ref_id])

        print('results of subproblem:', {v.name: v.varValue for v in NBI_subproblem.variables()})

        print('Constraints of subproblem:', NBI_subproblem.constraints)
        print('Constraints of solution: ', NBI_solution.prob.constraints)
        
        print('insolver objective values:', NBI_solution.objective_values())
        '''
        print('** New iteration **')    
        print('Constraints: ', NBI_solution.prob.constraints)
        print('results of subproblem:', {v.name: v.varValue for v in NBI_subproblem.variables() if v.name in self.objective_names()})
        print('insolver reference point: ', ref_id, ' ', self.ref_points_dict[ref_id])
        print('insolver objective values: ', NBI_solution.objective_values())
        print('insolver decision values: ', NBI_solution.variable_values())
        '''

        return deepcopy(NBI_solution)


    def compute_subset_pareto_front(self):
        # If not already computed, compute reference points
        if self.ref_points_dict is None:
            self.ref_points_dict = self.compute_ref_points()
        if self.normal_vector is None:
            self.normal_vector = self.compute_normal_vector
        # Compute the representative Pareto front
        self.solutions_dict = {}
        # Solve NBI subproblems for each reference point
        for ref_id in self.ref_points_dict.keys():
            NBI_solution = self.solve_NBI_subproblem(ref_id)
            # Add the solution to the representative Pareto front
            self.solutions_dict[ref_id] = NBI_solution
        return self.solutions_dict


    
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
            plt.xlabel(self.MOLPproblem.objective_names[0])
            plt.ylabel(self.MOLPproblem.objective_names[1])
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
            ax.set_xlabel(self.MOLPproblem.objective_names[0])
            ax.set_ylabel(self.MOLPproblem.objective_names[1])
            ax.set_zlabel(self.MOLPproblem.objective_names[2])
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
            plt.xlabel(self.MOLPproblem.objective_names[0])
            plt.ylabel(self.MOLPproblem.objective_names[1])
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
            ax.set_xlabel(self.MOLPproblem.objective_names[0])
            ax.set_ylabel(self.MOLPproblem.objective_names[1])
            ax.set_zlabel(self.MOLPproblem.objective_names[2])
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
            ax.set_xlabel(self.MOLPproblem.objective_names[0])
            ax.set_ylabel(self.MOLPproblem.objective_names[1])
            ax.set_zlabel(self.MOLPproblem.objective_names[2])
            ax.set_title('Reference points and individual optima in 3D')
            plt.show()