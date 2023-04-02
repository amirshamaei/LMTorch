import numpy as np
from scipy.optimize import minimize

# Define the Rosenbrock function
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Define the jacobian of the Rosenbrock function
def rosenbrock_jacobian(x):
    jacobian = np.zeros((2, 2))
    jacobian[0, 0] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    jacobian[0, 1] = 200 * (x[1] - x[0]**2)
    jacobian[1, 0] = 2 * (1 - x[0])
    jacobian[1, 1] = -200 * (x[1] - x[0]**2)
    return jacobian

# Set the initial guess for the solution
x0 = np.array([-2, 2])

# Use the Levenberg-Marquardt algorithm to minimize the Rosenbrock function
result = minimize(rosenbrock, x0, method='BFGS')

# Print the optimized solution
print(result.x)
