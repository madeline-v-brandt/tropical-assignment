import numpy as np
from functools import lru_cache
from scipy.optimize import linear_sum_assignment
from ortools.linear_solver import pywraplp
import math

def tropcost(X) :
    """Return the minimal cost of an assignment problem by naive tropical recursion"""
    if X.shape == (1,1): # what to do with a 1x1 matrix
        return X[0,0]
    best = math.inf # this will be our answer
    for j in range(len(X)): # do cofactor expansion along first column, recursively calling tropdet on the minor
        candidate = X[j,0] + tropcost(np.delete(np.delete(X,j,axis=0),0,axis=1))
        best = min(best, candidate) #update best if we find a better candidate value
    return best

def tropassignment(X) :
    """Solve the assignment problem by naive tropical recursion"""
    # take np array and return the best assignment as a list of tuples, indexing from 0
    if X.shape == (1,1): #what to do with 1x1 matrix
        return [X[0,0],np.array([[0,0]])] # returning the value of tropdet, together with array of coordinates to find the optimal value (aka the assignment)
    best = math.inf
    bestindices = np.array([])
    n = len(X)
    for j in range(n): #run through column indices j
        smallanswer = tropassignment(np.delete(np.delete(X,j,axis=1),n-1,axis=0)) # find the answer for the submatrix, delting last row and jth column
        candidate = X[n-1,j] + smallanswer[0] #our candidate value for tropdet
        if candidate < best: #if we find a minor with a better value...
            best = candidate #update the best value
            #fiddle with the indices before updating the value of bestindices
            unsortedindices=smallanswer[1] #indices returned after calling tropassignment on the small matrix
            indices= unsortedindices[unsortedindices[:, 1].argsort()] #sort these by what column they appear in
            for i in range(len(indices)): #translate indices from small matrix to the big matrix -- need to add 1 in the column for anythign appearing after j
                if i >= j:
                    indices[i] = indices[i] + np.array([0,1])
            indices = np.append(indices, np.array([[n-1,j]]),axis=0) #update the indices with the ones we just computed, together with the index for the last row
            bestindices = indices #update the value of bestindices
    return [best,bestindices[bestindices[:, 0].argsort()]]


def tropsubsetDPcost(X) :
    """Return the minimal cost of an assignment problem using subset dynamic programming"""
    # takes a matrix X and returns the minimal cost
    storedvalues = {} # this is going to be a dictionary of stored F(S) values as we compute them
    def F(X, S = range(len(X))) : # take cost matrix X as np array and a subset S of column indices. Return lowest cost for assigning first |S| rows to the columns given by S
        S = frozenset(S)
        # first check if S is in the known values
        if S in storedvalues:
            return storedvalues[S] 
        # base case: what to do with a 1x1 matrix
        if len(S) == 1:
            storedvalues[S] = X[0,next(iter(S))] ## next iter S gets the element in S
            return storedvalues[S]
        storedvalues[S] = min(X[len(S)-1,j] + F(X,S - {j}) for j in S )
        return storedvalues[S]
    return F(X)

def tropsubsetDPassignment(X) :
    """Solve the tropical assignment problem using subset dynamic programming"""
     # takes a matrix X and returns the minimal cost and the assignment that gets you there
    storedvalues = {} # this is going to be a dictionary of stored F(S) values as we compute them
    choice = {} ## remembering index choices

    def F(S = range(len(X))) : # take cost matrix X as np array and a subset S of column indices. Return lowest cost for assigning first |S| rows to the columns given by S

        S = frozenset(S)

        # first check if S is in the known values
        if S in storedvalues:
            return storedvalues[S] 
        
        # base case: what to do with a 0x0 matrix
        if len(S) == 0:
            return 0
        
        best_cost = float("inf")
        best_j = None

        for j in S:
            candidate = X[len(S)-1,j] + F(S - {j})

            if candidate < best_cost:
                best_cost = candidate
                best_j = j
        
        storedvalues[S] = best_cost
        choice[S] = best_j
        return storedvalues[S]

    #now we need to reconstruct the assignment
    columns = {} #constructing a column dictionary

    S = frozenset(range(len(X))) #start with the full set 
    cost = F(S)

    while len(S) > 0:
        j = choice[S] #get the assignment of the last row 
        columns[len(S)-1] = j # last row gets assigned jth column
        S = S - {j}

   # assignment = [np.array[i,columns[i]] for i in range(len(X))] 

    return [cost, columns]

# make a version of tropical subset dp that ignores entries larger than M = 999
def forbiddentropsubsetDPassignment(X, M = 999) :
    """Solve the assignment problem using sparsity-aware tropical subset DP"""
     # takes a matrix X and returns the minimal cost and the assignment that gets you there
    storedvalues = {} # this is going to be a dictionary of stored F(S) values as we compute them
    choice = {} ## remembering index choices

    def F(S = range(len(X))) : # take cost matrix X as np array and a subset S of column indices. Return lowest cost for assigning first |S| rows to the columns given by S

        S = frozenset(S)

        # first check if S is in the known values
        if S in storedvalues:
            return storedvalues[S] 
        
        # base case: what to do with a 0x0 matrix
        if len(S) == 0:
            return 0
        
        best_cost = float("inf")
        best_j = None

        for j in S:
            if X[len(S)-1,j] < M :
                candidate = X[len(S)-1,j] + F(S - {j})

                if candidate < best_cost:
                    best_cost = candidate
                    best_j = j
        
        storedvalues[S] = best_cost
        choice[S] = best_j
        return storedvalues[S]

    #now we need to reconstruct the assignment
    columns = {} #constructing a column dictionary

    S = frozenset(range(len(X))) #start with the full set 
    cost = F(S)

    while len(S) > 0:
        j = choice[S] #get the assignment of the last row 
        columns[len(S)-1] = j # last row gets assigned jth column
        S = S - {j}

   # assignment = [np.array[i,columns[i]] for i in range(len(X))] 

    return [cost, columns]

def MIPassignment(costs):
    """Solve the assignment problem as binary MIP using SCIP"""
    num_workers = len(costs)
    num_tasks = len(costs[0])
    # Solver
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return
    # Variables
    # x[i, j] is an array of 0-1 variables, which will be 1
    # if worker i is assigned to task j.
    x = {}
    for i in range(num_workers):
        for j in range(num_tasks):
            x[i, j] = solver.IntVar(0, 1, "")
    # Constraints
    # Each worker is assigned to at most 1 task.
    for i in range(num_workers):
        solver.Add(solver.Sum([x[i, j] for j in range(num_tasks)]) <= 1)
    # Each task is assigned to exactly one worker.
    for j in range(num_tasks):
        solver.Add(solver.Sum([x[i, j] for i in range(num_workers)]) == 1)
    # Objective
    objective_terms = []
    for i in range(num_workers):
        for j in range(num_tasks):
            objective_terms.append(costs[i][j] * x[i, j])
    solver.Minimize(solver.Sum(objective_terms))
    # Solve
    status = solver.Solve()
    # Print solution.
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        indices = []
        for i in range(num_workers):
                    for j in range(num_tasks):
                        # Test if x[i,j] is 1 (with tolerance for floating point arithmetic).
                        if x[i, j].solution_value() > 0.5:
                            indices.append([i,j])   
        return [np.int64(solver.Objective().Value()), np.array(indices)]
        
    else:
        print("No solution found.")


def LPassignment(costs):
    """Solve the LP relaxation of the assignment problem using GLOP"""
    num_workers = len(costs)
    num_tasks = len(costs[0])
    # Solver
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        return
    # Variables
    # x[i, j] is an array of 0-1 variables, which will be 1
    # if worker i is assigned to task j.
    x = {}
    for i in range(num_workers):
        for j in range(num_tasks):
            x[i, j] = solver.NumVar(0, 1, f"x_{i}_{j}")
    # Constraints
    # Each worker is assigned to at most 1 task.
    for i in range(num_workers):
        solver.Add(solver.Sum([x[i, j] for j in range(num_tasks)]) <= 1)
    # Each task is assigned to exactly one worker.
    for j in range(num_tasks):
        solver.Add(solver.Sum([x[i, j] for i in range(num_workers)]) == 1)
    # Objective
    objective_terms = []
    for i in range(num_workers):
        for j in range(num_tasks):
            objective_terms.append(costs[i][j] * x[i, j])
    solver.Minimize(solver.Sum(objective_terms))
    # Solve
    status = solver.Solve()
    # Print solution.
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        indices = []
        for i in range(num_workers):
                    for j in range(num_tasks):
                        # Test if x[i,j] is 1 (with tolerance for floating point arithmetic).
                        if x[i, j].solution_value() > 0.5:
                            indices.append([i,j])   
        return [np.int64(solver.Objective().Value()), np.array(indices)]
        
    else:
        print("No solution found.")