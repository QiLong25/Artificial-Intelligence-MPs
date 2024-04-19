'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3

def compute_transition(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''

    P = np.zeros((model.M, model.N, 4, model.M, model.N))

    for r in range(model.M):
        for c in range(model.N):

            ## Terminal state
            if model.TS[r, c]:
                P[r, c, :, :, :] = 0
                continue

            ## Intend moving left
            # left
            if c-1 < 0 or model.W[r, c-1]:                      # reach boundary or wall
                P[r, c, 0, r, c] += model.D[r, c, 0]
            else:
                P[r, c, 0, r, c-1] += model.D[r, c, 0]
            # down
            if r+1 >= model.M or model.W[r+1, c]:               # reach boundary or wall
                P[r, c, 0, r, c] += model.D[r, c, 1]
            else:
                P[r, c, 0, r+1, c] += model.D[r, c, 1]
            # up
            if r-1 < 0 or model.W[r-1, c]:                      # reach boundary or wall
                P[r, c, 0, r, c] += model.D[r, c, 2]
            else:
                P[r, c, 0, r-1, c] += model.D[r, c, 2]

            ## Intend moving up
            # up
            if r-1 < 0 or model.W[r-1, c]:                      # reach boundary or wall
                P[r, c, 1, r, c] += model.D[r, c, 0]
            else:
                P[r, c, 1, r-1, c] += model.D[r, c, 0]
            # left
            if c-1 < 0 or model.W[r, c-1]:                      # reach boundary or wall
                P[r, c, 1, r, c] += model.D[r, c, 1]
            else:
                P[r, c, 1, r, c-1] += model.D[r, c, 1]
            # right
            if c+1 >= model.N or model.W[r, c+1]:               # reach boundary or wall
                P[r, c, 1, r, c] += model.D[r, c, 2]
            else:
                P[r, c, 1, r, c+1] += model.D[r, c, 2]

            ## Intend moving right
            # right
            if c + 1 >= model.N or model.W[r, c + 1]:           # reach boundary or wall
                P[r, c, 2, r, c] += model.D[r, c, 0]
            else:
                P[r, c, 2, r, c + 1] += model.D[r, c, 0]
            # up
            if r-1 < 0 or model.W[r-1, c]:                      # reach boundary or wall
                P[r, c, 2, r, c] += model.D[r, c, 1]
            else:
                P[r, c, 2, r-1, c] += model.D[r, c, 1]
            # down
            if r + 1 >= model.M or model.W[r + 1, c]:           # reach boundary or wall
                P[r, c, 2, r, c] += model.D[r, c, 2]
            else:
                P[r, c, 2, r + 1, c] += model.D[r, c, 2]

            ## Intend moving down
            # down
            if r + 1 >= model.M or model.W[r + 1, c]:           # reach boundary or wall
                P[r, c, 3, r, c] += model.D[r, c, 0]
            else:
                P[r, c, 3, r + 1, c] += model.D[r, c, 0]
            # right
            if c + 1 >= model.N or model.W[r, c + 1]:           # reach boundary or wall
                P[r, c, 3, r, c] += model.D[r, c, 1]
            else:
                P[r, c, 3, r, c + 1] += model.D[r, c, 1]
            # left
            if c-1 < 0 or model.W[r, c-1]:                      # reach boundary or wall
                P[r, c, 3, r, c] += model.D[r, c, 2]
            else:
                P[r, c, 3, r, c-1] += model.D[r, c, 2]

    return P

def compute_utility(model, U_current, P):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    U_current - The current utility function, which is an M x N array
    P - The precomputed transition matrix returned by compute_transition()

    Output:
    U_next - The updated utility function, which is an M x N array
    '''

    U_next = model.R + model.gamma * np.max(np.sum(np.sum(P * U_current.reshape(1, 1, 1, model.M, model.N), axis=4), axis=3), axis=2)

    return U_next

def value_iterate(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''

    diff = 1
    P = compute_transition(model)
    U = np.zeros((model.M, model.N))

    while diff > epsilon:
        U_next = compute_utility(model, U, P)
        diff = np.max(np.abs(U_next - U))
        U = U_next

    return U

def policy_evaluation(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP();
    
    Output:
    U - The converged utility function, which is an M x N array
    '''

    diff = 1
    U = np.zeros((model.M, model.N))

    while diff >= epsilon:
        U_next = model.R + model.gamma * np.sum(np.sum(model.FP * U.reshape(1, 1, model.M, model.N), axis=3), axis=2)
        diff = np.max(np.abs(U_next - U))
        U = U_next

    return U
