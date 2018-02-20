# Yang Le   yangl23
# CSE 415 Assignment 3
# Astar.py, Oct 2017 
# Based on ItrDFS.py, Ver 0.4a, October 14, 2017.

# A* Search of a problem space.
# The Problem should be given in a separate Python
# file using the "QUIET" file format.
# See the TowerOfHanoi.py example file for details.
# Examples of Usage:

# A small change was made on Oct. 14, so that backtrace
# uses None as the BACKLINK value for the initial state,
# just as in ItrDFS.py, rather than using -1 as it did
# in an earlier version.

# python3 AStar.py EightPuzzleWithHeuristics h_manhattan

import sys
from priorityq import PriorityQ

# DO NOT CHANGE THIS SECTION 
if sys.argv==[''] or len(sys.argv)<2:
    import EightPuzzleWithHeuristics as Problem
    curr = 'h_euclidean'
    heuristics = lambda s: Problem.HEURISTICS[curr](s)
    
else:
    import importlib
    Problem = importlib.import_module(sys.argv[1])
    heuristics = lambda s: Problem.HEURISTICS[sys.argv[2]](s)


print("\nWelcome to AStar")
print("\nThe Current Heuristics is " + curr)
COUNT = None
BACKLINKS = {}

# DO NOT CHANGE THIS SECTION
def runAStar():
    #initial_state = Problem.CREATE_INITIAL_STATE(keyVal)
    initial_state = Problem.CREATE_INITIAL_STATE()
    print("Initial State:")
    print(initial_state)
    global COUNT, BACKLINKS
    COUNT = 0
    BACKLINKS = {}
    path, name = AStar(initial_state)
    print(str(COUNT)+" states examined.")
    return path, name

# A star search algorithm
# TODO: finish A star implementation
def AStar(initial_state):
    global COUNT, BACKLINKS
    # TODO: initialze and put first state into 
    # priority queue with respective priority
    # add any auxiliary data structures as needed
    OPEN = PriorityQ()
    CLOSED = []
    BACKLINKS[initial_state] = None
    OPEN.insert(initial_state, heuristics(initial_state))
    
    while OPEN.__len__() > 0:
        item = OPEN.deletemin()
        S = item[0]
        weight = item[1]- heuristics(S)
        while S in CLOSED:
            item = OPEN.deletemin()
            S = item[0]
            weight = item[1]- heuristics(S)
        CLOSED.append(S)
        
        # DO NOT CHANGE THIS SECTION: begining 
        if Problem.GOAL_TEST(S):
            print(Problem.GOAL_MESSAGE_FUNCTION(S))
            path = backtrace(S)
            return path, Problem.PROBLEM_NAME
        # DO NOT CHANGE THIS SECTION: end

        # TODO: finish A* implementation
        COUNT += 1
        for op in Problem.OPERATORS:
            if op.precond(S):
                new_state = op.state_transf(S)
                if new_state not in CLOSED:
                    new_weight = weight + 1
                    if new_state not in OPEN:
                        BACKLINKS[new_state] = S
                        OPEN.insert(new_state, new_weight + heuristics(new_state))
                    elif OPEN.getpriority(new_state) - heuristics(new_state) > new_weight: 
                        BACKLINKS[new_state] = S
                        OPEN.remove(new_state)
                        OPEN.insert(new_state, new_weight + heuristics(new_state))

# DO NOT CHANGE
def backtrace(S):
    global BACKLINKS
    path = []
    while S:
        path.append(S)
        S = BACKLINKS[S]
    path.reverse()
    print("Solution path: ")
    for s in path:
        print(s)
    print("\nPath length = "+str(len(path)-1))
    return path    

if __name__=='__main__':
    path, name = runAStar()
