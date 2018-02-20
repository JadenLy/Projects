# Yang Le   yangl23
# CSE 415 Assignment 3

'''
A QUIET2 Solving Tool problem formulation.
QUIET = Quetzal User Intelligence Enhancing Technology.
The XML-like tags used here serve to identify key sections of this 
problem formulation.  It is important that COMMON_CODE come
before all the other sections (except METADATA), including COMMON_DATA.
CAPITALIZED constructs are generally present in any problem
formulation and therefore need to be spelled exactly the way they are.
Other globals begin with a capital letter but otherwise are lower
case or camel case.
'''

#<METADATA>
QUIET_VERSION = "0.2"
PROBLEM_NAME = "Heuristics of Eight Puzzle"
PROBLEM_VERSION = "0.1"
PROBLEM_AUTHORS = ['Yang Le']
PROBLEM_CREATION_DATE = "19-OCT-2017"
PROBLEM_DESC=\
'''
This formulation of the Eight Puzzle problem uses generic
Python 3 constructs and has been tested with Python 3.6.
It is designed to work according to the QUIET2 tools interface.
'''
#</METADATA>

#<COMMON_CODE>
class State():
    def __init__(self, d):
        self.d = d

    def __eq__(self, s2):
        d1 = self.d
        d2 = s2.d
        return d1 == d2

    def __str__(self):
	    # Produces a textual description of a state.
	    # Might not be needed in normal operation with GUIs.
        d = self.d
        return str(d[0:3]) + '\n' + str(d[3:6]) + '\n' + str(d[6::]) + '\n'

    def __hash__(self):
        return (self.__str__()).__hash__()

    def __copy__(self):
        # Performs an appropriately deep copy of a state,
        # for use by operators in creating new states.
        news = State([])
        news.d = [i for i in self.d]
        return news

def can_move(s, start, end):
    '''Check, under the current state, the tile can be moved
    from start tile to end tile'''
    if start == end:
        return False
    if s.d[start] == 0 or s.d[end] != 0:
        return False
    if start not in Final_state or end not in Final_state:
        return False
    if abs(end % 3 - start % 3) == 1 and int(end/3) == int(start/3):
        return True
    if abs(start - end) / 3 == 1 and start % 3 == end % 3:  
        return True
    return False


def move(s, start, end):
    '''Move the tile at start position to end position'''
    news = s.__copy__()
    newd = news.d
    newd[end] = newd[start]
    newd[start] = 0
    return news

from math import sqrt
def h_euclidean(s):
    '''Calculate the Euclidean distance for each tile to reach 
    the correct position'''
    distance = 0
    for i in range(9):
        coord_s = (int(s.d.index(i) / 3), s.d.index(i) % 3)
        coord_f = (int(Final_state.index(i) / 3), Final_state.index(i) % 3)
        distance += sqrt((coord_s[0] - coord_f[0])**2 + (coord_s[1] - coord_f[1])**2)
    return distance
    
    
def h_hamming(s):
    '''Calculate the number of tiles that are out of place'''
    count = 0
    for i in range(len(s.d)):
        if s.d[i] != Final_state[i]:
            count += 1
    return count
    
    
def h_manhattan(s):
    '''Calculate the Manhattan distance for each tiles that are 
    out of place to reach the right position'''
    distance = 0
    for i in range(9):
        coord_s = (int(s.d.index(i) / 3), s.d.index(i) % 3)
        coord_f = (int(Final_state.index(i) / 3), Final_state.index(i) % 3)
        distance += abs(coord_s[0] - coord_f[0]) + abs(coord_s[1] - coord_f[1])
    return distance

def h_custom(s):
    '''Calculate a weighted Manhattan distance, weighted more havily 
    towards those are further away from 0'''
    distance = 0
    zero = (int(s.d.index(0) / 3), s.d.index(0) % 3)
    for i in range(1,9):
        coord_s = (int(s.d.index(i) / 3), s.d.index(i) % 3)
        coord_f = (int(Final_state.index(i) / 3), Final_state.index(i) % 3)
        zero_distance = abs(zero[0] - coord_s[0]) + abs(zero[1] - coord_s[1]) 
        distance += zero_distance * (abs(coord_s[0] - coord_f[0]) + abs(coord_s[1] - coord_f[1]))
    return distance
    
def goal_test(s):
    '''If the first two pegs are empty, then s is a goal state.'''
    return s.d == Final_state

def goal_message(s):
    return "The Eight Puzzle is solved!"

class Operator:
    def __init__(self, name, precond, state_transf):
        self.name = name
        self.precond = precond
        self.state_transf = state_transf

    def is_applicable(self, s):
        return self.precond(s)

    def apply(self, s):
        return self.state_transf(s)
#</COMMON_CODE>

#<COMMON_DATA>
Final_state = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#</COMMON_DATA>

#<OPERATORS>
from itertools import product
tiles = product(range(9), range(9))
OPERATORS = [Operator("Move tile from "+ str(p)+" to "+str(q),
                      lambda s,p1=p,q1=q: can_move(s,p1,q1),
                      # The default value construct is needed
                      # here to capture the values of p&q separately
                      # in each iteration of the list comp. iteration.
                      lambda s,p1=p,q1=q: move(s,p1,q1))
             for (p,q) in tiles]
#</OPERATOR>

#<INITIAL_STATE>
# puzzle10a.py:
CREATE_INITIAL_STATE = lambda: State([4, 5, 0, 1, 2, 3, 6, 7, 8])
# puzzle12a.py:
# CREATE_INITIAL_STATE = lambda: State([3, 1, 2, 6, 8, 7, 5, 4, 0])
# puzzle14a.py:
# CREATE_INITIAL_STATE = lambda: State([4, 5, 0, 1, 2, 8, 3, 7, 6])
# puzzle16a.py:
# CREATE_INITIAL_STATE = lambda: State([0, 8, 2, 1, 7, 4, 3, 6, 5])
#</INITIAL_STATE>

#<GOAL_TEST>
GOAL_TEST = lambda s: goal_test(s)
#</GOAL_TEST>

#<GOAL_MESSAGE_FUNCTION>
GOAL_MESSAGE_FUNCTION = lambda s: goal_message(s)
#</GOAL_MESSAGE_FUNCTION>


#<HEURISTICS> (optional)
HEURISTICS = {'h_hamming': h_hamming, 'h_euclidean': h_euclidean, 
              'h_manhattan' : h_manhattan, 'h_custom': h_custom}
#</HEURISTICS>