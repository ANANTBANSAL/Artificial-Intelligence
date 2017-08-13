#
# Artificial Intelligence
# HW2
# Author: Akanksha Tyagi
# akanksht@usc.edu

import sys
from operator import itemgetter
import copy


'''Declaring global variables
'''
lines = []
n_inf = float('-inf')
p_inf = float('inf')
p1_pref = []
p2_pref = []
cut_off_depth = 0
colors = []
initial_moves = {}
graph = {}
domain_dict = {}
neighbors_list = []
fo = open('output.txt', 'wb')


class State:
    def __init__(self):
        self.name = ""
        self.domain = []
        self.depth = 0
        self.color = ""
        self.action = []
        self.alpha = n_inf
        self.beta = p_inf

    def set_values(self, na, do, d, c, a, alpha, beta):
        self.name = na
        self.domain = do
        self.depth = d
        self.color = c
        self.action = a
        self.alpha = alpha
        self.beta = beta


'''
Function to perform arc consistency on states so that 
no two adjacent states have same colors
Used it only for initial moves
'''


def arc_consistency(s, color, graph):
    neighbors = []
    neighbors = list(graph[s])
    for n in neighbors:
        domain = []
        domain = list(domain_dict[n])
        if color in domain:
            domain.remove(color)
            domain_dict[n] = list(domain)


'''
Function to revise domains of adjacent states based on 
current move taken by any player
'''

def revise_domain(key, action):
    reduced_domain = list(domain_dict[key])
    neighbors = list(graph[key])
    for each in action:
        if each[0] in neighbors:
            if each[1] in reduced_domain:
                reduced_domain.remove(each[1])
    return reduced_domain


'''
Function to store preferences of the players
Returns a preference list in sorted order
'''

def storing_preferences(l):
    m = []
    int_max = l.split(',')
    test = map(lambda x: x.strip().split(":"), int_max)
    for x in test:
        t = ()
        c = x[0].strip()
        v = int(x[1].strip())
        t = t + (c,)
        t = t + (v,)
        m.append(t)
    sorted_m = sorted(m, key=itemgetter(1), reverse=True)
    return sorted_m



'''
Function to evaluate the value for terminal states
Iterates over a list of actions taken till now and calculates utility for
each player
Action list for each state contains (state name, color, player)
Returns evaluation value
'''
def Eval(state):
    utility1 = 0
    utility2 = 0
    action = list(state.action)
    for i in range(0, len(action)):
        active_player = action[i][2]
        if active_player == 'P2':
            utility2 += dict(p2_pref)[action[i][1]]
        elif active_player == 'P1':
            utility1 += dict(p1_pref)[action[i][1]]
    return utility1 - utility2


'''
Function to test if current state is at a depth >= cutoff, 
returns true in that case 
'''
def Cut_off_test(state):
    if state.depth >= cut_off_depth:
        return 1
    return 0

'''
Function to test if current state has successors or no
Returns true if successors list(Frontier) is empty  
'''
def terminal_test(state,frontier):
    if frontier:
        return 0
    return 1

'''
Function to test if the entire graph has been colored
Input: current state, states that have been colored already (gets this list from get_successors function)
Returns true if all states in the graph have been colored
'''
def graph_colored(state,states_already_colored):
    if set(graph.keys())==set(states_already_colored):
        return 1
    return 0

'''
Function to generate a list of successors
Successors for a state are those states which are adjacent to all the colored states till now
Input: current state, active player
returns a list of successors(sorted by name and color of state) and already colored states
'''
def get_successors(state, active_player):
    domain = []
    tup=()
    tup_check=[]
    tup_list = []
    states_already_colored = map(lambda x: x[0], state.action)
    for action in state.action:
        # Getting neighbors from main graph
        for each in sorted(graph[action[0]]):
            domain = []
            # check if it has already been colored
            if each not in states_already_colored:
                domain = revise_domain(each, state.action)
                if len(domain) > 0:
                    for color in domain:
                        tup=()
                        succ = State()
                        state_action = []
                        state_action = list(state.action)
                        new_action = (each, color, active_player)
                        state_action.append(new_action)
                        succ.set_values(each, domain, state.depth + 1, color, state_action, n_inf, p_inf)
                        tup=(succ.depth,succ.name,succ.color)
                        if tup not in tup_check:
                            tup_check.append(tup)
                            tup_list.append(succ)
    return sorted(tup_list, key=lambda element: (element.name, element.color)),states_already_colored

'''
Function to write output to the file
Input: curret state, alpha, beta, value
'''

def write_state(state, alpha, beta, value):
    fo.write(state.name + ', ' + state.color + ', ' + str(state.depth) + ', ' + str(
        value) + ', ' + str(alpha) + ', ' + str(
        beta) + "\n")


'''
Function used by max player(P1) to maximize value
'''

def Max_value(state, alpha, beta, depth):
    frontier_list = []
    states_already_colored=[]
    global best_action
    if Cut_off_test(state):
        state.alpha = alpha
        return Eval(state)

    v = n_inf
    frontier_list,states_already_colored = get_successors(state, "P1")
    if terminal_test(state,frontier_list)==1:
        state.alpha = alpha
        return Eval(state)
    if graph_colored(state,states_already_colored):
        state.alpha = alpha
        return Eval(state)

    for currentstate in frontier_list:
        # write_state(state, alpha, beta, v)
        fo.write(state.name + ', ' + state.color + ', ' + str(state.depth) + ', ' + str(
            value) + ', ' + str(alpha) + ', ' + str(
            beta) + "\n")

        # child = copy.deepcopy(currentstate)
        child=currentstate
        result_value = Min_value(child, alpha, beta, depth + 1)
        v = max(v, result_value)
        # write_state(child, alpha, child.beta, result_value)
        fo.write(child.name + ', ' + child.color + ', ' + str(child.depth) + ', ' + str(
            value) + ', ' + str(alpha) + ', ' + str(
            child.beta) + "\n")

        if v >= beta:
            del child.action[-1]
            state.alpha = alpha
            return v
        if v > alpha:
            alpha=v
            if state is root_state:
                best_action=child
        state.alpha = alpha
    return v

'''
Function used by min player(P2) to choose its best move (minimizes the value)
'''
def Min_value(state, alpha, beta, depth):
    frontier_list = []
    states_already_colored=[]
    global best_action
    if Cut_off_test(state) == 1:
        state.beta = beta
        return Eval(state)
    v = p_inf
    frontier_list,states_already_colored = get_successors(state, "P2")
    if terminal_test(state,frontier_list):
        state.beta = beta
        return Eval(state)

    if graph_colored(state,states_already_colored):
        state.beta = beta
        return Eval(state)

    for currentstate in frontier_list:
        # write_state(state, alpha, beta, v)
        fo.write(state.name + ', ' + state.color + ', ' + str(state.depth) + ', ' + str(
            value) + ', ' + str(alpha) + ', ' + str(
            beta) + "\n")

        # child = copy.deepcopy(currentstate)
        child = currentstate
        result_value = Max_value(child, alpha, beta, depth + 1)
        v = min(v, result_value)
        # write_state(child, child.alpha, beta, result_value)
        fo.write(child.name + ', ' + child.color + ', ' + str(child.depth) + ', ' + str(
            value) + ', ' + str(child.alpha) + ', ' + str(
            beta) + "\n")

        if v <= alpha:
            del child.action[-1]
            state.beta = beta
            return v
        if v < beta:
            beta=v
            if state is root_state:
                best_action=child
        state.beta = beta
    return v


'''
Function to implement alpha beta search with pruning
'''
def alpha_beta_search(state, depth):
    global best_action
    v = Max_value(state, n_inf, p_inf, depth)
    write_state(state, state.alpha, state.beta, v)
    fo.write(best_action.name + ', ' + best_action.color + ', ' + str(v))
    fo.close()
    return v


'''
Parsing input
'''

with open(sys.argv[2]) as f:
    lines.extend(f.read().splitlines())

colors = lines[0].split(',')
colors = map(lambda x: x.strip(), colors)

# Sorting alphabetically
colors = sorted(colors)

# Storing preferences of both players
p1_pref = storing_preferences(lines[3])
p2_pref = storing_preferences(lines[4])

''' storing initial moves by both players
    initial_moves has state name as keys 
    and value contains the color assigned to that state
'''
init_move = lines[1].split(",")
init_move = map(lambda x: x.strip(), init_move)
state_action = []
for each in init_move:
    temp = each.split(":")
    tempsplit = temp[1].strip().split("-")
    color = tempsplit[0]
    player = tempsplit[1].strip()
    node = temp[0].strip()
    initial_moves[node] = color
    if player == '1':
        active_player = 'P1'
    elif player == '2':
        active_player = 'P2'
    action = (node, color, active_player)
    state_action.append(action)

last_action = node
cut_off_depth = int(lines[2].strip())

# storing the graph
temp_graph = lines[5:]
temp_graph = map(lambda x: x.split(":"), temp_graph)

for each in temp_graph:
    graph[each[0].strip()] = each[1].strip().split(",")

for key, value in graph.iteritems():
    for each in range(0, len(value)):
        value[each] = value[each].strip()
    # Sorting the neighbors in alphabetical order
    graph[key] = sorted(value)

# Initial assignment of domains(sorted alphabetically) to all states
for key in graph:
    domain_dict[key] = colors

# Modifying the domains of states which have been assigned a color initially
for key, value in initial_moves.iteritems():
    domain_dict[key] = [value]

# Call arc consistency function to revise domains according to initial moves
for key, value in initial_moves.iteritems():
    arc_consistency(key, value, graph)

key = last_action
root_state = State()
root_state.set_values(key, [initial_moves[key]], 0, initial_moves[key], state_action, n_inf, p_inf)
res = alpha_beta_search(root_state, 0)
