class Node():

    def __init__(self, name):
        self.name = name
        self.transition = [None, None]

    def set_transition(self, _input, transition):
        self.transition[_input] = transition

    def next(self, _input):
        return self.transition[_input]

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

class DFA():

    def __init__(self, initial_state=None, states=None, final_states=None):
        #assert initial_state != None, "must specify initial_state"
        #assert states != None, "must specify set of states"
        #assert final_states != None, "must specify set of final states"
        self.initial_state = initial_state
        self.current_state = initial_state
        self.states = states
        self.final_states = final_states

    def transition_table(self):
        print(" ","|","0","|","1")
        print("---------")
        for state in self.states:
            print(state.name, "|", state.next(0).name, "|", state.next(1).name)

    def next(self, _input):
        assert _input == 0 or _input == 1, "input must be 0 or 1"
        self.current_state = self.current_state.next(_input)

    def minimize(self):
        equivalence_sets = [[], []]
        another_iteration = True

        # zero equivalence
        for state in self.states:
            if state in self.final_states:
                equivalence_sets[1].append(state)
            else: equivalence_sets[0].append(state)

        # higher order equivalences
        while another_iteration:
            tmp_sets = [[]]
            current_set = tmp_sets[-1]
            for e_set in equivalence_sets:
                if current_set != [] and current_set == tmp_sets[-1]: tmp_sets.append([])
                current_set = tmp_sets[-1]
                for s1 in e_set:
                    already = False
                    for tmp_set in tmp_sets:
                        if s1 in tmp_set: already = True
                    if already: continue
                    current_set.append(s1)
                    for s2 in e_set:
                        if s1 == s2: continue
                        if s2 in tmp_sets[-1]: continue
                        e0 = False
                        e1 = False
                        for _set in equivalence_sets:
                            e0 = e0 or (s1.next(0) in _set and s2.next(0) in _set)
                            e1 = e1 or (s1.next(1) in _set and s2.next(1) in _set)
                        if e0 and e1: current_set.append(s2)
                        else: tmp_sets.append([s2])
                print(tmp_sets, equivalence_sets)
            if tmp_sets == equivalence_sets: another_iteration = False
            equivalence_sets = tmp_sets

#        print(equivalence_sets)
        states = []
        final_states = []
        initial_state = None
        for e_set in equivalence_sets:
            name = ""
            is_initial_state = False
            for state in e_set:
                name += state.name
                is_initial_state = state == self.initial_state
            states.append(Node(name))
            if is_initial_state: initial_state = states[-1] 
            if state in self.final_states: final_states.append(states[-1])

        for e_set in equivalence_sets:
            for state in states:
                if e_set[0].name not in state.name: continue
                for s in states:
                    if e_set[0].next(0).name in s.name: state.set_transition(0, s)
                    if e_set[0].next(1).name in s.name: state.set_transition(1, s)
        return DFA(initial_state=initial_state, states=states, final_states=final_states)

def get_states1():
    states = []
    for x in range(65, 73):
        states.append(Node(chr(x)))

    states[0].set_transition(0, states[1])
    states[0].set_transition(1, states[5])
    states[1].set_transition(0, states[6])
    states[1].set_transition(1, states[2])
    states[2].set_transition(0, states[0])
    states[2].set_transition(1, states[2])
    states[3].set_transition(0, states[2])
    states[3].set_transition(1, states[6])
    states[4].set_transition(0, states[7])
    states[4].set_transition(1, states[5])
    states[5].set_transition(0, states[2])
    states[5].set_transition(1, states[6])
    states[6].set_transition(0, states[6])
    states[6].set_transition(1, states[4])
    states[7].set_transition(0, states[6])
    states[7].set_transition(1, states[2])

    return states

def get_states2():
    states = []
    for x in range(65, 70):
        states.append(Node(chr(x)))

    states[0].set_transition(0, states[1])
    states[0].set_transition(1, states[2])
    states[1].set_transition(0, states[1])
    states[1].set_transition(1, states[3])
    states[2].set_transition(0, states[1])
    states[2].set_transition(1, states[2])
    states[3].set_transition(0, states[1])
    states[3].set_transition(1, states[4])
    states[4].set_transition(0, states[1])
    states[4].set_transition(1, states[2])

    return states


if __name__ == "__main__":
    states = get_states1()

    dfa = DFA(initial_state=states[0], states=states, final_states=[states[2]])
    dfa.transition_table()
    min_dfa = dfa.minimize()
    min_dfa.transition_table()
