import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

class CourseMechanism:
    def __init__(self, n, m, var1=1, var2=1, alpha=1):
        self.n = n # number of students
        self.m = m # number of teachers

        self.var1 = var1 # determines how correlated first round preferences are across students
        self.var2 = var2 # determines how correlated second round preferences are with first round

        self.alpha = alpha # the amount that class 0 increases in popularity during shopping week

        self.teacher_preferences = []
        self.student_preferences = []
        self.student_matching = {}

        self.TTCpref = {}
        self.G = nx.DiGraph()
    
    """
    Returns:
        List of preference values for each student/teacher for each teacher/student
    """
    def generate_preferences(self):
        # Sample teacher preferences randomly
        self.teacher_preferences = [{i:np.random.uniform(0.0, 10.0) for i in range(self.n)} for _ in range(self.m)]
        # Sample student preferences from multivariate normal distribution
        means = np.random.uniform(0.0, 10.0, self.m)
        cov = np.diag(np.array([self.var1] * self.m))
        for _ in range(self.n):
            sample_prefs = np.random.multivariate_normal(means, cov)
            self.student_preferences.append({j:sample_prefs[j] for j in range(self.m)})
    
    """
    Args:
        bump: True if one class becomes popular during shopping week
        p: proportion of students who receive a bump
    Returns:
        List of preference values for each student for each teacher
    """
    def resample_preferences(self, bump=False, p=1):
        for i, student in enumerate(self.student_preferences):
            means = np.array(list(student.values()))
            cov = np.diag(np.array([self.var2] * self.m))
            sample_prefs = np.random.multivariate_normal(means, cov)
            self.student_preferences[i] = {j:sample_prefs[j] for j in range(self.m)}
            if bump and i > 0:
                if random.random() < p:
                    self.student_preferences[i][0] += self.alpha * abs(self.student_preferences[i][0])

    """
    Args: 
        subset: subset of agents from preferences to rank
        preferences: preference (0-10 float) for each agent in index i
    Returns:
        List of indices ranked by most preferred first
    """
    def sort_preferences(self, subset, preferences):
        subset_dict = {index: preferences[index] for index in subset}
        sorted_pref = sorted(subset_dict.items(), key=lambda x: x[1], reverse=True)
        ranked_indices = [index for index, _ in sorted_pref]
        return ranked_indices

    """
    Returns:
        Most preferred class for every student
    """
    def print_student_preferences(self):
        print_dict = {}
        for i, student in enumerate(self.student_preferences):
            full_preference = np.array(list(student.values()))
            max_preference = np.argmax(full_preference)
            print_dict[i] = max_preference
        return print_dict

    """
    Returns:
        The probability that a student receives their most preferred matching
    """
    def prob_success(self, single=False):
        successful = 0
        if not single:
            for i, student in enumerate(self.student_preferences):
                full_preference = np.array(list(student.values()))
                max_preference = np.argmax(full_preference)
                if max_preference == self.student_matching[i]:
                    successful += 1
            return successful / self.n
        if single:
            full_preference = np.array(list(self.student_preferences[0].values()))
            max_preference = np.argmax(full_preference)
            if max_preference == self.student_matching[0]:
                successful += 1
            return successful

    """
    Args:
        capacity: maximum number of students allowed to each class
        deviate: true if agent 0 should deviate
    Returns:
        a dictionary mapping each student to their accepted class
    """
    def studentDA(self, capacity, deviate=False):
        students = [self.sort_preferences([i for i in range(self.m)], preference) for preference in self.student_preferences]
        assert(len(students[i]) == self.m for i in range(self.n))
        if deviate:
            # Move class 0 to the top of student 0's preference ordering, pushing everything else down
            students[0].remove(0)
            students[0].insert(0, 0)
        teacher_matching = {i:[] for i in range(self.m)}
        student_matching = {j:-1 for j in range(self.n)} # -1 implies unmatched
        exhausted = False
        all_matched = False
        condition = True
        while condition:
            for i in range(self.m):
                assert(len(teacher_matching[i]) <= capacity)
            to_check = {} # Dict of teachers to check for each n student, -1 implies no proposal was made this round
            # Make proposals for unmatched students
            for i, ranking in enumerate(students):
                # If student is currently matched and still has proposals left
                if student_matching[i] == -1 and len(ranking) > 0:
                    proposal = ranking.pop(0)
                    to_check[i] = proposal
            # Iterate and evaluate through each proposal
            for student, teacher in to_check.items():
                # Just accept proposal if not at capacity
                if len(teacher_matching[teacher]) < capacity: 
                    teacher_matching[teacher].append(student)
                    student_matching[student] = teacher 
                    teacher_matching[teacher] = self.sort_preferences(teacher_matching[teacher], self.teacher_preferences[teacher])
                # If at capacity, swap with last student if desirable for teacher
                else:
                    last_student = teacher_matching[teacher][-1]
                    if self.teacher_preferences[teacher][last_student] < self.teacher_preferences[teacher][student]:
                        student_matching[last_student] = -1 # Unmatch student to be swapped out
                        teacher_matching[teacher][-1] = student # Swap last accepted student with new student
                        student_matching[student] = teacher # Store matching of new student
                        teacher_matching[teacher] = self.sort_preferences(teacher_matching[teacher], self.teacher_preferences[teacher])
            exhausted = True
            all_matched = True
            # Check if students have exhausted all their proposals
            for i, pending in enumerate(students):
                if len(pending) > 0 and student_matching[i] == -1:
                    exhausted = False
            # Check if all students have been matched
            for matching in student_matching.values():
                if matching == -1:
                    all_matched = False
            # If more students than number of spots, keep running until all proposals made
            if self.n > self.m * capacity:
                condition = not exhausted
            # Else, keep running until everyone is matched
            else:
                condition = not all_matched
        self.student_matching = student_matching
    
    """
    Args:
        preferences: new preference ordering from each student for each teacher
    Returns:
        Updated preferences
    """
    def generateDigraph(self, preferences):
        # Add each DA match as a student,teacher node to TTC graph
        for student, teacher in self.student_matching.items():
            self.G.add_node((student, teacher))
        # Check for any consistent matching (i.e, self-edge) after preference change and remove matchings from graph
        for i, pref in enumerate(preferences):
            # If DA matching is the most preferred, keep matching and remove relevant node
            if self.student_matching[i] == pref[0]:
                self.G.remove_node((i, pref[0]))
        # Iterate through remaining nodes and create edge to target node
        for node in self.G.nodes():
            edged_added = False
            while not edged_added:
                top_preference = preferences[node[0]].pop(0)
                dest = self.findNode(top_preference, 1)
                if len(dest) != 0:
                    edged_added = True
                    random.shuffle(dest)
                    self.TTCpref[node[0]] = dest
                    popped = self.TTCpref[node[0]].pop(0)
                    self.G.add_edge(node, popped)
        return preferences

    """
    Args:
        G: graph
        index: index of desired agent
        type: 0 if student, 1 if teacher
    Returns:
        nodes in G corresponding to desired agent
    """
    def findNode(self, index, type):
        matches = []
        for node in self.G.nodes():
            if node[type] == index:
                matches.append(node)
        return matches

    """
    Args:
        G: graph
        preferences: updated preference orderings
    Returns:
        updated graph with traded nodes removed and edges redirected to next preferred
    """
    def redirect(self, preferences):
        for node in self.G.nodes():
            if self.G.out_degree(node) == 0:
                if len(self.TTCpref[node[0]]) != 0:
                    edge_added = False
                    while not edge_added:
                        target = preferences[node[0]].pop(0)
                        dest = self.findNode(target, 1)
                        if len(dest) != 0:
                            self.TTCpref[node[0]] = dest
                            edge_added = True
                popped = self.TTCpref[node[0]].pop(0)
                self.G.add_edge(node, popped)
        return preferences

    """
    Returns:
        a dictionary mapping each student to old/new class based on changed preferences
    """
    def TTC(self):
        done = False
        sorted_student_preferences = [self.sort_preferences([i for i in range(len(self.teacher_preferences))], preference) for preference in self.student_preferences]
        sorted_student_preferences = self.generateDigraph(sorted_student_preferences)
        # Keep running TTC until all students have been matched
        while not done:
            for node in self.G.nodes():
                assert(self.G.out_degree(node) == 1)
            # Each node should point to exactly one other node
            cycles_left = True
            # Iterate through all cycles at current iteration and trade along each cycle
            while cycles_left:
                try:
                    if nx.number_of_nodes(self.G) != 0:
                        cycles = list(nx.simple_cycles(self.G))
                        for c in cycles:
                            for i in range(len(c) - 1):
                                self.student_matching[c[i][0]] = c[i+1][1]
                                self.G.remove_node(c[i])
                            self.student_matching[c[len(c) - 1][0]] = c[0][1]
                            self.G.remove_node(c[len(c) - 1])
                        sorted_student_preferences = self.redirect(sorted_student_preferences)
                    else:
                        cycles_left = False
                except:
                    if nx.number_of_nodes(self.G) != 0:
                        self.G.clear()
                    cycles_left = False
            # Update graph after trades
            done = nx.number_of_nodes(self.G) == 0 
        return