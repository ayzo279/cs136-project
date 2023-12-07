import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

class CourseMechanism:
    def __init__(self, n, m, var1=1, var2=1):
        self.n = n # number of students
        self.m = m # number of teachers

        self.var1 = var1 # determines how correlated first round preferences are across students
        self.var2 = var2 # determines how correlated second round preferences are with first round

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
        self.teacher_preferences = [{i:np.random.uniform(0.0,10.0) for i in range(self.n)} for _ in range(self.m)]
        # Sample student preferences from multivariate normal distribution
        means = np.linspace(0.0, 10.0, self.m)
        # np.random.uniform(0.0,10.0,self.m)
        cov = np.diag(np.array([self.var1] * self.m))
        for _ in range(self.n):
            sample_prefs = np.random.multivariate_normal(means, cov)
            self.student_preferences.append({j:sample_prefs[j] for j in range(self.m)})
    
    """
    Args:
        bump: True if one class becomes popular during shopping week
    Returns:
        List of preference values for each student for each teacher
    """
    def resample_preferences(self, bump=False):
        for i, student in enumerate(self.student_preferences):
            means = np.array(list(student.values()))
            if bump and i > 0:
                means[0] *= 2
            cov = np.diag(np.array([self.var2] * self.m))
            sample_prefs = np.random.multivariate_normal(means, cov)
            self.student_preferences[i] = {j:sample_prefs[j] for j in range(self.m)}
            # if bump and i > 0:
            #     self.student_preferences[i][0] += np.random.uniform(0, 10)

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

def enumerate_reassign(match1, match2):
    reassigned = 0
    for i in range(len(match1)):
        if match1[i] != match2[i]:
            reassigned += 1
    return reassigned

def reassign_test(epochs=200, var1=1, var2=1):
    total = 0
    teacher_pref = []
    for i in range(epochs):
        test = CourseMechanism(100, 20, var1=var1, var2=var2)
        test.generate_preferences()
        if i == 0:
            teacher_pref = test.teacher_preferences
        test.teacher_preferences = teacher_pref
        test.studentDA(10)
        match1 = test.student_matching.copy()
        # test.resample_preferences(bump=False)
        test.TTC()
        match2 = test.student_matching.copy()
        total += enumerate_reassign(match1, match2)
    return total/epochs

def var1_test(epochs=10):
    var1 = np.arange(1, 11)
    prob_success = []
    for v1 in tqdm(var1):
        success = 0
        for _ in range(epochs):
            test = CourseMechanism(100,20, var1=v1)
            test.generate_preferences()
            print("Running Student DA...")
            test.studentDA(10)
            test.resample_preferences(bump=False)
            print("Running TTC...")
            test.TTC()
            success += test.prob_success()
        prob_success.append(success / epochs)
    
    plt.plot(var1, prob_success)
    plt.xlabel("var1")
    plt.ylabel("Probability that student gets their favorite class")
    plt.show()

def var2_test(epochs=10):
    var2 = np.arange(1, 11)
    prob_success = []
    for v2 in tqdm(var2):
        success = 0
        for _ in range(epochs):
            test = CourseMechanism(100,10, var2=v2)
            test.generate_preferences()
            test.studentDA(10)
            test.resample_preferences(bump=True)
            test.TTC()
            success += test.prob_success()
        prob_success.append(success / epochs)
    
    plt.plot(var2, prob_success)
    plt.xlabel("var2")
    plt.ylabel("Probability that student gets their favorite class")
    plt.show()
    

def deviate_test(epochs = 100, var1=1, var2=1):
    nodev_success = 0
    dev_success = 0

    for _ in range(epochs):
        test = CourseMechanism(100,20,var1=var1, var2=var2)

        test.generate_preferences()
        student_prefs = test.student_preferences.copy()
        test.studentDA(10, deviate=False)
        test.resample_preferences(bump=False)
        resample_prefs = test.student_preferences.copy()
        test.TTC()
        nodev_success += test.prob_success(single=True)
        
        test.student_preferences = student_prefs
        test.studentDA(10, deviate=True)
        test.student_preferences = resample_prefs
        test.TTC()
        dev_success += test.prob_success(single=True)
    
    # print(f"No deviation: {nodev_success/epochs}")
    # print(f"Deviation: {dev_success/epochs}")
    return dev_success/nodev_success
    print(f"Usefulness ratio: {dev_success/nodev_success}, Difference:{dev_success/epochs - nodev_success/epochs}")





def var_test(v1, v2, epochs=100):
    success = 0
    for _ in range(epochs):
        test = CourseMechanism(100, 10, var1=v1, var2=v2)
        test.generate_preferences()
        test.studentDA(10)
        test.resample_preferences(bump=True)
        test.TTC()
        success += test.prob_success()
    return success / epochs



def main():
    test = CourseMechanism(100, 20)
    test.generate_preferences()
    test.studentDA(10)
    print("Running Student DA...")
    # print(test.print_student_preferences())
    print(test.student_matching)
    # print(test.prob_success())
    
    test.resample_preferences(bump=True)
    test.TTC()
    print("Running TTC...")
    # print(test.print_student_preferences())
    print(test.student_matching)
    # print(test.prob_success())

if __name__ == "__main__":
    # main()
    # var1_test()
    # var2_test()
    # deviate_test(var1=1, var2=1)
    mat = np.zeros((10,10))
    count = 0
    for i, var1 in enumerate(range(1,101, 10)):
        for j, var2 in enumerate(range(1, 101, 10)):
            count += 1
            print(count)
            mat[j][i] = reassign_test(var1 = var1, var2 = var2)
    
    # mat = np.zeros((10,10))
    # for i, var1 in enumerate(range(1,101, 10)):
    #     for j, var2 in enumerate(range(1, 101, 10)):
    #         print(f"{i * j + j}/{100}")
    #         mat[j][i] = deviate_test(var1 = var1, var2 = var2)
        
    matplot = plt.matshow(mat)
    plt.xlabel("var1")
    plt.ylabel("var2")
    plt.colorbar(matplot)
    plt.show()
    # deviate_test()
    

    # # Variance test for entire population
    # var1 = (1,10)
    # var2 = (1,10) 
    # print(f"v1={var1[0]}, v2={var2[0]}: {var_test(v1=var1[0], v2=var2[0])}")
    # print(f"v1={var1[1]}, v2={var2[0]}: {var_test(v1=var1[1], v2=var2[0])}")
    # print(f"v1={var1[0]}, v2={var2[1]}: {var_test(v1=var1[0], v2=var2[1])}")
    # print(f"v1={var1[1]}, v2={var2[1]}: {var_test(v1=var1[1], v2=var2[1])}")

    # Variance test for an agent that deviates / does not deviate
    # TODO
            


