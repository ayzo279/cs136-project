import numpy as np
import networkx as nx

class CourseMechanism:
    def __init__(self):
        self.teacher_preferences = {}
        self.student_preferences = {}
        self.student_matching = {}
        self.var1 = 1
    """
    Args:
        n: number of students
        m: number of teachers
    Returns:
        List of randomized preference values for each student/teacher for each teacher/student
    """
    def generate_preferences(self, n, m):
        # Sample teacher preferences randomly
        self.teacher_preferences = [{i:np.random.uniform(0.0,10.0) for i in range(n)} for _ in range(m)]

        # Sample student preferences from multivariate normal distribution
        means = np.random.uniform(0.0,10.0,m)
        cov = np.diag(np.array([self.var1] * m))
        sample_prefs = np.random.multivariate_normal(means, cov)
        self.student_preferences = [{j:sample_prefs[j] for j in range(m)} for _ in range(n)]

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
    Args:
        students: list of ranked teacher preferences [[t_1, ..., t_m], ... ]
        teachers: list of ranked student preferences [[s_1, ..., s_n], ...]
    Returns:
        a dictionary mapping each student to their accepted class
    """

    def studentDA(self, capacity):
        students = [self.sort_preferences([i for i in range(len(self.teacher_preferences))], preference) for preference in self.student_preferences]
        teacher_matching = {i:[] for i in range(len(self.teacher_preferences))}
        student_matching = {j:-1 for j in range(len(students))} # -1 implies unmatched
        at_capacity = False
        exhausted = False
        all_matched = False
        while not at_capacity and not exhausted and not all_matched:
            to_check = {} # Dict of teachers to check for each n student, -1 implies no proposal was made this round
            # Make proposals for unmatched students
            for i, ranking in enumerate(students):
                if student_matching[i] == -1:
                    proposal = ranking.pop(0)
                    to_check[i] = proposal
            # Iterate and evaluate through each proposal
            for student, teacher in to_check.items():
                # Just accept proposal if not at capacity
                if len(teacher_matching[teacher]) < capacity: 
                    teacher_matching[teacher].append(student)
                    student_matching[student] = teacher 
                    self.sort_preferences(teacher_matching[teacher], self.teacher_preferences[teacher])
                else:
                    if self.teacher_preferences[teacher][teacher_matching[teacher][-1]] < self.teacher_preferences[teacher][student]:
                        student_matching[teacher_matching[teacher][-1]] = -1 # Unmatch student to be swapped out
                        teacher_matching[teacher][-1] = student # Swap last accepted student with new student
                        student_matching[student] = teacher # Store matching of new student
            at_capacity = True
            exhausted = True
            all_matched = True
            # Check if all class spots have been filled
            for matching in teacher_matching.values():
                if len(matching) < capacity:
                    at_capacity = False
            # Check if students have exhausted all their proposals
            for pending in students:
                if len(pending) > 0:
                    exhausted = False
            # Check if all students have been matched
            for matching in student_matching.values():
                if matching == -1:
                    all_matched = False
        self.student_matching = student_matching
    
    """
    Args:
        preferences: new preference ordering from each student for each teacher
    Returns:
        initial directed graph to run TTC
    """
    def generateDigraph(self, preferences):
        G = nx.DiGraph()
        for student, teacher in self.student_matching.items():
            G.add_node((student, teacher))
        for i, student in enumerate(preferences):
            top_preference = student.pop(0)
            src = self.findNode(G, i, 0)
            dest = self.findNode(G, top_preference, 1)
            G.add_edge(src, dest)
        return G

    """
    Args:
        G: graph
        index: index of desired agent
        type: 0 if student, 1 if teacher
    Returns:
        node in G corresponding to desired agent
    """
    def findNode(self, G, index, type):
        for node in G.nodes():
            if node[type] == index:
                return node
        return None

    """
    Args:
        G: graph
        preferences: updated preference orderings
    Returns:
        updated graph with traded nodes removed and edges redirected to next preferred
    """
    def redirect(self, G, preferences):
        for node in G.nodes():
            while G.out_degree(node) == 0:
                target = preferences[node[0]].pop(0)
                dest = self.findNode(G, target, 1)
                if dest is not None:
                    G.add_edge(node, dest)
        return G

    """
    Returns:
        a dictionary mapping each student to old/new class based on changed preferences
    """
    def TTC(self):
        new_matching = {i:-1 for i in range(len(self.student_matching))}
        done = False
        sorted_student_preferences = [self.sort_preferences([i for i in range(len(self.teacher_preferences))], preference) for preference in self.student_preferences]
        G = self.generateDigraph(sorted_student_preferences)
        # Keep running TTC until all students have been matched
        while not done:
            # Each node should point to exactly one other node
            for node in G.nodes():
                assert(G.out_degree(node) == 1)
            cycles_left = True
            # Iterate through all cycles at current iteration and trade along each cycle
            while cycles_left:
                try:
                    c = nx.find_cycle(G)
                    # self-edge cycle, keep original matching
                    if len(c) == 1:
                        new_matching[c[0][0][0]] = c[0][0][1]
                        G.remove_node(c[0][0])
                    # Multi-agent trading, trade along cycle
                    else:
                        c = [src for src, _ in c]
                        for i in range(len(c) - 1):
                            new_matching[c[i][0]] = c[i+1][1]
                            G.remove_node(c[i])
                        new_matching[c[len(c) -1][0]] = c[0][1]
                        G.remove_node(c[len(c) - 1])
                except:
                    cycles_left = False
            # Update graph after trades
            G = self.redirect(G, sorted_student_preferences)
            done = True
            for matching in new_matching.values():
                if matching == -1:
                    done = False
        self.student_matching = new_matching
        return

test = CourseMechanism()
print("Running Student DA...")
test.generate_preferences(3,3)
print(test.student_preferences)
test.studentDA(1)
print(test.student_matching)
print("Running TTC...")
test.generate_preferences(3,3)
print(test.student_preferences)
test.TTC()
print(test.student_matching)


                    
            


