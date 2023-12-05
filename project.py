import numpy as np

class CourseMechanism:
    def __init__(self):
        self.teacher_preferences = {}
        self.student_preferences = {}

    """
    Args:
        n: number of students
        m: number of teachers
    """
    def generate_preferences(self, n, m):
        self.teacher_preferences = [{i:np.random.uniform(0.0,10.0) for i in range(n)} for _ in range(m)]
        self.student_preferences = [{j:np.random.uniform(0.0,10.0) for j in range(m)} for _ in range(n)]

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
        while not at_capacity and not exhausted:
            to_check = {} # Dict of teachers to check for each n student, -1 implies no proposal was made this round
            # Make proposals for unmatched students
            for i, ranking in enumerate(students):
                if student_matching[i] == -1:
                    proposal = ranking.pop(0)
                    to_check[i] = proposal
            # Iterate and evaluate through each proposal
            for student, teacher in to_check.items():
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
            # Check if all class spots have been filled
            for matching in teacher_matching.values():
                if len(matching) < capacity:
                    at_capacity = False
            # Check if students have exhausted all their proposals
            for pending in students:
                if len(pending) > 0:
                    exhausted = False
        return student_matching

test = CourseMechanism()
test.generate_preferences(3,3)
print(test.student_preferences)
print(test.teacher_preferences)
print(test.studentDA(1))


                    
            


