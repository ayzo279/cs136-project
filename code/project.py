from mechanism import CourseMechanism
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

STUDENTS = 100
TEACHERS = 15
CAPACITY = 10
EPOCHS = 500

def success_test(epochs = EPOCHS, var1=1, var2=1):
    success = 0
    for _ in range(epochs):
        test = CourseMechanism(STUDENTS, TEACHERS, var1=var1, var2=var2)
        test.generate_preferences()
        test.studentDA(CAPACITY, deviate=False)
        test.resample_preferences(bump=False)
        test.TTC()
        success += test.prob_success(single = True)
    return success/EPOCHS

def deviate_test(epochs = EPOCHS, var1=1, var2=1, alpha=1, p=1):
    """
    Calculate the usefuleness of deviation with different parameters
    """
    
    nodev_success = 0
    dev_success = 0

    for _ in range(epochs):
        test = CourseMechanism(STUDENTS, TEACHERS, var1=var1, var2=var2, alpha=alpha)

        test.generate_preferences()
        student_prefs = test.student_preferences.copy()
        test.studentDA(CAPACITY, deviate=False)
        test.resample_preferences(bump=True, p=p)
        resample_prefs = test.student_preferences.copy()
        test.TTC()
        nodev_success += test.prob_success(single=True)
        
        test.student_preferences = student_prefs.copy()
        test.studentDA(CAPACITY, deviate=True)
        test.student_preferences = resample_prefs.copy()
        test.TTC()
        dev_success += test.prob_success(single=True)
    
    # print(f"No deviation: {nodev_success/epochs}")
    # print(f"Deviation: {dev_success/epochs}")
    # print(f"Usefulness ratio: {dev_success/nodev_success}, Difference:{dev_success/epochs - nodev_success/epochs}")
    return dev_success/nodev_success


def enumerate_reassign(match1, match2):
    """
    Helper function: calculates the number of students reassigned 
    given a pair of matchings (from round 1 and round 2)
    """
    reassigned = 0
    for i in range(len(match1)):
        if match1[i] != match2[i]:
            reassigned += 1
    return reassigned


def exp_reassign(epochs=EPOCHS):
    """
    Calculate the number of students reassigned from round 1 to round 2
    """
    preferences = []
    test = CourseMechanism(STUDENTS, TEACHERS)
    matchings = []
    # for _ in range(epochs):
    test.generate_preferences()
    preferences = test.student_preferences.copy()
    test.studentDA(CAPACITY)
    matchings = test.student_matching.copy()
    avg = []
    for i in range(1, 101, 10):
        num = 0
        for j in range(epochs):
            test.var2 = i
            test.student_matching = matchings.copy()
            test.student_preferences = preferences.copy()
            test.resample_preferences()
            test.TTC()
            num += enumerate_reassign(matchings, test.student_matching)
        avg.append(num / epochs)
    plt.plot(range(1,101,10), avg)
    plt.show()


def main():
    """
    Run a toy example
    """
    test = CourseMechanism(3, 6)
    test.generate_preferences()
    test.studentDA(3)
    print("Running Student DA...")
    print(test.print_student_preferences())
    print(test.student_matching)
    print(test.prob_success())
    
    test.resample_preferences(bump=True)
    test.TTC()
    print("Running TTC...")
    print(test.print_student_preferences())
    print(test.student_matching)
    print(test.prob_success())


if __name__ == "__main__":
    # Experiments and plots
    var_range = range(1, 111, 10)
    alpha_range = np.linspace(0.0, 2.0, 11)
    p_range =np.linspace(0.0, 1.0, 10)

    # Varying var1
    lst = []
    for var1 in tqdm(var_range):
        lst.append(deviate_test(var1=var1))
    
    plt.plot(var_range, lst, color="forestgreen")
    plt.title("Usefulness of Deviation")
    plt.xlabel(r"$\sigma_1^2$")
    plt.ylabel("Usefulness")
    plt.tight_layout()
    plt.show()

    # Varying var2
    lst = []
    for var2 in tqdm(var_range):
        lst.append(deviate_test(var2=var2))
    
    plt.plot(var_range, lst, color="forestgreen")
    plt.title("Usefulness of Deviation")
    plt.xlabel(r"$\sigma_2^2$")
    plt.ylabel("Usefulness")
    plt.tight_layout()
    plt.show()

    # Varying alpha
    lst = []
    for a in tqdm(alpha_range):
        lst.append(deviate_test(alpha=a))
    
    plt.plot(alpha_range, lst, color="forestgreen")
    plt.title("Usefulness of Deviation")
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Usefulness")
    plt.tight_layout()
    plt.show()

    # Heat maps of usefulness, varying alpha and var1 (given pre-set var2)
    v2=1
    mat = np.zeros((11,11))
    for i, v1 in enumerate(tqdm(var_range)):
        for j, a in enumerate(alpha_range):
            mat[j][i] = deviate_test(var1=v1, var2=v2, alpha=a)

    matplot = plt.matshow(mat, cmap="RdYlGn", vmin=0.3, vmax=1.7)
    plt.xlabel(r"$\sigma_1^2$")
    plt.ylabel(r"$\alpha$")
    plt.title("Usefulness of Deviation")
    plt.colorbar(matplot)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')  # Set tick position
    ax.xaxis.set_label_position('bottom')  # Set label position
    ax.invert_yaxis()
    plt.xticks(np.arange(0, 11, 1), var_range)
    plt.yticks(np.arange(0, 11, 1), np.round(alpha_range,1))
    plt.show()

    # Heat maps of usefulness, varying alpha and p (given pre-set var2)
    mat = np.zeros((10,11))
    for i, a in enumerate(tqdm(alpha_range)):
        for j, p in enumerate(p_range):
            mat[j][i] = deviate_test(alpha=a, p=p)

    matplot = plt.matshow(mat, cmap="RdYlGn")
    plt.xlabel(r"$\alpha$")
    plt.ylabel("p")
    plt.title("Usefulness")
    plt.xticks(np.arange(0, 11, 1), np.round(alpha_range, 1))
    plt.yticks(np.arange(0, 10, 1),  np.round(p_range, 1))
    plt.colorbar(matplot)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')  # Set tick position
    ax.xaxis.set_label_position('bottom')  # Set label position
    ax.invert_yaxis()
    plt.show()
            

