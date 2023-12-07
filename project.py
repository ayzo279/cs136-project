from mechanism import CourseMechanism
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

STUDENTS = 100
TEACHERS = 15
CAPACITY = 10
EPOCHS = 200

def deviate_test(epochs = EPOCHS, var1=1, var2=1, alpha=5):
    nodev_success = 0
    dev_success = 0

    for _ in range(epochs):
        test = CourseMechanism(STUDENTS, TEACHERS, var1=var1, var2=var2, alpha=alpha)

        test.generate_preferences()
        student_prefs = test.student_preferences.copy()
        test.studentDA(CAPACITY, deviate=False)
        test.resample_preferences(bump=True)
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
    reassigned = 0
    for i in range(len(match1)):
        if match1[i] != match2[i]:
            reassigned += 1
    return reassigned

def reassign_test(epochs=200, var1=1, var2=1):
    total = 0
    student_pref = []
    match1 = {}
    for i in range(epochs):
        if i == 0:
            test = CourseMechanism(100, 20, var1=var1, var2=var2)
            test.generate_preferences()
            student_pref = test.student_preferences.copy()
            test.studentDA(10)
            match1 = test.student_matching.copy()
        test.student_preferences = student_pref.copy()
        test.student_matching = match1.copy()
        test.resample_preferences(bump=False)
        test.TTC()
        match2 = test.student_matching
        total += enumerate_reassign(match1, match2)
    return total/epochs

def exp_reassign(epochs=400):
    preferences = []
    test = CourseMechanism(100, 20)
    matchings = []
    # for _ in range(epochs):
    test.generate_preferences()
    preferences = test.student_preferences.copy()
    test.studentDA(10)
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
    # var_range = range(1, 111, 10)
    # lst = []
    # for var1 in tqdm(var_range):
    #     lst.append(deviate_test(var1=var1))
    
    # plt.plot(var_range, lst, color="forestgreen")
    # plt.title("Usefulness of Deviation")
    # plt.xlabel(r"$\sigma_1^2$")
    # plt.ylabel("Usefulness")
    # plt.tight_layout()
    # plt.show()


    alpha_range = np.linspace(0.0, 2.0, 20)
    lst = []
    for a in tqdm(alpha_range):
        lst.append(deviate_test(alpha=a))
    
    plt.plot(alpha_range, lst, color="forestgreen")
    plt.title("Usefulness of Deviation")
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Usefulness")
    plt.tight_layout()
    plt.show()

    # mat = np.zeros((11,11))
    # for i, var1 in enumerate(tqdm(var_range)):
    #     for j, var2 in enumerate(var_range):
    #         mat[j][i] = deviate_test(var1=var1, var2=var2)

    # matplot = plt.matshow(mat, cmap="summer_r")
    # plt.xlabel(r"$\sigma_1^2$")
    # plt.ylabel(r"$\sigma_2^2$")
    # plt.title("Usefulness of Deviation")
    # plt.colorbar(matplot)
    # ax = plt.gca()
    # ax.xaxis.set_ticks_position('bottom')  # Set tick position
    # ax.xaxis.set_label_position('bottom')  # Set label position
    # ax.invert_yaxis()
    # plt.xticks(np.arange(0, 11, 1), var_range)
    # plt.yticks(np.arange(0, 11, 1), var_range)
    # plt.show()

    # matplot = plt.matshow((mat > 1.0), cmap="Greens")
    # plt.xlabel(r"$\sigma_1^2$")
    # plt.ylabel(r"$\sigma_2^2$")
    # plt.title("Usefulness of Deviation > 1")
    # ax = plt.gca()
    # ax.xaxis.set_ticks_position('bottom')  # Set tick position
    # ax.xaxis.set_label_position('bottom')  # Set label position
    # ax.invert_yaxis()
    # plt.xticks(np.arange(0, 11, 1), var_range)
    # plt.yticks(np.arange(0, 11, 1), var_range)
    # plt.show()



    # main()
    # var1_test()
    # var2_test()
    # deviate_test(var1=1, var2=1)
    # mat = np.zeros((10,10))
    # count = 0
    # for i, var1 in enumerate(range(1,101, 10)):
    #     for j, var2 in enumerate(range(1, 101, 10)):
    #         count += 1
    #         print(count)
    #         mat[j][i] = reassign_test(var1 = var1, var2 = var2)
    # exp_reassign()

    
    # mat = np.zeros((10,10))
    # for i, var1 in enumerate(range(1,101, 10)):
    #     for j, var2 in enumerate(range(1, 101, 10)):
    #         print(f"{i * j + j}/{100}")
    #         mat[j][i] = deviate_test(var1 = var1, var2 = var2)
        
    # matplot = plt.matshow(mat)
    # plt.xlabel("var1")
    # plt.ylabel("var2")
    # plt.colorbar(matplot)
    # plt.show()
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
            


