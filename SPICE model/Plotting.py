import pandas as pd
import matplotlib.pyplot as plt


# print(data)
def remove_duplicates(x):
    return list(dict.fromkeys(x))


def plot_function():
    avg_resistance = []
    data = pd.read_csv("./outputs/07v.txt", sep="	")
    t = data["time"]
    r = data["V(pre)/Ix(U1:TE)"]
    for i in range(len(r)-1):
        if r[i] == r[i+1]:
            avg_resistance.append(r[i])
    r = remove_duplicates(avg_resistance)
    plt.plot(range(0, len(r)), r, "o")
    plt.yscale("log")
    plt.show()

def average_resistances(r):
    avg_resistance = []
    for i in range(len(r)-1):
        if r[i] == r[i+1]:
            avg_resistance.append((r[i]*10))
    r = remove_duplicates(avg_resistance)
    return r
#plot_function()

data1 = pd.read_csv("./outputs/05v.txt", sep="	")
r1 = average_resistances(data1["R1"])
r2 = average_resistances(data1["R2"])
r3 = average_resistances(data1["R3"])
r4 = average_resistances(data1["R4"])
r5 = data1["R4"]
plt.plot(range(0, len(r1)), r1, "o", range(0, len(r2)), r2, "o", range(0, len(r3)), r3, "o", range(0, len(r4)), r4, "o")
plt.yscale("log")
plt.legend(["0.5V", "0.6V", "0.7V", "1V"])
plt.show()
