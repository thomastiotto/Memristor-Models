Memristor_Thomas = {
    # internal state parameters
    "Ap": 90,
    "An": 10,
    "Vp": 0.5,
    "Vn": 0.5,
    "alphap": 1,
    "alphan": 1,
    "xp": 0.1,
    "xn": 0.242,
    "eta": 1,
    # electron transfer parameters
    "gmax_p": 9e-5,
    "bmax_p": 4.96,
    "gmax_n": 1.7e-4,
    "bmax_n": 3.23,
    "gmin_p": 1.5e-5,
    "bmin_p": 6.91,
    "gmin_n": 4.4e-7,
    "bmin_n": 2.6,
    # simulation parameters
    "dt": 0.0001,
    "x0": 0.0
}

Memristor_Alina = {
    # internal state parameters
    "Ap": 0.071,
    "An": 0.02662694665,
    "Vp": 0,
    "Vn": 0,
    "xp": 0.11,
    "xn": 0.1433673316,
    "alphap": 9.2,
    "alphan": 0.7013461469,
    "eta": 1,
    # electron transfer parameters
    "gmax_p": 0.0004338454236,
    "bmax_p": 4.988561168,
    "gmax_n": 8.44e-6,
    "bmax_n": 6.272960721,
    "gmin_p": 0.03135053798,
    "bmin_p": 0.002125127287,
    "gmin_n": 1.45e-05,
    "bmin_n": 3.295533935,
    # simulation parameters
    "dt": 0.0843,
    "x0": 0.0
}

dt_data = 0.0843  # Timestep Alina's model is fitted to

dt_thomas = 1 / 10000
dt_ratio = dt_data / Memristor_Thomas["dt"]
# Adjust the amplitude parameters to the timescale
Memristor_Thomas["Ap"] = Memristor_Thomas["Ap"] / dt_ratio
Memristor_Thomas["An"] = Memristor_Thomas["An"] / dt_ratio
print('Thomas Ap', Memristor_Thomas["Ap"])
print('Thomas An', Memristor_Thomas["An"])

dt_nengo = 1e-3  # Nengo timestep
# Adjust amplitudes
# Memristor_Alina["Ap"]=Memristor_Alina["Ap"]/dt_ratio
# Memristor_Alina["An"]=Memristor_Alina["An"]/dt_ratio
print('Alina Ap', Memristor_Alina["Ap"])
print('Alina An', Memristor_Alina["An"])

Memristor_Alina["dt"] = dt_nengo

input_pulses=""".001 120 .001 .01 1 0 1
.001 1 .001 .4 -2 -.1 10
.001 1 .001 .4 .6 -.1 10"""
input_iv="""10 0.00 10 0.00 1 0 1
15 0.00 15 0.00 -2 0 1"""

