import numpy as np
import functions
import yakopcic_model


class Experiment:
    def __init__(self, sim_args, model, memristor_args):
        self.name = None
        self.dt = sim_args["dt"]

        self.simulation = {"x0": sim_args["x0"], "dt": self.dt}
        self.memristor_args = memristor_args
        self.memristor_args.update({"x0": sim_args["x0"]})
        self.memristor = model(**self.memristor_args)
        # self.memristor.print()

        self.functions = {
            "dxdt": self.memristor.dxdt,
            "I": self.memristor.I,
        }

        # print("Simulation:")
        # print(f"\tTime range [ {self.t_min}, {self.t_max} ]")
        # print(f"\tSamples {self.simulation['N']}")
        # print(f"\tInitial value of state variable {self.simulation['x0']}")


class OldYakopcic(Experiment):  # Parameter setup for the old model in old_experiment.py.
    def __init__(self):
        super(OldYakopcic, self).__init__(
            sim_args={"dt": 0.001, "x0": 0.0},
            model=yakopcic_model.YakopcicNew,
            memristor_args={
                "Ap": 9,
                "An": 1,
                "Vp": 0.5,
                "Vn": 0.5,
                "alphap": 1,
                "alphan": 1,
                "xp": 0.1,
                "xn": 0.242,
                "eta": 1
            },
        )
        self.name = "oldSET"


class NewYakopcic(Experiment):  # Parameter setup for the new model in new_experiment.py.
    def __init__(self):
        super(NewYakopcic, self).__init__(
            sim_args={"dt": 0.001, "x0": 0.0},
            model=yakopcic_model.YakopcicNew,
            memristor_args={
                "gmax_p": 0.0004338454236,
                "bmax_p": 4.988561168,
                "gmax_n": 8.44e-6,
                "bmax_n": 6.272960721,
                "gmin_p": 0.03135053798,
                "bmin_p": 0.002125127287,
                "gmin_n": 1.45e-05,
                "bmin_n": 3.295533935,
                "Ap": 5.9214,
                "An": 2.2206,
                "Vp": 0,
                "Vn": 0,
                "xp": 0.11,
                "xn": 0.1433673316,
                "alphap": 9.2,
                "alphan": 0.7013461469,
                "eta": 1
            },
        )
        self.name = "NewSET"
