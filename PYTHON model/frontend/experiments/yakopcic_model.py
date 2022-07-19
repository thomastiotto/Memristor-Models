import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp


def mim_iv(v, g, b):
    return g * np.sinh(b * v)


def mim_mim_iv(v, gp, bp, gn, bn):
    return np.where(v >= 0, mim_iv(v, gp, bp), mim_iv(v, gn, bn))


def euler_step(x, t, f, dt, v, args):
    return x + f(t, v, x, *args) * dt


class YakopcicNew:
    def __init__(self, **kwargs):
        self.type = "Yakopcic new model"
        self.x0 = kwargs["x0"] if "x0" in kwargs else 0.1

        self.passed_parameters = kwargs
        self.passed_parameters.pop("x0")

        self.gmin = kwargs["gmin"] if "gmin" in kwargs else 0.17
        self.gmax = kwargs["gmax"] if "gmax" in kwargs else 0.17
        self.bmin = kwargs["bmin"] if "bmin" in kwargs else 0.05
        self.bmax = kwargs["bmax"] if "bmax" in kwargs else 0.05
        self.gmax_p = 9e-5
        self.bmax_p = 4.96
        self.gmax_n = 1.7e-4
        self.bmax_n = 3.23
        self.gmin_p = 1.5e-5
        self.bmin_p = 6.91
        self.gmin_n = 4.4e-7
        self.bmin_n = 2.6
        self.Ap = kwargs["Ap"] if "Ap" in kwargs else 4000
        self.An = kwargs["An"] if "An" in kwargs else 4000
        self.Vp = kwargs["Vp"] if "Vp" in kwargs else 0.16
        self.Vn = kwargs["Vn"] if "Vn" in kwargs else 0.15
        self.alphap = kwargs["alphap"] if "alphap" in kwargs else 1
        self.alphan = kwargs["alphan"] if "alphan" in kwargs else 5
        self.xp = kwargs["xp"] if "xp" in kwargs else 0.3
        self.xn = kwargs["xn"] if "xn" in kwargs else 0.5
        self.eta = kwargs["eta"] if "eta" in kwargs else 1

        self.h1 = kwargs["h1"] if "h1" in kwargs else mim_mim_iv
        self.h2 = kwargs["h2"] if "h2" in kwargs else mim_mim_iv

    def I(self, t, v, x, read=None, *args):
        """
        Function implementing the I-V relationship (memristance).

        The hyperbolic sine shape causes the device to have an increase in  conductivity beyond a certain voltage
        threshold.
        a1, a2 : These parameters closely relate to the thickness of the dielectric layer  in  a  memristor  device,
        as  more  electrons  can  tunnel  through a thinner barrier leading to an increase in conductivity.
        b : This parameter determines how much curvature is seen in the I-V  curve  relative  to  the  applied
        voltage.  This  relates  to  how  much of the conduction in the device is Ohmic and how much is due to the
        tunnel barrier.

        Parameters
        ----------
        t : time
        v: voltage
        x : state variable

        Attributes
        ----------
        a1 : amplitude of current in response to input voltage in forward bias (positive input voltage)
        a2 : amplitude of current in response to input voltage in reverse bias (negative input voltage)
        b :  used to control the intensity of the threshold function relating conductivity to input voltage  magnitude

        Returns
        -------
        I : current through the device at time t
        """
        read = False if read is None else read
        gmin = args[0] if len(args) > 0 else self.gmin
        gmax = args[1] if len(args) > 1 else self.gmax
        bmin = args[2] if len(args) > 2 else self.bmin
        bmax = args[3] if len(args) > 2 else self.bmax
        gmax_p = self.gmax_p
        bmax_p = self.bmax_p
        gmax_n = self.gmax_n
        bmax_n = self.bmax_n
        gmin_p = self.gmin_p
        bmin_p = self.bmin_p
        gmin_n = self.gmin_n
        bmin_n = self.bmin_n

        #v = self.V(t, read)
        i = self.h1(v, gmax_p, bmax_p, gmax_n, bmax_n) * x + self.h2(v, gmin_p, bmin_p, gmin_n, bmin_n) * (1 - x)

        return i

    def g(self, v, **kwargs):
        """
        Function implementing the threshold voltage that must be surpassed to induce a change in the value of the
        state variable.

        Provides the possibility of having different thresholds based on the polarity of the input voltage.
        The exponential value subtracted is a constant term during simulations and ensures that the value of the
        function g(V(t)) starts at 0 once either voltage threshold is surpassed.
        The magnitude of the exponential represents how quickly the state changes once the threshold
        is surpassed.

        Ap, An : These parameters control the speed of ion (or filament) motion.  This  could  be  related  to
        the dielectric  material  used  since  oxygen  vacancies  have  a  different  mobility  depending  which
        metal-oxide they are contained in.
        Vp, Vn : These parameters represent the threshold voltages.  These may  be related to the number of oxygen
        vacancies in a device in its initial state.  A device with more oxygen vacancies should have a larger  current
        draw  which  may  lead  to  a  lower  switching  threshold if switching is assumed to be based on the total
        charge applied.

        Parameters
        ----------
        v : Input voltage

        Attributes
        ----------
        Ap : Magnitude of exponential in forward bias (positive input voltage)
        An : Magnitude of exponential in reverse bias (negative input voltage)
        Vp : Positive voltage threshold
        Vn : Negative voltage threshold

        Returns
        -------
        g : The rate at which the state variable x changes
        """
        Ap = kwargs["Ap"] if "Ap" in kwargs else self.Ap
        An = kwargs["An"] if "An" in kwargs else self.An
        Vp = kwargs["Vp"] if "Vp" in kwargs else self.Vp
        Vn = kwargs["Vn"] if "Vn" in kwargs else self.Vn

        if v > Vp:
            return Ap * (np.exp(v) - np.exp(Vp))
        elif v < -Vn:
            return -An * (np.exp(-v) - np.exp(Vn))
        else:
            return 0

    def f(self, v, x, **kwargs):
        """
        Function used to divide the state variable motion into two different regions depending on the existing
        state of the device.  The state variable motion is constant up until the point xp or xn.  At this point the
        motion of the state variable is limited by an exponential function decaying with a rate of either alphap or
        alphan.
        The term eta was introduced to represent the direction of the motion of the
        state variable relative to the voltage polarity.  When eta=1, a positive voltage above the threshold will
        increase the value of the state variable, and when eta=–1, a positive voltage results in a decrease in state
        variable.

        alphap, alphan, xp, xn : These parameters determine where state variable motion is no longer linear,
        and they determine the degree to which state variable motion is dampened.  This could be related to the
        electrode metal used on either side of the dielectric film since the metals chosen may react to the oxygen
        vacancies differently.

        Parameters
        ----------
        v : Input voltage
        x : State variable

        Attributes
        ----------
        alphap : Exponential decay rate for when state variable x is increasing
        alphan : Exponential decay rate for when state variable x is decreasing
        xp : Positive threshold up till which x's motion is linear
        xn : Negative threshold up till which x's motion is linear

        Returns
        -------
        f : State variable x motion
        """
        xp = kwargs["xp"] if "xp" in kwargs else self.xp
        xn = kwargs["xn"] if "xn" in kwargs else self.xn
        eta = kwargs["eta"] if "eta" in kwargs else self.eta

        if eta * v >= 0:
            if x >= xp:
                return np.exp(-(x - xp)) * self.wp(x, **kwargs)
            else:
                return 1
        else:
            if x <= 1 - xn:
                return np.exp((x + xn - 1)) * self.wn(x, **kwargs)
            else:
                return 1

    def wp(self, x, **kwargs):
        """
        Windowing function ensuring that f(x)=0 when x(t)=1.

        Parameters
        ----------
        x : State variable

        Attributes
        ----------
        xp : Positive threshold up till which x's motion is linear

        Returns
        -------
        wp : Window for positive motion of state variable x
        """
        xp = kwargs["xp"] if "xp" in kwargs else self.xp

        return (xp - x) / (1 - xp) + 1

    def wn(self, x, **kwargs):
        """
        Windowing function ensuring that x(t)>=0 when the current  flow is reversed.

        Parameters
        ----------
        x : State variable

        Attributes
        ----------
        xn : Negative threshold up till which x's motion is linear

        Returns
        -------
        wn : Window for negative motion of state variable x
        """
        xn = kwargs["xn"] if "xn" in kwargs else self.xn

        return x / (1 - xn)

    def dxdt(self, t, v, x, *args, read=None):
        # Necessary to run curve_fit as it doesn't support passing named parameters
        read = False if read is None else read
        gmin = args[0] if len(args) > 0 else self.gmin
        gmax = args[1] if len(args) > 1 else self.gmax
        bmin = args[2] if len(args) > 2 else self.bmin
        bmax = args[3] if len(args) > 3 else self.bmax
        Ap = args[4] if len(args) > 4 else self.Ap
        An = args[5] if len(args) > 5 else self.An
        Vp = args[6] if len(args) > 6 else self.Vp
        Vn = args[7] if len(args) > 7 else self.Vn
        xp = args[8] if len(args) > 8 else self.xp
        xn = args[9] if len(args) > 9 else self.xn
        eta = args[10] if len(args) > 10 else self.eta

        # compile args into kwargs for other functions
        kwargs = {}
        for k in YakopcicNew.parameters():
            kwargs[k] = locals()[k]
        kwargs["eta"] = eta

        #v = self.V(t, read)
        return eta * self.g(v, **kwargs) * self.f(v, x, **kwargs)

    def fit(self):

        def ode_fitting(t, gmin, bmin, gmax, bmax, Ap, An, Vp, Vn, xp, xn):
            args = [gmin, bmin, gmax, bmax, Ap, An, Vp, Vn, xp, xn]
            print(args)
            sol = solve_ivp(self.dxdt, (t[0], t[-1]), [self.x0], method="LSODA",
                            t_eval=t,
                            args=args,
                            # p0=[0]
                            )

            return self.I(t, sol.y[0, :], *args)

        return ode_fitting

    def print(self):
        print(f"{self.type}:")
        self.print_equations()
        self.print_parameters()
        print("\tInput V:")
        self.input.print()

    def print_equations(self, start="\t"):
        start_lv2 = start + "\t"
        print(start, "Equations:")
        print(start_lv2, "I(t) = a_1 * x(t) * sinh(b * V(t)), V(t) >= 0")
        print(start_lv2, "I(t) = a_2 * x(t) * sinh(b * V(t)), V(t) < 0")
        print(start_lv2, "g(V(t)) = A_p * ( e^V(t) - e^V_p ), V(t) > V_p")
        print(start_lv2, "g(V(t)) = -A_n * ( e^-V(t) - e^V_n ), V(t) < -V_n")
        print(start_lv2, "g(V(t)) = 0,  -V_n <= V(t) <= V_p")
        print(start_lv2, "f(x) = e^( -( x - x_p ) ) * w_p(x, x_p), eta * V(t) >= 0, x >= X_p")
        print(start_lv2, "f(x) = 1, eta * V(t) > 0, x < X_p")
        print(start_lv2, "f(x) = e^( ( x + x_n - 1 ) ) * w_n(x, x_n), eta * V(t) < 0, x <= 1 - X_n")
        print(start_lv2, "f(x) = 1, eta * V(t) < 0, x <= 1 - X_n")

    def print_parameters(self, start="\t", simple=False):
        start_lv2 = start + "\t" if not simple else ""
        if not simple:
            print(start, "Parameters:")
            print(start_lv2, f"Speeds of ion motion A_p {self.Ap}, A_n {self.An}")
            print(start_lv2, f"Threshold voltages V_p {self.Vp}, V_n {self.Vn}")
            print(start_lv2, f"Linearity threshold for state variable motion x_p {self.xp}, x_n {self.xn}")
            print(start_lv2, f"Direction of state variable motion eta {self.eta}")
        else:
            return ([self.gmin, self.bmin, self.gmax, self.bmin, self.Ap, self.An, self.Vp, self.Vn, self.xp, self.xn,
                     self.eta])

    @staticmethod
    def parameters():
        return ["gmin", "bmin", "gmax", "bmax", "Ap", "An", "Vp", "Vn", "xp", "xn"]


class Yakopcic():
    def __init__(self, input, **kwargs):
        self.type = "Yakopcic model"

        self.input = input
        self.V = input
        self.x0 = kwargs["x0"] if "x0" in kwargs else 0.1

        self.passed_parameters = kwargs
        self.passed_parameters.pop("x0")

        self.a1 = kwargs["a1"] if "a1" in kwargs else 0.17
        self.a2 = kwargs["a2"] if "a2" in kwargs else 0.17
        self.b = kwargs["b"] if "b" in kwargs else 0.05
        self.Ap = kwargs["Ap"] if "Ap" in kwargs else 4000
        self.An = kwargs["An"] if "An" in kwargs else 4000
        self.Vp = kwargs["Vp"] if "Vp" in kwargs else 0.16
        self.Vn = kwargs["Vn"] if "Vn" in kwargs else 0.15
        self.alphap = kwargs["alphap"] if "alphap" in kwargs else 1
        self.alphan = kwargs["alphan"] if "alphan" in kwargs else 5
        self.xp = kwargs["xp"] if "xp" in kwargs else 0.3
        self.xn = kwargs["xn"] if "xn" in kwargs else 0.5
        self.eta = kwargs["eta"] if "eta" in kwargs else 1

    def I(self, t, v, x, *args, read=None):
        """
        Function implementing the I-V relationship (memristance).

        The hyperbolic sine shape causes the device to have an increase in  conductivity beyond a certain voltage
        threshold.
        a1, a2 : These parameters closely relate to the thickness of the dielectric layer  in  a  memristor  device,
        as  more  electrons  can  tunnel  through a thinner barrier leading to an increase in conductivity.
        b : This parameter determines how much curvature is seen in the I-V  curve  relative  to  the  applied
        voltage.  This  relates  to  how  much of the conduction in the device is Ohmic and how much is due to the
        tunnel barrier.

        Parameters
        ----------
        t : time
        v: voltage
        x : state variable

        Attributes
        ----------
        a1 : amplitude of current in response to input voltage in forward bias (positive input voltage)
        a2 : amplitude of current in response to input voltage in reverse bias (negative input voltage)
        b :  used to control the intensity of the threshold function relating conductivity to input voltage  magnitude

        Returns
        -------
        I : current through the device at time t
        """
        read = False if read is None else read
        a1 = args[0] if len(args) > 0 else self.a1
        a2 = args[1] if len(args) > 1 else self.a2
        b = args[2] if len(args) > 2 else self.b

        i = np.where(v >= 0, np.multiply(a1, x * np.sinh(b * v)), np.multiply(a2, x * np.sinh(b * v)))

        return i

    def g(self, v, **kwargs):
        """
        Function implementing the threshold voltage that must be surpassed to induce a change in the value of the
        state variable.

        Provides the possibility of having different thresholds based on the polarity of the input voltage.
        The exponential value subtracted is a constant term during simulations and ensures that the value of the
        function g(V(t)) starts at 0 once either voltage threshold is surpassed.
        The magnitude of the exponential represents how quickly the state changes once the threshold
        is surpassed.

        Ap, An : These parameters control the speed of ion (or filament) motion.  This  could  be  related  to
        the dielectric  material  used  since  oxygen  vacancies  have  a  different  mobility  depending  which
        metal-oxide they are contained in.
        Vp, Vn : These parameters represent the threshold voltages.  These may  be related to the number of oxygen
        vacancies in a device in its initial state.  A device with more oxygen vacancies should have a larger  current
        draw  which  may  lead  to  a  lower  switching  threshold if switching is assumed to be based on the total
        charge applied.

        Parameters
        ----------
        v : Input voltage

        Attributes
        ----------
        Ap : Magnitude of exponential in forward bias (positive input voltage)
        An : Magnitude of exponential in reverse bias (negative input voltage)
        Vp : Positive voltage threshold
        Vn : Negative voltage threshold

        Returns
        -------
        g : The rate at which the state variable x changes
        """
        Ap = kwargs["Ap"] if "Ap" in kwargs else self.Ap
        An = kwargs["An"] if "An" in kwargs else self.An
        Vp = kwargs["Vp"] if "Vp" in kwargs else self.Vp
        Vn = kwargs["Vn"] if "Vn" in kwargs else self.Vn

        if v > Vp:
            return Ap * (np.exp(v) - np.exp(Vp))
        elif v < -Vn:
            return -An * (np.exp(-v) - np.exp(Vn))
        else:
            return 0

    def f(self, v, x, **kwargs):
        """
        Function used to divide the state variable motion into two different regions depending on the existing
        state of the device.  The state variable motion is constant up until the point xp or xn.  At this point the
        motion of the state variable is limited by an exponential function decaying with a rate of either alphap or
        alphan.
        The term eta was introduced to represent the direction of the motion of the
        state variable relative to the voltage polarity.  When eta=1, a positive voltage above the threshold will
        increase the value of the state variable, and when eta=–1, a positive voltage results in a decrease in state
        variable.

        alphap, alphan, xp, xn : These parameters determine where state variable motion is no longer linear,
        and they determine the degree to which state variable motion is dampened.  This could be related to the
        electrode metal used on either side of the dielectric film since the metals chosen may react to the oxygen
        vacancies differently.

        Parameters
        ----------
        v : Input voltage
        x : State variable

        Attributes
        ----------
        alphap : Exponential decay rate for when state variable x is increasing
        alphan : Exponential decay rate for when state variable x is decreasing
        xp : Positive threshold up till which x's motion is linear
        xn : Negative threshold up till which x's motion is linear

        Returns
        -------
        f : State variable x motion
        """
        alphap = kwargs["alphap"] if "alphap" in kwargs else self.alphap
        alphan = kwargs["alphan"] if "alphan" in kwargs else self.alphan
        xp = kwargs["xp"] if "xp" in kwargs else self.xp
        xn = kwargs["xn"] if "xn" in kwargs else self.xn
        eta = kwargs["eta"] if "eta" in kwargs else self.eta

        if eta * v >= 0:
            if x >= xp:
                return np.exp(-alphap * (x - xp)) * self.wp(x, **kwargs)
            else:
                return 1
        else:
            if x <= 1 - xn:
                return np.exp(alphan * (x + xn - 1)) * self.wn(x, **kwargs)
            else:
                return 1

    def wp(self, x, **kwargs):
        """
        Windowing function ensuring that f(x)=0 when x(t)=1.

        Parameters
        ----------
        x : State variable

        Attributes
        ----------
        xp : Positive threshold up till which x's motion is linear

        Returns
        -------
        wp : Window for positive motion of state variable x
        """
        xp = kwargs["xp"] if "xp" in kwargs else self.xp

        return (xp - x) / (1 - xp) + 1

    def wn(self, x, **kwargs):
        """
        Windowing function ensuring that x(t)>=0 when the current  flow is reversed.

        Parameters
        ----------
        x : State variable

        Attributes
        ----------
        xn : Negative threshold up till which x's motion is linear

        Returns
        -------
        wn : Window for negative motion of state variable x
        """
        xn = kwargs["xn"] if "xn" in kwargs else self.xn

        return x / (1 - xn)

    def dxdt(self, t, v, x, *args, read=None):
        # Necessary to run curve_fit as it doesn't support passing named parameters
        read = False if read is None else read
        a1 = args[0] if len(args) > 0 else self.a1
        a2 = args[1] if len(args) > 1 else self.a2
        b = args[2] if len(args) > 2 else self.b
        Ap = args[3] if len(args) > 3 else self.Ap
        An = args[4] if len(args) > 4 else self.An
        Vp = args[5] if len(args) > 5 else self.Vp
        Vn = args[6] if len(args) > 6 else self.Vn
        alphap = args[7] if len(args) > 7 else self.alphap
        alphan = args[8] if len(args) > 8 else self.alphan
        xp = args[9] if len(args) > 9 else self.xp
        xn = args[10] if len(args) > 10 else self.xn
        eta = args[11] if len(args) > 11 else self.eta

        # compile args into kwargs for other functions
        kwargs = {}
        for k in Yakopcic.parameters():
            kwargs[k] = locals()[k]
        kwargs["eta"] = eta

        return eta * self.g(v, **kwargs) * self.f(v, x, **kwargs)

    def fit(self):

        def ode_fitting(t, a1, a2, b, Ap, An, Vp, Vn, alphap, alphan, xp, xn):
            args = [a1, a2, b, Ap, An, Vp, Vn, alphap, alphan, xp, xn]
            sol = solve_ivp(self.dxdt, (t[0], t[-1]), [self.x0], method="LSODA",
                            t_eval=t,
                            args=args,
                            # p0=[0]
                            )
            return self.I(t, sol.y[0, :], *args)

        return ode_fitting

    def print(self):
        print(f"{self.type}:")
        self.print_equations()
        self.print_parameters()
        print("\tInput V:")
        self.input.print()

    def print_equations(self, start="\t"):
        start_lv2 = start + "\t"
        print(start, "Equations:")
        print(start_lv2, "I(t) = a_1 * x(t) * sinh(b * V(t)), V(t) >= 0")
        print(start_lv2, "I(t) = a_2 * x(t) * sinh(b * V(t)), V(t) < 0")
        print(start_lv2, "g(V(t)) = A_p * ( e^V(t) - e^V_p ), V(t) > V_p")
        print(start_lv2, "g(V(t)) = -A_n * ( e^-V(t) - e^V_n ), V(t) < -V_n")
        print(start_lv2, "g(V(t)) = 0,  -V_n <= V(t) <= V_p")
        print(start_lv2, "f(x) = e^( -alpha_p * ( x - x_p ) ) * w_p(x, x_p), eta * V(t) >= 0, x >= X_p")
        print(start_lv2, "f(x) = 1, eta * V(t) > 0, x < X_p")
        print(start_lv2, "f(x) = e^( alpha_n * ( x + x_n - 1 ) ) * w_n(x, x_n), eta * V(t) < 0, x <= 1 - X_n")
        print(start_lv2, "f(x) = 1, eta * V(t) < 0, x <= 1 - X_n")

    def print_parameters(self, start="\t", simple=False):
        start_lv2 = start + "\t" if not simple else ""
        if not simple:
            print(start, "Parameters:")
            print(start_lv2, f"Dielectric layer thicknesses a_1 {self.a1}, a_2 {self.a2}")
            print(start_lv2, f"Curvature in I-V curve b {self.b}")
            print(start_lv2, f"Speeds of ion motion A_p {self.Ap}, A_n {self.An}")
            print(start_lv2, f"Threshold voltages V_p {self.Vp}, V_n {self.Vn}")
            print(start_lv2, f"Linearity threshold for state variable motion x_p {self.xp}, x_n {self.xn}")
            print(start_lv2, f"Dampening of state variable motion alpha_p {self.alphap}, alpha_n {self.alphan}")
            print(start_lv2, f"Direction of state variable motion eta {self.eta}")
        else:
            return ([self.a1, self.a2, self.b, self.Ap, self.An, self.Vp, self.Vn, self.alphap, self.alphan,
                     self.xp, self.xn, self.eta])

    @staticmethod
    def parameters():
        return ["a1", "a2", "b", "Ap", "An", "Vp", "Vn", "alphap", "alphan", "xp", "xn"]