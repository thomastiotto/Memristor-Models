import numpy as np


class Yakopcic():
    def __init__(self, input, window_function, **kwargs):
        self.type = "Yakopcic model"

        self.input = input
        self.window_function = window_function
        self.V = input.func
        self.F = window_function.func

        self.a1 = kwargs["a1"]
        self.a2 = kwargs["a2"]
        self.b = kwargs["b"]
        self.x0 = kwargs["x0"]
        self.Ap = kwargs["Ap"]
        self.An = kwargs["An"]
        self.Vp = kwargs["Vp"]
        self.Vn = kwargs["Vn"]
        self.alphap = kwargs["alphap"]
        self.alphan = kwargs["alphan"]
        self.xp = kwargs["xp"]
        self.xn = kwargs["xn"]
        self.eta = kwargs["eta"]

    def I(self, t, x, **kwargs):
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
        a1 = kwargs["a1"] if "a1" in kwargs else self.a1
        a2 = kwargs["a2"] if "a2" in kwargs else self.a2
        b = kwargs["b"] if "b" in kwargs else self.b

        v = self.V(t)
        if v >= 0:
            return a1 * x * np.sinh(b * v)
        else:
            return a2 * x * np.sinh(b * v)

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

    def dxdt(self, t, x, *args):
        a1 = args[0] if len(args) > 0 else self.a1
        a2 = args[1] if len(args) > 1 else self.a2
        b = args[2] if len(args) > 2 else self.b
        x0 = args[3] if len(args) > 3 else self.x0
        Ap = args[4] if len(args) > 4 else self.Ap
        An = args[5] if len(args) > 5 else self.An
        Vp = args[6] if len(args) > 6 else self.Vp
        Vn = args[7] if len(args) > 7 else self.Vn
        alphap = args[8] if len(args) > 8 else self.alphap
        alphan = args[9] if len(args) > 9 else self.alphan
        xp = args[10] if len(args) > 10 else self.xp
        xn = args[11] if len(args) > 11 else self.xn
        eta = args[12] if len(args) > 12 else self.eta

        # compile args into kwargs for other functions
        kwargs = { }
        for k in Yakopcic.parameters():
            kwargs[k] = locals()[k]

        v = self.V(t)

        return eta * self.g(v) * self.f(v, x)

    def print(self):
        print(f"{self.type}:")
        print("\tEquations:")
        print(f"\t\tx(t) = w(t)/D")
        print("\t\tV(t) = [ R_ON*x(t) + R_OFF*( 1-x(t) ) ]*I(t)*F(x)")
        print("\t\tnu_D = dx/dt = ( mu_D*R_ON/D )*I(t)")
        print("\tInput V:")
        self.input.print()
        print("\tWindow F:")
        self.window_function.print()
        self.print_equations()
        self.print_parameters()

    def print_equations(self, start="\t"):
        start_lv2 = start + "\t"
        print(start, "Equations:")
        print(start_lv2, "x(t) = w(t)/D")
        print(start_lv2, "V(t) = [ R_ON*x(t) + R_OFF*( 1-x(t) ) ]*I(t)*F(x)")
        print(start_lv2, "nu_D = dx/dt = ( mu_D*R_ON/D )*I(t)")

    def print_parameters(self, start="\t", simple=False):
        start_lv2 = start + "\t" if not simple else ""
        if not simple:
            print(start, "Parameters:")
            print(start_lv2, f"Device thickness D {self.D} m")
            print(start_lv2, f"Minimum resistance R_ON {self.R_ON} Ohm")
            print(start_lv2, f"Maximum resistance R_OFF {self.R_OFF} Ohm")
            print(start_lv2, f"Drift velocity of the oxygen deficiencies mu_D{self.m_D} m^2s^-1V^-1")
            print(start_lv2, f"Initial value of state variable x {self.x0} D")
        else:
            print([self.D, self.R_ON, self.R_OFF, self.m_D])

    @staticmethod
    def parameters():
        return ["a1", "a2", "b", "x0", "Ap", "An", "Vp", "Vn", "alphap", "alphan", "xp", "xn", "eta"]


class HPLabs():
    def __init__(self, input, window_function, **kwargs):
        self.type = "HP Labs ion-drift model"

        self.input = input
        self.window_function = window_function
        self.V = input.func
        self.F = window_function.func

        self.D = kwargs["D"] if "D" in kwargs else 27e-9
        self.R_ON = kwargs["R_ON"] if "R_ON" in kwargs else 10e3
        self.R_OFF = kwargs["R_OFF"] if "R_OFF" in kwargs else 100e3
        self.m_D = kwargs["m_D"] if "m_D" in kwargs else 1e-14
        self.x0 = kwargs["x0"] if "x0" in kwargs else 0.1

    def I(self, t, x, **kwargs):
        R_ON = kwargs["R_ON"] if "R_ON" in kwargs else self.R_ON
        R_OFF = kwargs["R_OFF"] if "R_OFF" in kwargs else self.R_OFF

        return self.V(t) / (R_ON * x + R_OFF * (1 - x))

    def mu_D(self, t, x, *args):
        D = args[0] if len(args) > 0 else self.D
        R_ON = args[1] if len(args) > 1 else self.R_ON
        R_OFF = args[2] if len(args) > 2 else self.R_OFF
        m_D = args[3] if len(args) > 3 else self.m_D

        # compile args into kwargs for other functions
        kwargs = { }
        for k in HPLabs.parameters():
            kwargs[k] = locals()[k]

        i = self.I(t, x, **kwargs)

        return ((m_D * R_ON) / D**2) * i * self.F(x=x, i=i)

    def print(self):
        print(f"{self.type}:")
        print("\tEquations:")
        print(f"\t\tx(t) = w(t)/D")
        print("\t\tV(t) = [ R_ON*x(t) + R_OFF*( 1-x(t) ) ]*I(t)*F(x)")
        print("\t\tmu_D = dx/dt = ( mu_D*R_ON/D )*I(t)")
        print("\tInput V:")
        self.input.print()
        print("\tWindow F:")
        self.window_function.print()
        self.print_equations()
        self.print_parameters()

    def print_equations(self, start="\t"):
        start_lv2 = start + "\t"
        print(start, "Equations:")
        print(start_lv2, "x(t) = w(t)/D")
        print(start_lv2, "V(t) = [ R_ON*x(t) + R_OFF*( 1-x(t) ) ]*I(t)*F(x)")
        print(start_lv2, "nu_D = dx/dt = ( mu_D*R_ON/D )*I(t)")

    def print_parameters(self, start="\t", simple=False):
        start_lv2 = start + "\t" if not simple else ""
        if not simple:
            print(start, "Parameters:")
            print(start_lv2, f"Device thickness D {self.D} m")
            print(start_lv2, f"Minimum resistance R_ON {self.R_ON} Ohm")
            print(start_lv2, f"Maximum resistance R_OFF {self.R_OFF} Ohm")
            print(start_lv2, f"Drift velocity of the oxygen deficiencies mu_D{self.m_D} m^2s^-1V^-1")
            print(start_lv2, f"Initial value of state variable x {self.x0} D")
        else:
            print([self.D, self.R_ON, self.R_OFF, self.m_D])

    @staticmethod
    def parameters():
        return ["D", "R_ON", "R_OFF", "m_D"]
