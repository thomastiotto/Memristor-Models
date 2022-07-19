import json
import pickle
import os
import pandas as pd

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.simpledialog import askstring

import yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from scipy.integrate import solve_ivp
from scipy import integrate
from scipy.optimize import curve_fit

from backend.functions import *


# TODO mouse clicks to set parameters?
# TODO make I-V equations modular
# TODO update model+simulation with correct experimental timestep
#####################################################
#                         MODEL
#####################################################


class Model():
    def __init__( self, time, voltage ):
        self.V = Interpolated( time, voltage )

        self.h1 = self.lrs
        self.h2 = self.hrs

    def lrs( self, v, g_p, b_p, g_n, b_n ):
        return np.where( v >= 0,
                         g_p * np.sinh( b_p * v ),
                         g_n * ( 1 - np.exp( -b_n * v ) )
                         )

    def hrs( self, v, g_p, b_p, g_n, b_n ):
        return np.where( v >= 0,
                         g_p * ( 1 - np.exp( -b_p * v ) ),
                         g_n * np.sinh( b_n * v )
                         )

    def I( self, t, x, on_pars, off_pars ):
        v = self.V( t )

        return self.h1( v, *on_pars ) * x + self.h2( v, *off_pars ) * (1 - x)
    
    def g( self, v, Ap, An, Vp, Vn ):
        return np.select( [ v > Vp, v < -Vn ],
                          [ Ap * (np.exp( v ) - np.exp( Vp )),
                            -An * (np.exp( -v ) - np.exp( Vn )) ],
                          default=0 )
    
    def wp( self, x, xp ):
        return ((xp - x) / (1 - xp)) + 1
    
    def wn( self, x, xn ):
        return x / xn
    
    def f( self, v, x, xp, xn, alphap, alphan, eta ):
        return np.select( [ eta * v >= 0, eta * v <= 0 ],
                          [ np.select( [ x >= xp, x < xp ],
                                       [ np.exp( -alphap * (x - xp) ) * self.wp( x, xp ),
                                         1 ] ),
                            np.select( [ x <= xn, x > xn ],
                                       [ np.exp( alphan * (x - xn) ) * self.wn( x, xn ),
                                         1 ] )
                            ] )
    
    def dxdt( self, t, x, Ap, An, Vp, Vn, xp, xn, alphap, alphan, eta=1 ):
        v = self.V( t )
        
        return eta * self.g( v, Ap, An, Vp, Vn ) * self.f( v, x, xp, xn, alphap, alphan, eta )


class FitWindow( tk.Toplevel ):
    def __init__( self, master, xy ):
        super( FitWindow, self ).__init__( master )
        
        self.title( "Fit " )
        self.geometry( f"1200x600+{xy[ 0 ] + 10}+{xy[ 1 ]}" )
        self.resizable( False, False )
        
        self.rowconfigure( 0, weight=10 )
        
        self.input_setup()
        self.output_setup()
        self.initial_plot()
        
        self.protocol( "WM_DELETE_WINDOW", self.on_close )
        # disable/re-enable the button so we can have only one instance
        self.master.fit_button[ "state" ] = "disabled"
    
    def on_close( self ):
        self.master.Vp.set( self.master.Vp_fit.get() )
        self.master.Vn.set( self.master.Vn_fit.get() )
        
        self.master.plot_update( None )
        
        self.master.fit_button[ "state" ] = "normal"
        self.destroy()
    
    def initial_plot( self ):
        v = self.master.voltage
        i = self.master.current
        _, _, on_mask, off_mask = self.master.fit( v, i, self.master.Vp_fit, self.master.Vn_fit )
        
        self.fig, self.axis = plt.subplots( 1, 1, figsize=(12, 6) )
        self.axis.scatter( v[ on_mask ],
                           i[ on_mask ],
                           color="b",
                           label=f"On state" )
        self.axis.scatter( v[ on_mask ],
                           mim_mim_iv( v[ on_mask ], *self.master.get_on_pars() ),
                           color="c",
                           label="On fit",
                           s=1 )
        self.axis.scatter( v[ off_mask ],
                           i[ off_mask ],
                           color="r",
                           label=f"Off state" )
        self.axis.scatter( v[ off_mask ],
                           mim_mim_iv( v[ off_mask ], *self.master.get_off_pars() ),
                           color="m",
                           label="Off fit",
                           s=1 )
        self.axis.set_xlabel( "Voltage (V)" )
        self.axis.set_ylabel( "Current" )
        self.fig.legend()
        self.fig.tight_layout()
        
        # creating the Tkinter canvas
        # containing the Matplotlib figure
        canvas = FigureCanvasTkAgg( self.fig, master=self )
        canvas.draw()
        # placing the canvas on the Tkinter window
        canvas.get_tk_widget().grid( column=0, row=0, sticky=tk.N + tk.S + tk.E + tk.W )
        self.fig.canvas.draw()
    
    def plot_update( self, _ ):
        v = self.master.voltage
        i = self.master.current
        popt_on, popt_off, on_mask, off_mask = self.master.fit( v, i, self.master.Vp_fit, self.master.Vn_fit )
        
        self.master.set_on_pars( popt_on )
        self.master.set_off_pars( popt_off )
        
        # remove old lines
        self.axis.clear()
        
        # Plot new graphs
        self.axis.scatter( v[ on_mask ],
                           i[ on_mask ],
                           color="b",
                           label=f"On state" )
        self.axis.scatter( v[ on_mask ],
                           mim_mim_iv( v[ on_mask ], *self.master.get_on_pars() ),
                           color="c",
                           label="On fit",
                           s=1 )
        self.axis.scatter( v[ off_mask ],
                           i[ off_mask ],
                           color="r",
                           label=f"Off state" )
        self.axis.scatter( v[ off_mask ],
                           mim_mim_iv( v[ off_mask ], *self.master.get_off_pars() ),
                           color="m",
                           label="Off fit",
                           s=1 )
        
        self.fig.canvas.draw()
    
    def output_setup( self ):
        self.output_frame = tk.Frame( self, highlightbackground="black", highlightthickness=1 )
        self.output_frame.grid( row=2, column=0, sticky=tk.N + tk.S + tk.E + tk.W )
        
        ttk.Label( self.output_frame,
                   text="Select the negative (Vn) and positive (Vp) threshold voltages which will be used to fit the "
                        "h1 and h2 equations.\nThese equations determine the conductivity profile in the Stable ON "
                        "and Stable OFF states.\nYou can change the values by pulling or clicking on the sliders, "
                        "inputting a value directly and pressing Enter, or selecting the entry box and using Up and "
                        "Down arrows." ).grid(
                column=0, row=0 )
    
    def set_fit_function( self ):
        return
    
    def input_setup( self ):
        self.input_frame = tk.Frame( self, highlightbackground="black", highlightthickness=1 )
        self.input_frame.grid( row=1, column=0, sticky=tk.N + tk.S + tk.E + tk.W )
        self.input_frame.columnconfigure( 1, weight=5 )
        self.input_frame.columnconfigure( 4, weight=5 )
        
        # Vp
        Vp_label = ttk.Label( self.input_frame, text="Vp" )
        Vp_label.grid( column=0, row=0, padx=0, pady=5 )
        
        Vp_slider = ttk.Scale( self.input_frame, from_=0, to=np.max( self.master.voltage ), variable=self.master.Vp_fit,
                               command=self.master.plot_update )
        Vp_slider.grid( column=1, row=0, padx=0, pady=5, sticky="EW" )
        
        Vp_entry = ttk.Entry( self.input_frame, textvariable=self.master.Vp_fit )
        Vp_entry.grid( column=2, row=0, padx=0, pady=5 )
        
        # Vn
        Vn_label = ttk.Label( self.input_frame, text="Vn" )
        Vn_label.grid( column=3, row=0, padx=0, pady=5 )
        
        Vn_slider = ttk.Scale( self.input_frame, from_=0, to=np.abs( np.min( self.master.voltage ) ),
                               variable=self.master.Vn_fit,
                               command=self.master.plot_update )
        Vn_slider.grid( column=4, row=0, padx=0, pady=5, sticky="EW" )
        
        Vn_entry = ttk.Entry( self.input_frame, textvariable=self.master.Vn_fit )
        Vn_entry.grid( column=5, row=0, padx=0, pady=5 )


class PlotWindow( tk.Toplevel ):
    iv_plt_type = "lin"

    def __init__( self, master, xy ):
        super( PlotWindow, self ).__init__( master )
        
        self.title( "Simulated memristor" )
        self.geometry( f"1200x1100+{xy[ 0 ] + 10}+{xy[ 1 ]}" )
        # self.resizable( False, False )
        
        self.rowconfigure( 0, weight=10 )
        
        self.input_setup()
        self.output_setup()
        self.initial_plot()
        self.plot_data()
        
        self.protocol( "WM_DELETE_WINDOW", self.on_close )
        # disable/re-enable the button so we can have only one instance
        self.master.plot_button[ "state" ] = "disabled"
    
    def on_close( self ):
        self.master.plot_button[ "state" ] = "normal"
        self.destroy()
    
    def update_output( self, _ ):
        self.on_label.set(
            f"{self.master.gmax_p.get():.2e} * sinh({self.master.bmax_p.get():.2f} * V(t)), V(t) >= 0"
            f"\n{self.master.gmax_n.get():.2e} * (1-exp({-self.master.bmax_n.get():.2f} * V(t))), "
            f"V(t) < 0")
        self.off_label.set(
            f"{self.master.gmin_p.get():.2e} * (1-exp({-self.master.bmin_p.get():.2f} * V(t))), V(t) >= 0"
            f"\n{self.master.gmin_n.get():.2e} * sinh({self.master.bmin_n.get():.2f} * V(t)), "
            f"V(t) < 0")

    def output_setup( self ):
        self.error = tk.StringVar()
        self.error.set( str( 0.0 ) )
        
        self.on_label = tk.StringVar()
        self.off_label = tk.StringVar()
        
        self.update_output( None )
        
        self.output_frame = tk.Frame( self )
        self.output_frame.grid( row=4, column=0, sticky=tk.N + tk.S + tk.E + tk.W, pady=10 )
        
        ttk.Label( self.function_frame, text="h1 =" ).grid( row=3, column=0, padx=0, pady=5 )
        ttk.Label( self.function_frame, text="h2 =" ).grid( row=3, column=3, padx=0, pady=5 )
        ttk.Label( self.function_frame, textvariable=self.on_label ).grid( row=3, column=1, columnspan=2, sticky=tk.W )
        ttk.Label( self.function_frame, textvariable=self.off_label ).grid( row=3, column=4, columnspan=2, sticky=tk.W )
        
        ttk.Label( self.output_frame,
                   text="Select the parameters used to simulated the state variable evolution and the Stable ON and "
                        "Stable OFF states fit.\nChanging Vp and Vn does not affect the fit for h1 and h2.\n"
                        "You can change the values by pulling or clicking on the sliders, "
                        "inputting a value directly and pressing Enter, or by selecting the entry box and using Up and "
                        "Down arrows." ).grid( column=0, row=2, columnspan=2 )
    
    def input_setup( self ):
        self.plot_frame = tk.Frame( self, highlightbackground="#A4A4A4", highlightthickness=1 )
        self.plot_frame.grid( row=1, column=0, sticky=tk.N + tk.S + tk.E + tk.W, pady=10 )
        self.plot_frame.columnconfigure( 1, weight=5 )
        
        # Ap
        Ap_label = ttk.Label( self.plot_frame, text="Ap" )
        Ap_label.grid( column=0, row=3, padx=0, pady=5 )
        
        Ap_slider = ttk.Scale( self.plot_frame, from_=0, to=2, variable=self.master.Ap,
                               command=self.master.plot_update )
        Ap_slider.grid( column=1, row=3, padx=0, pady=5, sticky="EW" )
        
        Ap_entry = ttk.Entry( self.plot_frame, textvariable=self.master.Ap )
        Ap_entry.grid( column=2, row=3, padx=0, pady=5 )
        
        # An
        An_label = ttk.Label( self.plot_frame, text="An" )
        An_label.grid( column=0, row=4, padx=0, pady=5 )
        
        An_slider = ttk.Scale( self.plot_frame, from_=0, to=2, variable=self.master.An,
                               command=self.master.plot_update )
        An_slider.grid( column=1, row=4, padx=0, pady=5, sticky="EW" )
        
        An_entry = ttk.Entry( self.plot_frame, textvariable=self.master.An )
        An_entry.grid( column=2, row=4, padx=0, pady=5 )
        
        # Vp
        Vp_label = ttk.Label( self.plot_frame, text="Vp" )
        Vp_label.grid( column=0, row=5, padx=0, pady=5 )
        
        Vp_slider = ttk.Scale( self.plot_frame, from_=0, to=np.max( self.master.voltage ), variable=self.master.Vp,
                               command=self.master.plot_update )
        Vp_slider.grid( column=1, row=5, padx=0, pady=5, sticky="EW" )
        
        Vp_entry = ttk.Entry( self.plot_frame, textvariable=self.master.Vp )
        Vp_entry.grid( column=2, row=5, padx=0, pady=5 )
        
        # Vn
        Vn_label = ttk.Label( self.plot_frame, text="Vn" )
        Vn_label.grid( column=0, row=6, padx=0, pady=5 )
        
        Vn_slider = ttk.Scale( self.plot_frame, from_=0, to=np.abs( np.min( self.master.voltage ) ),
                               variable=self.master.Vn,
                               command=self.master.plot_update )
        Vn_slider.grid( column=1, row=6, padx=0, pady=5, sticky="EW" )
        
        Vn_entry = ttk.Entry( self.plot_frame, textvariable=self.master.Vn )
        Vn_entry.grid( column=2, row=6, padx=0, pady=5 )
        
        # xp
        xp_label = ttk.Label( self.plot_frame, text="xp" )
        xp_label.grid( column=0, row=7, padx=0, pady=5 )
        
        xp_slider = ttk.Scale( self.plot_frame, from_=0.0001, to=1, variable=self.master.xp,
                               command=self.master.plot_update )
        xp_slider.grid( column=1, row=7, padx=0, pady=5, sticky="EW" )
        
        xp_entry = ttk.Entry( self.plot_frame, textvariable=self.master.xp )
        xp_entry.grid( column=2, row=7, padx=0, pady=5 )
        
        # xn
        xn_label = ttk.Label( self.plot_frame, text="xn" )
        xn_label.grid( column=0, row=8, padx=0, pady=5 )
        
        xn_slider = ttk.Scale( self.plot_frame, from_=0.0001, to=1, variable=self.master.xn,
                               command=self.master.plot_update )
        xn_slider.grid( column=1, row=8, padx=0, pady=5, sticky="EW" )
        
        xn_entry = ttk.Entry( self.plot_frame, textvariable=self.master.xn )
        xn_entry.grid( column=2, row=8, padx=0, pady=5 )
        
        # alphap
        alphap_label = ttk.Label( self.plot_frame, text="alphap" )
        alphap_label.grid( column=0, row=9, padx=0, pady=5 )
        
        alphap_slider = ttk.Scale( self.plot_frame, from_=0.0001, to=10, variable=self.master.alphap,
                                   command=self.master.plot_update )
        alphap_slider.grid( column=1, row=9, padx=0, pady=5, sticky="EW" )
        
        alphap_entry = ttk.Entry( self.plot_frame, textvariable=self.master.alphap )
        alphap_entry.grid( column=2, row=9, padx=0, pady=5 )
        
        # alphan
        alphan_label = ttk.Label( self.plot_frame, text="alphan" )
        alphan_label.grid( column=0, row=10, padx=0, pady=5 )
        
        alphan_slider = ttk.Scale( self.plot_frame, from_=0.0001, to=10, variable=self.master.alphan,
                                   command=self.master.plot_update )
        alphan_slider.grid( column=1, row=10, padx=0, pady=5, sticky="EW" )
        
        alphan_entry = ttk.Entry( self.plot_frame, textvariable=self.master.alphan )
        alphan_entry.grid( column=2, row=10, padx=0, pady=5 )
        
        x0_label = ttk.Label( self.plot_frame, text="x0" )
        x0_label.grid( column=0, row=11, padx=0, pady=5 )
        
        x0_slider = ttk.Scale( self.plot_frame, from_=0, to=1, variable=self.master.x0,
                               command=self.master.plot_update )
        x0_slider.grid( column=1, row=11, padx=0, pady=5, sticky="EW" )
        
        x0_entry = ttk.Entry( self.plot_frame, textvariable=self.master.x0 )
        x0_entry.grid( column=2, row=11, padx=0, pady=5 )
        
        # fitting functions
        self.function_frame = tk.Frame( self, highlightbackground="#A4A4A4", highlightthickness=1 )
        self.function_frame.grid( row=3, column=0, sticky=tk.N + tk.S + tk.E + tk.W, pady=10 )
        
        ttk.Label( self.function_frame, text="Stable ON:" ).grid( row=0, column=1, columnspan=2,
                                                                  sticky=tk.E + tk.W, padx=0, pady=5 )
        ttk.Label( self.function_frame, text="Stable OFF:" ).grid( row=0, column=4, columnspan=2,
                                                                   sticky=tk.E + tk.W, padx=0, pady=5 )
        error = f"Average error{order_of_magnitude.symbol( self.get_fit_error()[ 0 ] )[ 2 ]}A " \
                f"({np.mean( self.get_fit_error()[ 1 ] ):.2f}% )"
        ttk.Label( self.function_frame, text=error ).grid( row=0, column=7, columnspan=2,
                                                           sticky=tk.E + tk.W, padx=0, pady=5 )
        ttk.Label( self.function_frame, text="V >= 0" ).grid( column=0, row=1, padx=0, pady=5 )
        ttk.Label( self.function_frame, text="V < 0" ).grid( column=0, row=2, padx=0, pady=5 )
        
        gmax_p_entry = ttk.Entry( self.function_frame, textvariable=self.master.gmax_p )
        gmax_p_entry.grid( column=1, row=1, padx=0, pady=5 )
        bmax_p_entry = ttk.Entry( self.function_frame, textvariable=self.master.bmax_p )
        bmax_p_entry.grid( column=2, row=1, padx=0, pady=5 )
        gmin_p_entry = ttk.Entry( self.function_frame, textvariable=self.master.gmin_p )
        gmin_p_entry.grid( column=4, row=1, padx=0, pady=5 )
        bmin_p_entry = ttk.Entry( self.function_frame, textvariable=self.master.bmin_p )
        bmin_p_entry.grid( column=5, row=1, padx=0, pady=5 )
        gmax_n_entry = ttk.Entry( self.function_frame, textvariable=self.master.gmax_n )
        gmax_n_entry.grid( column=1, row=2, padx=0, pady=5 )
        bmax_n_entry = ttk.Entry( self.function_frame, textvariable=self.master.bmax_n )
        bmax_n_entry.grid( column=2, row=2, padx=0, pady=5 )
        gmin_n_entry = ttk.Entry( self.function_frame, textvariable=self.master.gmin_n )
        gmin_n_entry.grid( column=4, row=2, padx=0, pady=5 )
        bmin_n_entry = ttk.Entry( self.function_frame, textvariable=self.master.bmin_n )
        bmin_n_entry.grid( column=5, row=2, padx=0, pady=5 )
        
        self.button_frame = tk.Frame( self )
        self.button_frame.grid( row=5, column=0, pady=10 )
        ttk.Button( self.button_frame, text="Save fitting", command=self.save ).grid( row=0, column=0, padx=50 )
        ttk.Button( self.button_frame, text="Debug", command=self.debug ).grid( row=0, column=1, padx=50 )
        ttk.Button( self.button_frame, text="Change I-V view", command=self.change_iv_view ).grid( row=0, column=2, padx=50 )
        
        # bind keyboard input
        entries = [ gmax_p_entry, bmax_p_entry, gmax_n_entry, bmax_n_entry, gmin_p_entry, bmin_p_entry, gmin_n_entry,
                    bmin_n_entry, Ap_entry, An_entry, Vp_entry, Vn_entry, xp_entry, xn_entry, alphap_entry,
                    alphan_entry, x0_entry ]
        for var, ent in zip( self.master.get_fit_variables() + self.master.get_plot_variables(), entries ):
            ent.bind( "<Return>", self.master.plot_update, add="+" )
            ent.bind( "<Return>", self.update_output, add="+" )
            ent.bind( "<Tab>", self.master.plot_update, add="+" )
            ent.bind( "<Tab>", self.update_output, add="+" )
            ent.bind( '<Up>', lambda e, var_lmb=var: self.master.nudge_var( var_lmb, "up" ) )
            ent.bind( '<Down>', lambda e, var_lmb=var: self.master.nudge_var( var_lmb, "down" ) )

    def get_fit_error(self):
        x = solver( self.master.memristor.dxdt, self.master.time,
                    dt=np.mean( np.diff( self.master.time ) ),
                    iv=self.master.x0.get(),
                    args=self.master.get_sim_pars()
                    )
        t = self.master.time
        fitted_data = self.master.memristor.I( t, x, self.master.get_on_pars(), self.master.get_off_pars() )
        simulated_data = self.master.current

        error = np.sum( np.abs( simulated_data[ 1: ] - fitted_data[ 1: ] ) )
        error_average = np.mean( error )
        error_percent = 100 * error / np.sum( np.abs( fitted_data[ 1: ] ) )
        return error_average, error_percent

    def change_iv_view( self ):
        if self.iv_plt_type == "lin":
            self.iv_plt_type = "log"
        else:
            self.iv_plt_type = "lin"
        self.plot_update( self )

    def save( self ):
        parameters = {
                "Ap"    : self.master.Ap.get(),
                "An"    : self.master.An.get(),
                "Vp"    : self.master.Vp.get(),
                "Vn"    : self.master.Vn.get(),
                "xp"    : self.master.xp.get(),
                "xn"    : self.master.xn.get(),
                "alphap": self.master.alphap.get(),
                "alphan": self.master.alphan.get(),
                "x0"    : self.master.x0.get(),
                "gmax_p": self.master.gmax_p.get(),
                "bmax_p": self.master.bmax_p.get(),
                "gmax_n": self.master.gmax_n.get(),
                "bmax_n": self.master.bmax_n.get(),
                "gmin_p": self.master.gmin_p.get(),
                "bmin_p": self.master.bmin_p.get(),
                "gmin_n": self.master.gmin_n.get(),
                "bmin_n": self.master.bmin_n.get()
                }
        
        folder_name = askstring( "Save fitting", "Enter name or leave blank for default.  "
                                                 "Folder containing files will be placed in `../fitted/` " )
        
        if folder_name == "":
            folder_name = os.path.splitext( os.path.basename( self.master.device_file ) )[ 0 ]
        
        try:
            os.makedirs( f"../fitted/{folder_name}" )
        except:
            pass
        
        with open( "../fitted/" + folder_name + "/parameters.txt", "w" ) as file:
            yaml.dump( parameters, file )
        
        df = pd.DataFrame( [ self.master.time, self.master.current, self.master.voltage ] ).transpose()
        df.to_csv( "../fitted/" + folder_name + "/data.csv", index=False, header=[ "t", "I", "V" ] )
        
        self.fig.savefig( "../fitted/" + folder_name + "/iv.png" )
    
    def initial_plot( self ):
        # simulate the model
        # x_solve_ivp = solve_ivp( self.master.memristor.dxdt, (self.master.time[ 0 ], self.master.time[ -1 ]),
        #                          [ self.master.x0.get() ],
        #                          method="LSODA",
        #                          t_eval=self.master.time,
        #                          args=self.master.get_sim_pars() )
        #
        # # updated simulation results
        # t = x_solve_ivp.t
        # x = x_solve_ivp.y[ 0, : ]
        
        # TODO dt should also be changeable
        x = solver(self.master.memristor.dxdt, self.master.time,
                   dt=np.mean( np.diff( self.master.time ) ),
                   iv=self.master.x0.get(),
                   args=self.master.get_sim_pars())
        t = self.master.time
        v = self.master.memristor.V( t )
        i = self.master.memristor.I( t, x, self.master.get_on_pars(), self.master.get_off_pars() )
        
        # create the model plot
        self.fig, self.lines, self.axes = plot_memristor( v, i, t, "", figsize=(12, 6), iv_arrows=False )
        
        # plot real data
        self.plot_data()
        
        # creating the Tkinter canvas
        # containing the Matplotlib figure
        canvas = FigureCanvasTkAgg( self.fig, master=self )
        canvas.draw()
        # placing the canvas on the Tkinter window
        canvas.get_tk_widget().grid( row=0, column=0, sticky=tk.N + tk.S + tk.E + tk.W )
        self.fig.canvas.draw()
    
    def plot_data( self ):
        i = self.master.current
        
        self.axes[ 0 ].plot( self.master.time, i, color="g", alpha=0.5 )
        self.axes[ 2 ].plot( self.master.voltage, i, color="g", alpha=0.5 )

    def plot_data_log(self):
        i = abs(self.master.current)

        self.axes[ 0 ].set_yscale( "log" )
        self.axes[ 2 ].set_yscale( "log" )
        self.axes[ 0 ].plot( self.master.time, i, color="g", alpha=0.5 )
        self.axes[ 2 ].plot( self.master.voltage, i, color="g", alpha=0.5 )
    
    # TODO I don't think that the plots (eg current) are correct
    def debug( self ):
        x = solver(self.master.memristor.dxdt, self.master.time,
                   dt=np.mean( np.diff( self.master.time ) ),
                   iv=self.master.x0.get(),
                   args=self.master.get_sim_pars())
        t = self.master.time
        v = self.master.memristor.V( t )
        i = self.master.memristor.I( t, x, self.master.get_on_pars(), self.master.get_off_pars() )
        
        fig_debug, axes_debug = plt.subplots( 5, 1, figsize=(12, 10) )
        axes_debug[ 0 ].plot( t, v )
        axes_debug[ 0 ].set_ylabel( "Voltage" )
        axes_debug[ 1 ].plot( t, i )
        axes_debug[ 1 ].set_ylabel( "Current" )
        axes_debug[ 2 ].plot( t, x )
        axes_debug[ 2 ].set_ylabel( "State Variable" )
        axes_debug[ 3 ].plot( t, self.master.memristor.g( v, self.master.Ap.get(), self.master.An.get(),
                                                          self.master.Vp.get(),
                                                          self.master.Vn.get() ) )
        axes_debug[ 3 ].set_ylabel( "g" )
        axes_debug[ 4 ].plot( t,
                              self.master.memristor.f( v, x, self.master.xp.get(), self.master.xn.get(),
                                                       self.master.alphap.get(), self.master.alphan.get(), 1 ) )
        axes_debug[ 4 ].set_ylabel( "f" )
        for ax in axes_debug.ravel():
            ax.set_xlabel( "Time" )
        fig_debug.tight_layout()
        fig_debug.show()

        print( f"max(i): {np.max( i )}, min(i): {np.min( i )} " )
        print( f"max(x): {np.max( x )}, min(x): {np.min( x )} " )

        fig_p_e, axes_p_e = plt.subplots( 2, 1, figsize=( 8, 10 ) )
        power = lambda idx: ( v * i )[ int( idx ) ]          # Callable power based on index
        energy = [ integrate.quad( power, 0, b, limit=i.size )[ 0 ] for b in range( i.size ) ]
        axes_p_e[ 0 ].plot( t, v*i )
        axes_p_e[ 0 ].set_ylabel( "Power (W)" )
        axes_p_e[ 0 ].set_xlabel( "Time (s)" )
        axes_p_e[ 1 ].plot( t, energy )
        axes_p_e[ 1 ].set_ylabel( "Energy (J)" )
        axes_p_e[ 1 ].set_xlabel( "Time (s)" )
        fig_p_e.show()

    def plot_update( self, _ ):
        error = f"Average error {order_of_magnitude.symbol( self.get_fit_error()[ 0 ] )[ 2 ]}A " \
                f"({np.mean( self.get_fit_error()[ 1 ] ):.2f} %)"
        ttk.Label( self.function_frame, text=error ).grid( row=0, column=7, columnspan=2,
                                                           sticky=tk.E + tk.W, padx=0, pady=5 )
        # simulate the model
        # x_solve_ivp = solve_ivp( self.master.memristor.dxdt, (self.master.time[ 0 ], self.master.time[ -1 ]),
        #                          [ self.master.x0 ],
        #                          method="LSODA",
        #                          t_eval=self.master.time,
        #                          args=self.master.get_sim_pars() )
        #
        # # updated simulation results
        # t = x_solve_ivp.t
        # x = x_solve_ivp.y[ 0, : ]

        x = solver( self.master.memristor.dxdt, self.master.time,
                    dt=np.mean( np.diff( self.master.time ) ),
                    iv=self.master.x0.get(),
                    args=self.master.get_sim_pars() )
        t = self.master.time
        v = self.master.memristor.V( t )
        i = self.master.memristor.I( t, x, self.master.get_on_pars(), self.master.get_off_pars() )
        
        # remove old lines
        for ax in self.axes:
            ax.clear()

        if self.iv_plt_type == "lin":
            self.axes[ 0 ].set_yscale( "linear" )
            self.axes[ 1 ].set_yscale( "linear" )
            self.axes[ 2 ].set_yscale( "linear" )
            self.axes[ 0 ].plot( t, i, color="b" )
            self.axes[ 1 ].plot( t, v, color="r" )
            self.axes[ 2 ].plot( v, i, color="b" )

            self.axes[ 0 ].set_ylim( [ np.min( i ) - np.abs( 0.5 * np.min( i ) ),
                                       np.max( i ) + np.abs( 0.5 * np.max( i ) ) ] )
            self.axes[ 1 ].set_ylim( [ np.min( v ) - np.abs( 0.5 * np.min( v ) ),
                                       np.max( v ) + np.abs( 0.5 * np.max( v ) ) ] )
            self.axes[ 2 ].set_ylim( [ np.min( i ) - np.abs( 0.5 * np.min( i ) ),
                                       np.max( i ) + np.abs( 0.5 * np.max( i ) ) ] )

            # plot real data
            self.plot_data()
        else:
            self.axes[ 0 ].set_yscale( "log" )
            self.axes[ 2 ].set_yscale( "log" )
            self.axes[ 0 ].plot( t, abs(i), color="b" )
            self.axes[ 1 ].plot( t, v, color="r" )
            self.axes[ 2 ].plot( v, abs(i), color="b" )

            # plot real data
            self.plot_data_log()
        
        # Plot new graphs
        
        self.axes[ 0 ].set_xlim( np.min( t ), np.max( t ) )
        self.axes[ 1 ].set_xlim( np.min( t ), np.max( t ) )
        self.axes[ 2 ].set_xlim(
                [ np.min( v ) - np.abs( 0.5 * np.min( v ) ), np.max( v ) + np.abs( 0.5 * np.max( v ) ) ] )
        
        self.axes[ 0 ].set_ylabel( f"Current (A)", color="b" )
        self.axes[ 0 ].set_xlabel( f"Time (s)" )
        self.axes[ 1 ].set_ylabel( 'Voltage (V)', color='r' )
        self.axes[ 2 ].set_ylabel( f"Current (A)" )
        self.axes[ 2 ].set_xlabel( "Voltage (V)" )
        
        self.fig.canvas.draw()


class MainWindow( tk.Tk ):
    def __init__( self ):
        super( MainWindow, self ).__init__()
        
        # dimensions of the main window
        self.title( "Main" )
        self.xy = (180, 180)
        self.geometry( f"{self.xy[ 0 ]}x{self.xy[ 1 ]}+0+0" )
        self.resizable( False, False )
        
        # variables for simulation
        self.x0 = tk.DoubleVar()
        self.Ap = tk.DoubleVar()
        self.An = tk.DoubleVar()
        self.Vp = tk.DoubleVar()
        self.Vn = tk.DoubleVar()
        self.xp = tk.DoubleVar()
        self.xn = tk.DoubleVar()
        self.alphap = tk.DoubleVar()
        self.alphan = tk.DoubleVar()
        
        # variables for fitting
        self.Vp_fit = tk.DoubleVar()
        self.Vn_fit = tk.DoubleVar()
        
        # variables for I-V curve
        self.gmax_p = tk.DoubleVar()
        self.bmax_p = tk.DoubleVar()
        self.gmax_n = tk.DoubleVar()
        self.bmax_n = tk.DoubleVar()
        self.gmin_p = tk.DoubleVar()
        self.bmin_p = tk.DoubleVar()
        self.gmin_n = tk.DoubleVar()
        self.bmin_n = tk.DoubleVar()
        
        self.init_variables()
        
        self.input_setup()
        
        self.plot_window = self.fit_window = None
    
    def fit( self, v, i, Vp, Vn ):
        Vp = Vp.get()
        Vn = Vn.get()
        
        on_mask = ((v > 0) & (np.gradient( v ) < 0)) \
                  | ((v < 0) & (np.gradient( v ) < 0)
                     & (v > -Vn))
        off_mask = ((v < 0) & (np.gradient( v ) > 0)) \
                   | ((v > 0) & (np.gradient( v ) > 0)
                      & (v < Vp))
        
        popt_on, pcov_on = scipy.optimize.curve_fit( self.memristor.h1, v[ on_mask ], i[ on_mask ] )
        popt_off, pcov_off = scipy.optimize.curve_fit( self.memristor.h2, v[ off_mask ], i[ off_mask ] )
        
        return popt_on, popt_off, on_mask, off_mask
    
    def nudge_var( self, var, direction, amount=0.001 ):
        if direction == "up":
            var.set( var.get() + amount )
        if direction == "down":
            var.set( var.get() - amount )
        
        self.plot_update( None )
    
    def get_fit_variables( self ):
        return [ self.gmax_p, self.bmax_p, self.gmax_n, self.bmax_n, self.gmin_p, self.bmin_p, self.gmin_n,
                 self.bmin_n ]
    
    def get_plot_variables( self ):
        return [ self.Ap, self.An, self.Vp, self.Vn, self.xp, self.xn, self.alphap, self.alphan, self.x0 ]
    
    def get_sim_pars( self ):
        return [ self.Ap.get(), self.An.get(), self.Vp.get(),
                 self.Vn.get(), self.xp.get(), self.xn.get(),
                 self.alphap.get(), self.alphan.get() ]
    
    def get_on_pars( self ):
        on_pars = [ self.gmax_p.get(), self.bmax_p.get(), self.gmax_n.get(), self.bmax_n.get() ]
        
        return on_pars
    
    def set_on_pars( self, on_pars ):
        gmax_p, bmax_p, gmax_n, bmax_n = on_pars
        
        self.gmax_p.set( gmax_p )
        self.bmax_p.set( bmax_p )
        self.gmax_n.set( gmax_n )
        self.bmax_n.set( bmax_n )
    
    def set_off_pars( self, off_pars ):
        gmin_p, bmin_p, gmin_n, bmin_n = off_pars
        
        self.gmin_p.set( gmin_p )
        self.bmin_p.set( bmin_p )
        self.gmin_n.set( gmin_n )
        self.bmin_n.set( bmin_n )
    
    def get_off_pars( self ):
        off_pars = [ self.gmin_p.get(), self.bmin_p.get(), self.gmin_n.get(), self.bmin_n.get() ]
        
        return off_pars
    
    def plot_update( self, _ ):
        try:
            if self.plot_window.state() == "normal":
                self.plot_window.plot_update( self.plot_window )
                self.plot_window.update_output( None )
        except:
            pass
        try:
            if self.fit_window.state() == "normal":
                self.fit_window.plot_update( self.fit_window )
                self.fit_window.update_output( None )
        
        except:
            pass
    
    def read_input( self, device_file ):
        extension = os.path.splitext( device_file )[ 1 ]
        
        if extension == ".pkl":
            with open( device_file, "rb" ) as file:
                df = pickle.load( file )
                
                self.time = np.array( df[ "t" ].to_list() )
                self.current = np.array( df[ "I" ].to_list() )
                self.voltage = np.array( df[ "V" ].to_list() )
        elif extension == ".csv":
            df = pd.read_csv( device_file )
            self.time = df[ "t" ].to_numpy()
            self.current = df[ "I" ].to_numpy()
            self.voltage = df[ "V" ].to_numpy()
        else:
            raise UserWarning( "File is not of accepted type" )
    
    def open_fit_window( self ):
        self.fit_window = FitWindow( self, xy=self.xy )
    
    def open_plot_window( self ):
        self.plot_window = PlotWindow( self, xy=self.xy )
    
    # TODO loading with plot window open does not correctly rescale wrt real data
    def select_file( self ):
        filetypes = (
                ('Pickled files', '*.pkl'),
                ('CSV files', '*.csv'),
                ('All files', '*.*')
                )
        try:
            self.device_file = fd.askopenfilenames(
                    title="Select a device measurement file",
                    initialdir="../imported_data/pickles",
                    filetypes=filetypes )[ 0 ]
            
            self.parameters_button[ "state" ] = "normal"
            self.fit_button[ "state" ] = "normal"
            self.plot_button[ "state" ] = "normal"
            self.reset_button[ "state" ] = "normal"
            self.regress_button[ "state" ] = "normal"
            
            self.read_input( self.device_file )
            
            self.memristor = Model( self.time, self.voltage )
            
            popt_on, popt_off, _, _ = self.fit( self.voltage, self.current, self.Vp, self.Vn )
            self.set_on_pars( popt_on )
            self.set_off_pars( popt_off )
            
            self.plot_update( None )
        except:
            pass
    
    def init_variables( self, parameters={ } ):
        self.x0.set( parameters[ "x0" ] ) if "x0" in parameters else self.x0.set( 0.11 )
        self.Ap.set( parameters[ "Ap" ] ) if "Ap" in parameters else self.Ap.set( 1.0 )
        self.An.set( parameters[ "An" ] ) if "An" in parameters else self.An.set( 1.0 )
        self.Vp.set( parameters[ "Vp" ] ) if "Vp" in parameters else self.Vp.set( 0.0 )
        self.Vn.set( parameters[ "Vn" ] ) if "Vn" in parameters else self.Vn.set( 0.0 )
        self.xp.set( parameters[ "xp" ] ) if "xp" in parameters else self.xp.set( 0.1 )
        self.xn.set( parameters[ "xn" ] ) if "xn" in parameters else self.xn.set( 0.1 )
        self.alphap.set( parameters[ "alphap" ] ) if "alphap" in parameters else self.alphap.set( 1.0 )
        self.alphan.set( parameters[ "alphan" ] ) if "alphan" in parameters else self.alphan.set( 1.0 )
        
        self.gmax_p.set( parameters[ "gmax_p" ] ) if "gmax_p" in parameters else self.gmax_p.set( 1.0 )
        self.bmax_p.set( parameters[ "bmax_p" ] ) if "bmax_p" in parameters else self.bmax_p.set( 1.0 )
        self.gmax_n.set( parameters[ "gmax_n" ] ) if "gmax_n" in parameters else self.gmax_n.set( 1.0 )
        self.bmax_n.set( parameters[ "bmax_n" ] ) if "bmax_n" in parameters else self.bmax_n.set( 1.0 )
        self.gmin_p.set( parameters[ "gmin_p" ] ) if "gmin_p" in parameters else self.gmin_p.set( 1.0 )
        self.bmin_p.set( parameters[ "bmin_p" ] ) if "bmin_p" in parameters else self.bmin_p.set( 1.0 )
        self.gmin_n.set( parameters[ "gmin_n" ] ) if "gmin_n" in parameters else self.gmin_n.set( 1.0 )
        self.bmin_n.set( parameters[ "bmin_n" ] ) if "bmin_n" in parameters else self.bmin_n.set( 1.0 )
    
    def reset( self ):
        self.read_input( self.device_file )
        
        popt_on, popt_off, _, _ = self.fit( self.voltage, self.current, self.Vp, self.Vn )
        self.set_on_pars( popt_on )
        self.set_off_pars( popt_off )
        
        self.init_variables()
        
        self.plot_update( None )
    
    # TODO loading with plot window open does not update plot (works with load data)
    def load_parameters( self ):
        filetypes = (
                ('Text files', '*.txt'),
                ('YAML files', '*.yaml'),
                ('All files', '*.*')
                )
        
        try:
            self.parameters_file = fd.askopenfilenames(
                    title="Select a file containing model parameters",
                    initialdir="../fitted",
                    filetypes=filetypes )[ 0 ]
            
            self.read_input( self.device_file )
            
            self.memristor = Model( self.time, self.voltage )
            
            popt_on, popt_off, _, _ = self.fit( self.voltage, self.current, self.Vp, self.Vn )
            self.set_on_pars( popt_on )
            self.set_off_pars( popt_off )
            
            self.plot_update( None )
        except:
            pass
        
        with open( self.parameters_file, "r" ) as file:
            parameters = yaml.load( file, Loader=yaml.SafeLoader )
        
        self.init_variables( parameters )

    def fit_param(self):
        def ode_fitting( t, An, Ap, Vn, Vp, alphan, alphap, bmax_n, bmax_p,
                        bmin_n, bmin_p, gmax_n, gmax_p, gmin_n, gmin_p, x0, xn, xp ):
            args = [ Ap, An, Vp, Vn, xp, xn, alphap, alphan, 1 ]
            on_pars = [ gmax_p, bmax_p, gmax_n, bmax_n ]
            off_pars = [ gmin_p, bmin_p, gmin_n, bmin_n ]
            sol = solve_ivp(self.memristor.dxdt, (t[ 0 ], t[ -1 ]), [x0], method="Radau",
                            t_eval=t,
                            args=args,
                            # p0=[0]
                            )
            return self.memristor.I( t, sol.y[ 0, : ], on_pars, off_pars )

        return ode_fitting

    @staticmethod
    def parameters():
        return [ "An", "Ap", "Vn", "Vp", "alphan", "alphap", "bmax_n", "bmax_p",
                "bmin_n", "bmin_p", "gmax_n", "gmax_p", "gmin_n", "gmin_p", "x0", "xn", "xp" ]

    def regress_param( self ):
        print("Running curve_fit")
        popt, pcov = curve_fit(self.fit_param(), self.time, self.current,
                               bounds=([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [3, 3, 0.05, 0.05, 10, 10, 10, 10, 10, 20, 1, 1, 1, 10, 1, 1, 1]),
                               p0=[0.0255, 0.071, 0.00, 0.00, 1, 9.2, 6, 5.5, 3.13, 0.01,
                                   1.05e-05, 2.7e-04, 1.95e-05, 0.04, 0, 0.152, 0.11],
                               maxfev=100000
                               )

        fitted_params = {p: v for p, v in zip(self.parameters(), popt)}
        # print("Fitted parameters", [(p, np.round(v, 2)) for p, v in zip(self.parameters(), popt)])
        for k,v in fitted_params.items(): print(f'{k}: {v}')
    
    def input_setup( self ):
        self.data_button = ttk.Button( self, text="Load data", command=self.select_file )
        self.data_button.pack( side="top", fill="x" )
        self.parameters_button = ttk.Button( self, text="Load parameters", command=self.load_parameters,
                                             state="disabled" )
        self.parameters_button.pack( side="top", fill="x" )
        self.regress_button = ttk.Button(self, text="Regress parameters", command=self.regress_param, state="disabled")
        self.regress_button.pack(side="top", fill="x")
        self.fit_button = ttk.Button( self, text="Fit", command=self.open_fit_window, state="disabled" )
        self.fit_button.pack( side="top", fill="x" )
        self.plot_button = ttk.Button( self, text="Plot", command=self.open_plot_window, state="disabled" )
        self.plot_button.pack( side="top", fill="x" )
        self.reset_button = ttk.Button( self, text="Reset", command=self.reset, state="disabled" )
        # self.reset_button.pack( side="top", fill="x" )
        self.quit_button = ttk.Button( self, text="Quit", command=self.destroy )
        self.quit_button.pack( side="top", fill="x" )


app = MainWindow()
app.mainloop()
