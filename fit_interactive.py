import pickle
import matplotlib.pyplot as plt
import numpy as np
from block_timer.timer import Timer
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from progressbar import progressbar
import scipy.stats as stats
from order_of_magnitude import order_of_magnitude
import os
import multiprocessing as mp
import argparse

from functions import *
from models import *
from experiments import *

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from functions import *


# TODO mouse clicks to set parameters?
#####################################################
#                         MODEL
#####################################################

def fit( v, i, Vp, Vn, on_fit=mim_mim_iv, off_fit=mim_mim_iv ):
    Vp = Vp.get()
    Vn = Vn.get()
    
    on_mask = ((v > 0) & (np.gradient( v ) < 0)) \
              | ((v < 0) & (np.gradient( v ) < 0)
                 & (v > -Vn))
    off_mask = ((v < 0) & (np.gradient( v ) > 0)) \
               | ((v > 0) & (np.gradient( v ) > 0)
                  & (v < Vp))
    
    popt_on, pcov_on = scipy.optimize.curve_fit( on_fit, v[ on_mask ],
                                                 i[ on_mask ] )
    popt_off, pcov_off = scipy.optimize.curve_fit( off_fit, v[ off_mask ],
                                                   i[ off_mask ] )
    
    return popt_on, popt_off, on_mask, off_mask


class Model():
    def __init__( self, time, voltage ):
        self.V = Interpolated( time, voltage )
        self.I = self.I_mim_mim_mim_mim
    
    def g( self, v, Ap, An, Vp, Vn ):
        if v > Vp:
            return Ap * (np.exp( v ) - np.exp( Vp ))
        elif v < -Vn:
            return -An * (np.exp( -v ) - np.exp( Vn ))
        else:
            return 0
    
    def wp( self, x, xp ):
        return ((xp - x) / (1 - xp)) + 1
    
    def wn( self, x, xn ):
        return x / (1 - xn)
    
    def f( self, v, x, xp, xn, alphap, alphan, eta ):
        if eta * v > 0:
            if x >= xp:
                return np.exp( -alphap * (x - xp) ) * self.wp( x, xp )
            else:
                return 1
        else:
            if x <= 1 - xn:
                return np.exp( alphan * (x + xn - 1) ) * self.wn( x, xn )
            else:
                return 1
    
    def ohmic_iv( self, v, g ):
        return g * v
    
    def mim_iv( self, v, g, b ):
        return g * np.sinh( b * v )
    
    def mim_mim_iv( self, v, gp, bp, gn, bn ):
        return np.piecewise( v, [ v < 0, v >= 0 ],
                             [ lambda v: mim_iv( v, gn, bn ), lambda v: mim_iv( v, gp, bp ) ] )
    
    def I_mim_mim( self, t, x, gmax, bmax, gmin, bmin ):
        v = self.V( t )
        return mim_iv( v, gmax, bmax ) * x + mim_iv( v, gmin, bmin ) * (1 - x)
    
    def I_mim_mim_mim_mim( self, t, x, on_pars, off_pars ):
        v = self.V( t )
        return mim_mim_iv( v, *on_pars ) * x + mim_mim_iv( v, *off_pars ) * (1 - x)
    
    def dxdt( self, t, x, Ap, An, Vp, Vn, xp, xn, alphap, alphan, ):
        eta = 1
        v = self.V( t )
        return eta * self.g( v, Ap, An, Vp, Vn ) * self.f( v, x, xp, xn, alphap, alphan, eta )


class FitWindow( tk.Toplevel ):
    def __init__( self, master, xy ):
        super( FitWindow, self ).__init__( master )
        
        self.title( "Fit " )
        self.geometry( f"1200x600+{xy[ 0 ] + 10}+{xy[ 1 ]}" )
        self.resizable( False, False )
        
        # input window setup
        self.rowconfigure( 0, weight=10 )
        self.columnconfigure( 1, weight=10 )
        
        self.input_setup()
        self.output_setup()
        self.initial_plot()
        
        self.protocol( "WM_DELETE_WINDOW", self.on_close )
        # disable/re-enable the button so we can have only one instance
        self.master.fit_button[ "state" ] = "disabled"
    
    def on_close( self ):
        self.master.fit_button[ "state" ] = "normal"
        self.destroy()
    
    def initial_plot( self ):
        v = self.master.voltage
        i = self.master.current
        _, _, on_mask, off_mask = fit( v, i, self.master.Vp, self.master.Vn )
        
        self.fig, self.axis = plt.subplots( 1, 1 )
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
        canvas.get_tk_widget().grid( column=0, row=0, columnspan=3 )
        self.fig.canvas.draw()
        
        self.update_output()
    
    def plot_update( self, _ ):
        v = self.master.voltage
        i = self.master.current
        popt_on, popt_off, on_mask, off_mask = fit( v, i, self.master.Vp, self.master.Vn )
        
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
    
    # TODO vary the polarity threshold
    def update_output( self ):
        self.on_label.set( f"{self.master.gmax_p.get():.2e} * sinh({self.master.bmax_p.get():.2f} * V(t)), V(t) >= 0  "
                           f"|  {self.master.gmax_n.get():.2e} * sinh({self.master.bmax_n.get():.2f} * V(t)), "
                           f"V(t) < 0" )
        self.off_label.set( f"{self.master.gmin_p.get():.2e} * sinh({self.master.bmin_p.get():.2f} * V(t)), V(t) >= 0  "
                            f"|  {self.master.gmin_n.get():.2e} * sinh({self.master.bmin_n.get():.2f} * V(t)), "
                            f"V(t) < 0" )
    
    def output_setup( self ):
        self.error = tk.StringVar()
        self.error.set( str( 0.0 ) )
        
        # error_frame = ttk.Frame( self )
        # ttk.Label( error_frame, text="Error" ).grid( column=0, row=0 )
        # ttk.Label( error_frame, textvariable=self.error ).grid( column=0, row=1, sticky=tk.W )
        # for widget in error_frame.winfo_children():
        #     widget.grid( padx=0, pady=0 )
        # error_frame.grid( column=2, row=0 )
        #
        
        self.on_label = tk.StringVar()
        self.off_label = tk.StringVar()
        
        self.update_output()
        
        ttk.Label( self, text="On fit" ).grid( column=0, row=1 )
        ttk.Label( self, textvariable=self.on_label ).grid( column=1, row=1, sticky=tk.W )
        ttk.Label( self, text="Off fit" ).grid( column=0, row=2 )
        ttk.Label( self, textvariable=self.off_label ).grid( column=1, row=2, sticky=tk.W )
    
    def input_setup( self ):
        # Vp
        Vp_label = ttk.Label( self, text="Vp" )
        Vp_label.grid( column=0, row=3, padx=0, pady=5 )
        
        Vp_slider = ttk.Scale( self, from_=0, to=np.max( self.master.voltage ), variable=self.master.Vp,
                               command=self.master.plot_update )
        Vp_slider.grid( column=1, row=3, padx=0, pady=5, sticky="EW" )
        
        Vp_entry = ttk.Entry( self, textvariable=self.master.Vp )
        Vp_entry.grid( column=2, row=3, padx=0, pady=5 )
        
        Vp_entry.bind( "<Return>", self.master.plot_update )
        Vp_entry.bind( '<Up>', lambda e: self.master.nudge_var( self.master.Vp, "up" ) )
        Vp_entry.bind( '<Down>', lambda e: self.master.nudge_var( self.master.Vp, "down" ) )
        
        # Vn
        Vn_label = ttk.Label( self, text="Vn" )
        Vn_label.grid( column=0, row=4, padx=0, pady=5 )
        
        Vn_slider = ttk.Scale( self, from_=0, to=np.abs( np.min( self.master.voltage ) ), variable=self.master.Vn,
                               command=self.master.plot_update )
        Vn_slider.grid( column=1, row=4, padx=0, pady=5, sticky="EW" )
        
        Vn_entry = ttk.Entry( self, textvariable=self.master.Vn )
        Vn_entry.grid( column=2, row=4, padx=0, pady=5 )
        
        Vn_entry.bind( "<Return>", self.master.plot_update )
        Vn_entry.bind( '<Up>', lambda e: self.master.nudge_var( self.master.Vn, "up" ) )
        Vn_entry.bind( '<Down>', lambda e: self.master.nudge_var( self.master.Vn, "down" ) )


class PlotWindow( tk.Toplevel ):
    def __init__( self, master, xy ):
        super( PlotWindow, self ).__init__( master )
        
        self.title( "Simulated memristor" )
        self.geometry( f"1200x1000+{xy[ 0 ] + 10}+{xy[ 1 ]}" )
        self.resizable( False, False )
        
        self.memristor = Model( self.master.time, self.master.voltage )
        
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
    
    def update_output( self ):
        self.on_label.set( f"{self.master.gmax_p.get():.2e} * sinh({self.master.bmax_p.get():.2f} * V(t)), V(t) >= 0  "
                           f"|  {self.master.gmax_n.get():.2e} * sinh({self.master.bmax_n.get():.2f} * V(t)), "
                           f"V(t) < 0" )
        self.off_label.set( f"{self.master.gmin_p.get():.2e} * sinh({self.master.bmin_p.get():.2f} * V(t)), V(t) >= 0  "
                            f"|  {self.master.gmin_n.get():.2e} * sinh({self.master.bmin_n.get():.2f} * V(t)), "
                            f"V(t) < 0" )
    
    def output_setup( self ):
        self.error = tk.StringVar()
        self.error.set( str( 0.0 ) )
        
        # error_frame = ttk.Frame( self )
        # ttk.Label( error_frame, text="Error" ).grid( column=0, row=0 )
        # ttk.Label( error_frame, textvariable=self.error ).grid( column=0, row=1, sticky=tk.W )
        # for widget in error_frame.winfo_children():
        #     widget.grid( padx=0, pady=0 )
        # error_frame.grid( column=2, row=0 )
        #
        
        self.on_label = tk.StringVar()
        self.off_label = tk.StringVar()
        
        self.update_output()
        
        ttk.Label( self, text="On fit" ).grid( column=0, row=1 )
        ttk.Label( self, textvariable=self.on_label ).grid( column=1, row=1, sticky=tk.W )
        ttk.Label( self, text="Off fit" ).grid( column=0, row=2 )
        ttk.Label( self, textvariable=self.off_label ).grid( column=1, row=2, sticky=tk.W )
    
    def input_setup( self ):
        # Ap
        Ap_label = ttk.Label( self, text="Ap" )
        Ap_label.grid( column=0, row=3, padx=0, pady=5 )
        
        Ap_slider = ttk.Scale( self, from_=0.0001, to=10, variable=self.master.Ap,
                               command=self.master.plot_update )
        Ap_slider.grid( column=1, row=3, padx=0, pady=5, sticky="EW" )
        
        Ap_entry = ttk.Entry( self, textvariable=self.master.Ap )
        Ap_entry.grid( column=2, row=3, padx=0, pady=5 )
        
        Ap_entry.bind( "<Return>", self.master.plot_update )
        Ap_entry.bind( '<Up>', lambda e: self.master.nudge_var( self.master.Ap, "up" ) )
        Ap_entry.bind( '<Down>', lambda e: self.master.nudge_var( self.master.Ap, "down" ) )
        
        # An
        An_label = ttk.Label( self, text="An" )
        An_label.grid( column=0, row=4, padx=0, pady=5 )
        
        An_slider = ttk.Scale( self, from_=0.0001, to=10, variable=self.master.An,
                               command=self.master.plot_update )
        An_slider.grid( column=1, row=4, padx=0, pady=5, sticky="EW" )
        
        An_entry = ttk.Entry( self, textvariable=self.master.An )
        An_entry.grid( column=2, row=4, padx=0, pady=5 )
        
        An_entry.bind( "<Return>", self.master.plot_update )
        An_entry.bind( '<Up>', lambda e: self.master.nudge_var( self.master.An, "up" ) )
        An_entry.bind( '<Down>', lambda e: self.master.nudge_var( self.master.An, "down" ) )
        
        # Vp
        Vp_label = ttk.Label( self, text="Vp" )
        Vp_label.grid( column=0, row=5, padx=0, pady=5 )
        
        Vp_slider = ttk.Scale( self, from_=0, to=np.max( self.master.voltage ), variable=self.master.Vp,
                               command=self.master.plot_update )
        Vp_slider.grid( column=1, row=5, padx=0, pady=5, sticky="EW" )
        
        Vp_entry = ttk.Entry( self, textvariable=self.master.Vp )
        Vp_entry.grid( column=2, row=5, padx=0, pady=5 )
        
        Vp_entry.bind( "<Return>", self.master.plot_update )
        Vp_entry.bind( '<Up>', lambda e: self.master.nudge_var( self.master.Vp, "up" ) )
        Vp_entry.bind( '<Down>', lambda e: self.master.nudge_var( self.master.Vp, "down" ) )
        
        # Vn
        Vn_label = ttk.Label( self, text="Vn" )
        Vn_label.grid( column=0, row=6, padx=0, pady=5 )
        
        Vn_slider = ttk.Scale( self, from_=0, to=np.abs( np.min( self.master.voltage ) ), variable=self.master.Vn,
                               command=self.master.plot_update )
        Vn_slider.grid( column=1, row=6, padx=0, pady=5, sticky="EW" )
        
        Vn_entry = ttk.Entry( self, textvariable=self.master.Vn )
        Vn_entry.grid( column=2, row=6, padx=0, pady=5 )
        
        Vn_entry.bind( "<Return>", self.master.plot_update )
        Vn_entry.bind( '<Up>', lambda e: self.master.nudge_var( self.master.Vn, "up" ) )
        Vn_entry.bind( '<Down>', lambda e: self.master.nudge_var( self.master.Vn, "down" ) )
        
        # xp
        xp_label = ttk.Label( self, text="xp" )
        xp_label.grid( column=0, row=7, padx=0, pady=5 )
        
        xp_slider = ttk.Scale( self, from_=0.0001, to=1, variable=self.master.xp,
                               command=self.master.plot_update )
        xp_slider.grid( column=1, row=7, padx=0, pady=5, sticky="EW" )
        
        xp_entry = ttk.Entry( self, textvariable=self.master.xp )
        xp_entry.grid( column=2, row=7, padx=0, pady=5 )
        
        xp_entry.bind( "<Return>", self.master.plot_update )
        xp_entry.bind( '<Up>', lambda e: self.master.nudge_var( self.master.xp, "up" ) )
        xp_entry.bind( '<Down>', lambda e: self.master.nudge_var( self.master.xp, "down" ) )
        
        # xn
        xn_label = ttk.Label( self, text="xn" )
        xn_label.grid( column=0, row=8, padx=0, pady=5 )
        
        xn_slider = ttk.Scale( self, from_=0.0001, to=1, variable=self.master.xn,
                               command=self.master.plot_update )
        xn_slider.grid( column=1, row=8, padx=0, pady=5, sticky="EW" )
        
        xn_entry = ttk.Entry( self, textvariable=self.master.xn )
        xn_entry.grid( column=2, row=8, padx=0, pady=5 )
        
        xn_entry.bind( "<Return>", self.master.plot_update )
        xn_entry.bind( '<Up>', lambda e: self.master.nudge_var( self.master.xn, "up" ) )
        xn_entry.bind( '<Down>', lambda e: self.master.nudge_var( self.master.xn, "down" ) )
        
        # alphap
        alphap_label = ttk.Label( self, text="alphap" )
        alphap_label.grid( column=0, row=9, padx=0, pady=5 )
        
        alphap_slider = ttk.Scale( self, from_=0.0001, to=10, variable=self.master.alphap,
                                   command=self.master.plot_update )
        alphap_slider.grid( column=1, row=9, padx=0, pady=5, sticky="EW" )
        
        alphap_entry = ttk.Entry( self, textvariable=self.master.alphap )
        alphap_entry.grid( column=2, row=9, padx=0, pady=5 )
        
        alphap_entry.bind( "<Return>", self.master.plot_update )
        alphap_entry.bind( '<Up>', lambda e: self.master.nudge_var( self.master.alphap, "up" ) )
        alphap_entry.bind( '<Down>', lambda e: self.master.nudge_var( self.master.alphap, "down" ) )
        
        # alphan
        alphan_label = ttk.Label( self, text="alphan" )
        alphan_label.grid( column=0, row=10, padx=0, pady=5 )
        
        alphan_slider = ttk.Scale( self, from_=0.0001, to=10, variable=self.master.alphan,
                                   command=self.master.plot_update )
        alphan_slider.grid( column=1, row=10, padx=0, pady=5, sticky="EW" )
        
        alphan_entry = ttk.Entry( self, textvariable=self.master.alphan )
        alphan_entry.grid( column=2, row=10, padx=0, pady=5 )
        
        alphan_entry.bind( "<Return>", self.master.plot_update )
        alphan_entry.bind( '<Up>', lambda e: self.master.nudge_var( self.master.alphan, "up" ) )
        alphan_entry.bind( '<Down>', lambda e: self.master.nudge_var( self.master.alphan, "down" ) )
    
    def initial_plot( self ):
        # simulate the model
        # x_solve_ivp = solve_ivp( self.memristor.dxdt, (self.master.time[ 0 ], self.master.time[ -1 ]),
        #                          [ self.master.x0 ],
        #                          method="LSODA",
        #                          t_eval=self.master.time,
        #                          args=[ self.master.Ap.get(), self.master.An.get(), self.master.Vp.get(),
        #                                 self.master.Vn.get(), self.master.xp.get(), self.master.xn.get() ] )
        #
        # # updated simulation results
        # t = x_solve_ivp.t
        # x = x_solve_ivp.y[ 0, : ]
        
        x = euler_solver( self.memristor.dxdt, self.master.time,
                          dt=np.mean( np.diff( self.master.time ) ),
                          iv=self.master.x0,
                          args=self.master.get_sim_pars() )
        t = self.master.time
        v = self.memristor.V( t )
        i = self.memristor.I( t, x, self.master.get_on_pars(), self.master.get_off_pars() )
        
        # create the model plot
        self.fig, self.lines, self.axes = plot_memristor( v, i, t, "", figsize=(12, 6), iv_arrows=False )
        
        # plot real data
        self.plot_data()
        
        # creating the Tkinter canvas
        # containing the Matplotlib figure
        canvas = FigureCanvasTkAgg( self.fig, master=self )
        canvas.draw()
        # placing the canvas on the Tkinter window
        canvas.get_tk_widget().grid( column=0, row=0, columnspan=3 )
        
        self.fig.canvas.draw()
    
    def plot_data( self ):
        i_oom = order_of_magnitude.symbol( np.max( self.master.current ) )
        i_scaled = self.master.current * 1 / i_oom[ 0 ]
        
        self.axes[ 0 ].plot( self.master.time, i_scaled, color="g", alpha=0.5 )
        self.axes[ 2 ].plot( self.master.voltage, i_scaled, color="g", alpha=0.5 )
    
    def plot_update( self, _ ):
        # simulate the model
        # x_solve_ivp = solve_ivp( self.memristor.dxdt, (self.master.time[ 0 ], self.master.time[ -1 ]),
        #                          [ self.master.x0 ],
        #                          method="LSODA",
        #                          t_eval=self.master.time,
        #                          args=[ self.master.Ap.get(), self.master.An.get(), self.master.Vp.get(),
        #                                 self.master.Vn.get(), self.master.xp.get(), self.master.xn.get() ] )
        #
        # # updated simulation results
        # t = x_solve_ivp.t
        # x = x_solve_ivp.y[ 0, : ]
        
        x = euler_solver( self.memristor.dxdt, self.master.time,
                          dt=np.mean( np.diff( self.master.time ) ),
                          iv=self.master.x0,
                          args=self.master.get_sim_pars() )
        t = self.master.time
        v = self.memristor.V( t )
        i = self.memristor.I( t, x, self.master.get_on_pars(), self.master.get_off_pars() )
        
        # Adjust to new limits
        # self.axes[ 0 ].set_xlim( [ 0, np.max( t ) ] )
        # self.axes[ 1 ].set_xlim( [ 0, np.max( t ) ] )
        # self.axes[ 1 ].set_ylim(
        #         [ np.min( v ) - np.abs( 0.5 * np.min( v ) ), np.max( v ) + np.abs( 0.5 * np.max( v ) ) ] )
        # self.axes[ 2 ].set_xlim(
        #         [ np.min( v ) - np.abs( 0.5 * np.min( v ) ), np.max( v ) + np.abs( 0.5 * np.max( v ) ) ] )
        
        i_oom = order_of_magnitude.symbol( np.max( i ) )
        i_scaled = i * 1 / i_oom[ 0 ]
        
        # remove old lines
        for ax in self.axes:
            ax.clear()
        
        # Plot new graphs
        self.axes[ 0 ].plot( t, i_scaled, color="b" )
        self.axes[ 1 ].plot( t, v, color="r" )
        self.axes[ 2 ].plot( v, i_scaled, color="b" )
        
        # plot real data
        self.plot_data()
        
        # Adjust to new limits
        # self.axes[ 0 ].set_ylim( [ np.min( i_scaled ) - np.abs( 0.5 * np.min( i_scaled ) ),
        #                            np.max( i_scaled ) + np.abs( 0.5 * np.max( i_scaled ) ) ] )
        # self.axes[ 0 ].set_ylabel( f"Current ({i_oom[ 1 ]}A)", color="b" )
        # self.axes[ 2 ].set_ylim( [ np.min( i_scaled ) - np.abs( 0.5 * np.min( i_scaled ) ),
        #                            np.max( i_scaled ) + np.abs( 0.5 * np.max( i_scaled ) ) ] )
        
        self.fig.canvas.draw()


class MainWindow( tk.Tk ):
    def __init__( self ):
        super( MainWindow, self ).__init__()
        
        # dimensions of the main window
        self.title( "Main" )
        self.xy = (200, 120)
        self.geometry( f"{self.xy[ 0 ]}x{self.xy[ 1 ]}+0+0" )
        self.resizable( False, False )
        
        # variables for simulation
        self.x0 = 0.11
        self.Ap = tk.DoubleVar()
        self.An = tk.DoubleVar()
        self.Vp = tk.DoubleVar()
        self.Vn = tk.DoubleVar()
        self.xp = tk.DoubleVar()
        self.xn = tk.DoubleVar()
        self.alphap = tk.DoubleVar()
        self.alphan = tk.DoubleVar()
        
        self.Ap.set( 1.0 )
        self.An.set( 1.0 )
        self.Vp.set( 0.0 )
        self.Vn.set( 0.0 )
        self.xp.set( 0.1 )
        self.xn.set( 0.1 )
        self.alphap.set( 0.1 )
        self.alphan.set( 0.1 )
        
        self.read_input()
        
        self.main_setup()
        
        self.plot_window = self.fit_window = None
        
        # variables for I-V curve
        self.gmax_p = tk.DoubleVar()
        self.bmax_p = tk.DoubleVar()
        self.gmax_n = tk.DoubleVar()
        self.bmax_n = tk.DoubleVar()
        self.gmin_p = tk.DoubleVar()
        self.bmin_p = tk.DoubleVar()
        self.gmin_n = tk.DoubleVar()
        self.bmin_n = tk.DoubleVar()
        
        popt_on, popt_off, _, _ = fit( self.voltage, self.current, self.Vp, self.Vn )
        self.set_on_pars( popt_on )
        self.set_off_pars( popt_off )
    
    def nudge_var( self, var, direction, amount=0.001 ):
        if direction == "up":
            var.set( var.get() + amount )
        if direction == "down":
            var.set( var.get() - amount )
        
        self.plot_update( None )
    
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
                self.plot_window.update_output()
        except:
            pass
        try:
            if self.fit_window.state() == "normal":
                self.fit_window.plot_update( self.fit_window )
                self.fit_window.update_output()
        
        except:
            pass
    
    def read_input( self ):
        with open( f"./plots/Radius 10 um/-4V_1.pkl", "rb" ) as file:
            df = pickle.load( file )
        
        self.time = np.array( df[ "t" ].to_list() )[ :-1 ]
        self.current = np.array( df[ "I" ].to_list() )[ :-1 ]
        self.resistance = np.array( df[ "R" ].to_list() )[ :-1 ]
        self.conductance = 1 / self.resistance
        self.voltage = np.array( df[ "V" ].to_list() )[ :-1 ]
    
    def open_fit_window( self ):
        self.fit_window = FitWindow( self, xy=self.xy )
    
    def open_plot_window( self ):
        self.plot_window = PlotWindow( self, xy=self.xy )
    
    def main_setup( self ):
        self.fit_button = ttk.Button( self, text="Fit", command=self.open_fit_window )
        self.fit_button.pack()
        self.plot_button = ttk.Button( self, text="Plot", command=self.open_plot_window )
        self.plot_button.pack()
        self.quit_button = ttk.Button( self, text="Quit", command=self.destroy )
        self.quit_button.pack()


app = MainWindow()
app.mainloop()
