# Yakopcic Proof of Concept
This project is intended to serve as a proof of concept that Yakopcic memristor model
(https://ieeexplore.ieee.org/document/8695752) can "learn". Namely, the expectation is 
that its behaviour (resistance) in response to depression/potentiation voltage pulses will 
match that of the memristor described in: https://www.frontiersin.org/articles/10.3389/fnins.2020.627276/full#h3. 
A short description of each file and its purpose is provided below.

## Experiment Setup (experiment_setup.py)
Responsible for the setup of the experiment and provision of parameters to the memristor.
Also contains the input strings for the voltage pulses to be simulated.


## Functions (functions.py)
Contains all the functions used in the project that are not related to initialization or the 
experiment setup. 

`interactive_iv` function creates voltage pulses based on parameters such as _on_ and _off_ voltages among others.
When done, it concatenates all the pulses, as well as creates the resulting _time_ array.  


`generate_wave` function is responsible for actually constructing the array for the 
voltage pulse(s), depending on the set number of cycles. This involves creating four segments:
rising, on-time, falling, and off-time, which are concatenated (with the total array if not the
first pulse) and returned after.  


`solver2` function is used to calculate the state variable with the differential equation 
provided by a memristor model (namely, Yakopcic). Given a starting point _x0_, the solver uses the 
Euler step to iteratively calculate the state variable _x(t)_ using the voltage _v(t)_ and
the previous state variable _x(t-1)_.

`plot_images` takes data such as voltage and current among others and uses those to produce visual
output of the results. This function can also generate debug plots, depending on the supplied input.

## Yakopcic Memristor Model (yakopcic_model.py)
Contains the old and new (more interest is on the latter) Yakopcic memristor models. This involves
setting a multitude of parameters (e.g., _Vp_ and _Vn_ defining positive and negative voltage 
thresholds beyond which voltages begin altering the state variable), as well as defining numerous
functions, for example, the current _I_. For more information on the specifics of the functions 
within, refer to the docstrings within the file.

## Running the Project (old_experiment.py/new_experiment.py)
The code can be run from those two files.  
`iv_experiment.py` computes the IV-curve experiment, runs on the Yakopcic model that includes 
work by Dima (2022).  
`pulse_experiment.py` computes the pulse experiment (a long SET pulse, followed by a series of 
RESET, then SET pulses). Runs on the Yakopcic model that includes work by Dima (2022).

Example images of the run files can be seen below:

<img alt="plot_type_1.png" height="300" src="img_1.png" title="Plot type 1" width="450"/>

*Figure 1: Pulse plot (`pulse_experiment.py`). The top half depicts the resistance (in blue) 
and the voltage (in red), with the 120s SET pulse trimmed.  
The bottom half shows only the local
peaks from the resistance plot, useful to identify the changes after each pulse.  
**Note: the local peak functionality may capture extra values to show up.***

<img alt="img.png" height="300" src="img_2.png" width="450"/>

*Figure 2: IV-curve plot (`iv_experiment.py`). The top half depicts the resistance (in blue) 
and the voltage (in red) for the IV experiment.   
The bottom half shows the nonlinearity between
the current and voltage.

## Input Formatting
The formatting order in the voltage input follows the order below:
* `t_rise`**(s)**: time for the voltage to go from 'off' to 'on' state.
* `t_on`**(s)**: time the voltage remains in the 'on' state.
* `t_fall`**(s)** time for the voltage to go from 'on' to 'off' state.
* `t_off`**(s)**: time the voltage remains in the 'off' state.
* `V_on`**(V)**:  the voltage during the 'on' state.
* `V_off`**(V)**: the voltage during the 'off' state.
* `n_cycles`: the number of times the pulse is repeated.