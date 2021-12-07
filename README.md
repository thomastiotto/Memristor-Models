# Memristor Models

Python implementations and simulations of HP Labs Ion Drift and Yakopcic memristor models.

![iv](https://github.com/Tioz90/Memristor-Models/blob/master/images/iv.png?raw=true)


## Files

### Frontend

- `simulate.py` runs experiment based on simulation of the HP Labs ion drift memristor model [1] or of the Yakopcic
  memristor model [2]. The latter can be used to match characterisations for devices from literature [3-5]. The model is
  simulated, noise is added to the current measurement, and this noisy measurement is used to regress the model
  parameters to obtain a model capable of reproducing the original data.

    - `-e` selects the experiment to run `[hp_sine/hp_pulsed/oblea_sine/oblea_pulsed/miao/jo]`.
    - `-s` selects the solver to use to simulate the model `[LSODA/EU/RK4]`; more than one can be passed, and the
      experiment will be run for each one.
    - `--video` generates a video of the simulation of the memristor model.

- `data_import.py`  reads the `.csv` files containing the I-V measurements and outputs them as pickled files in
  the `. /pickles` folder, and as plots in the `./plots` folder. The data for our memristive device is available on
  request.

- `hp_labs_interactive.py` launches a matplotlib-based GUI that lets the user experiment in real time with changing the
  HP Labs ion drift memristor model [1] parameters.

- `yakopcic_interactive.py` launches a matplotlib-based GUI that lets the user experiment in real time with changing the
  Yakopcic model [2] parameters.

- `fit_interactive.py` launches a Tk-based GUI application that lets the user load a pickled file created by
  `data_import.py`, and fit the updated Yakopcic model [6] to the real device's behaviour.

  ![interactive3](https://github.com/Tioz90/Memristor-Models/blob/master/images/interactive3.png?raw=true)

- `fit_yakopcic.py` loads a pickled file created by `data_import.py`, and attempts to reproduce the semi-automated model
  parameter fitting procedure outlined in [6].

- `load_and_fit.py` loads a pickled file created by `data_import.py`, and attempts a blind regression on model the
  Yakopcic model parameters.

### Backend

- `models.py` contains implementation of the HP Labs ion drift memristor model [1], Yakopcic generalised memristor
  model [2], and an updated version of Yakopcic's model [6].

- `experiments.py` contains the experiment definitions used by `simulate.py`.

- `functions.py` contains various helper functions used throughout the code.

## References

*[1] Yang, J. J. et al. Memristive switching mechanism for metal/oxide/metal nanodevices. Nat Nanotechnol 3, 429–433
(2008).*

*[2] Yakopcic, C., Taha, T. M., Subramanyam, G., Pino, R. E. & Rogers, S. A Memristor Device Model. Ieee Electr Device L
32, 1436–1438 (2011).*

*[3] Oblea, A. S., Timilsina, A., Moore, D. & Campbell, K. A. Silver Chalcogenide Based Memristor Devices. 2010 Int Jt
Conf Neural Networks Ijcnn 1, 1–3 (2010).*

*[4] Miao, F. et al. Anatomy of a Nanoscale Conduction Channel Reveals the Mechanism of a High‐Performance Memristor.
Adv Mater 23, 5633–5640 (2011).*

*[5] C. Yakopcic, T. M. Taha, G. Subramanyam, and R. E. Pino, “Generalized Memristive Device SPICE Model and its
Application in Circuit Design,” IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 32(8)
August, 2013 pp. 1201-1214.*

*[6] Yakopcic, C. et al. Memristor Model Optimization Based on Parameter Extraction From Device Characterization Data.
Ieee T Comput Aid D 39, 1084–1095 (2020).*

  
