# Pairing a Compact Memristor Model to a Spiking Neural Network Simulator

<p align="center">
  <img width="760" height="400" src="https://github.com/dmkhitaryan/Pairing-a-Compact-Memristor-Model-to-a-SNN-Simulator/blob/main/qualitative.png">
</p>

## Abstract
As learning algorithms continue to expand in scale and complexity, their computational and energy demands become increasingly harder to meet. A potential solution to those problems is memristors: devices with the ability to “memorize” their resistance, even when the power supply is halted. Their properties could be leveraged to build low-energy profile neural networks. Previous research has explored this prospect by simulating a spiking neural network with Nb:STO-memristor-based synapses. Implemented within the Nengo framework and updating from local knowledge, its performance matched traditional neural networks. Here, a different memristor model is used to build a spiking neural network, after which the experiment was reproduced. The results were shown to be overall competitive with current learning algorithms, as well as the Nb:STO-memristor based solution, further solidifying the viability of memristive devices for computing purposes in the future.

## Running the code
* ``mPES.py`` runs a single simulation by using the simulated Yakopcic memristors in the ``memristor_nengo`` library.
* ``averaging_mPES.py`` runs the previous file across ``n`` runs, averaging statistical measurements taken.

## Original Paper and Model
This research project was based on the model designed by Tiotto _et al._ (2021). The original model can be found here via the GitHub link below:  
https://github.com/thomastiotto/Learning-to-approximate-functions-using-niobium-doped-strontium-titanate-memristors

