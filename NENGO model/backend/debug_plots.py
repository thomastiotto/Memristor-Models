import numpy as np
import matplotlib.pyplot as plt


def debugger_plots(currents, xs, rs, pulse):
    num = 4
    currents = np.array(currents)
    xs = np.array(xs)
    rs = np.array(rs)
    fig1, axs1 = plt.subplots(num, num, figsize=(10, 10))
    fig2, axs2 = plt.subplots(num, num, figsize=(10, 10))
    fig3, axs3 = plt.subplots(num, num, figsize=(10, 10))
    for j in range(0, num):
        for l in range(0, num):
            axs1[j, l].plot(currents[:, j, l])
            axs1[j, l].set_title(f"{l}->{j}")
            axs2[j, l].plot(xs[:, j, l])
            axs2[j, l].set_title(f"{l}->{j}")
            axs3[j, l].plot(rs[:, j, l])
            axs3[j, l].set_title(f"{l}->{j}")
            # axs1[j, l].set_yticklabels([])
            # axs2[j, l].set_yticklabels([])
            # axs3[j, l].set_yticklabels([])
            # axes[ i, j ].set_xticklabels( [ ] )
            plt.subplots_adjust(hspace=0.7)
    fig1.get_axes()[0].annotate("Currents over time", (0.5, 0.94),
                                xycoords='figure fraction', ha='center',
                                fontsize=20
                                )
    fig2.get_axes()[0].annotate("State variables over time", (0.5, 0.94),
                                xycoords='figure fraction', ha='center',
                                fontsize=20
                                )
    fig3.get_axes()[0].annotate("Resistance (V / I) over time", (0.5, 0.94),
                                xycoords='figure fraction', ha='center',
                                fontsize=20
                                )
    fig1.show()
    fig2.show()
    fig3.show()
    return pulse + 1
