import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from extras import ci

while True:
    experiment = input('Choose an experiment (1-5): ')
    if experiment in ['1', '2', '3', '4', '5']:
        break
    else:
        print('Invalid input. Please try again.')
experiment = int(experiment)

experiment_dict = {
    1: "Multiplying two numbers",
    2: "Combining two products",
    3: "Three separate products",
    4: "Two-dimensional circular convolution",
    5: "Three-dimensional circular convolution"
}

errors = pickle.load(open(f'testing_errors_exp_{experiment}_SAVED.pkl', 'rb'))

# compute mean testing error and confidence intervals
ci_mpes = ci(errors['mpes'])
ci_pes = ci(errors['pes'])
ci_nef = ci(errors['nef'])

# print error and ci
print(f"Mean testing error for mPES: {np.mean(ci_mpes[0]):.0f} ± {np.mean(np.abs(ci_mpes[0]-ci_mpes[1])):.0f}")
print(f"Mean testing error for PES: {np.mean(ci_pes[0]):.0f} ± {np.mean(np.abs(ci_pes[0]-ci_pes[1])):.0f}")
print(f"Mean testing error for NEF: {np.mean(ci_nef[0]):.0f} ± {np.mean(np.abs(ci_nef[0]-ci_nef[1])):.0f}")





# plot testing error
size_L = 10
size_M = 8
size_S = 6
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
# fig.set_size_inches((3.5, 3.5 * ((5. ** 0.5 - 1.0) / 2.0)))
fig.suptitle(experiment_dict[experiment], fontsize=size_L)
ax.set_ylabel("Total error", fontsize=size_M)
ax.set_xlabel("Seconds", fontsize=size_M)
ax.tick_params(axis='x', labelsize=size_S)
ax.tick_params(axis='y', labelsize=size_S)

x = (np.arange(errors['pes'].shape[1]) * 2 * 2.5) + 2.5

ax.plot(x, ci_mpes[0], label="Learned (mPES)", c="g")
ax.plot(x, ci_mpes[1], linestyle="--", alpha=0.5, c="g")
ax.plot(x, ci_mpes[2], linestyle="--", alpha=0.5, c="g")
ax.fill_between(x, ci_mpes[1], ci_mpes[2], alpha=0.3, color="g")
ax.plot(x, ci_pes[0], label="Control (PES)", c="b")
ax.plot(x, ci_pes[1], linestyle="--", alpha=0.5, c="b")
ax.plot(x, ci_pes[2], linestyle="--", alpha=0.5, c="b")
ax.fill_between(x, ci_pes[1], ci_pes[2], alpha=0.3, color="b")
ax.plot(x, ci_nef[0], label="Control (NEF)", c="r")
ax.plot(x, ci_nef[1], linestyle="--", alpha=0.5, c="r")
ax.plot(x, ci_nef[2], linestyle="--", alpha=0.5, c="r")
ax.fill_between(x, ci_nef[1], ci_nef[2], alpha=0.5, color="r")
ax.fill_between(x, ci_mpes[1], ci_mpes[2], alpha=0.3, color="g")

# -- plot linear regression too
txt_anchor = ''
for y, col, lr in zip([ci_mpes[0], ci_pes[0], ci_nef[0]], ['g', 'b', 'r'], ['mPES', 'PES', 'NEF']):
    coef = np.polyfit(x, y, 1)
    poly1d_fn = np.poly1d(coef)
    ax.plot(x, poly1d_fn(x), f'--{col}')
    print(f'{lr} learning linear regression slope: {np.polyder(poly1d_fn)[0]}')
    txt_anchor += f'{lr} slope: {np.polyder(poly1d_fn)[0]:.2f}\n'

anchored_text = AnchoredText(txt_anchor[:-1], loc=2)
ax.add_artist(anchored_text)

ax.legend(loc="best", fontsize=size_M)
fig.tight_layout()
fig.show()
fig.savefig(f'testing_errors_exp_{experiment}_PRINT.png', dpi=300)
