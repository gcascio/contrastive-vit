import numpy as np
import matplotlib.pyplot as plt


total_steps = 1000
warmup_steps = 100
steps = range(0, warmup_steps + 1)
steps2 = range(warmup_steps, total_steps)
lr = 1.


def cosine_weight_decay(x: int):
    if x < warmup_steps:
        return lr * x / warmup_steps

    inner = (x - warmup_steps) * np.pi / (total_steps - warmup_steps)

    return 0.5 * (1 + np.cos(inner)) * lr


y = list(map(cosine_weight_decay, steps))
y2 = list(map(cosine_weight_decay, steps2))

plt.rcParams["figure.figsize"] = (7, 3.5)

plt.plot(steps, y, color='green', label='Warmup Steps')
plt.plot(steps2, y2, color='blue', label='Cosine Annealing')

plt.vlines(x=[100], ymin=0, ymax=1, colors='dimgrey', ls='--', lw=1.5)

locs, labels = plt.xticks(list(plt.xticks()[0]) + [100])
labels[-1].set_weight("bold")

plt.xlim([-2, 1000])
plt.ylim([0, 1.05])

plt.xlabel("Steps")
plt.ylabel("Learning Rate")

plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
