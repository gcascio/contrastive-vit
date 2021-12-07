import numpy as np
import matplotlib.pyplot as plt


BASE_PATH = '~/HHU/Bachelor/training_results/grid_search/'
LR = ['0.0001', '0.0004', '0.001', '0.004']
LR_TITLES = ['lr: 1e-4', 'lr: 4e-4', 'lr: 1e-3', 'lr: 4e-3']
RUNS = [
    'wd:0.001_do:0',
    'wd:0.001_do:0.1',
    'wd:0.001_do:0.3',
]


def load_data(path: str, lr: str, postfix=''):
    out = []
    root = f'{BASE_PATH}{path}/lr:{lr}_'

    for run in RUNS:
        _, x, y = np.genfromtxt(
            f'{root}{run}{postfix}.csv',
            delimiter=',',
            skip_header=1,
            unpack=True,
        )

        out.append((x, y))

    return out


fig, ax = plt.subplots(nrows=4, ncols=2)
fig.tight_layout()
fig.set_size_inches(8, 10)

for i, lr in enumerate(LR):
    wd0001_do0, wd0001_do01, wd0001_do03 = load_data(
        f'runs0{i+1}/accuracy', lr
    )

    wd0001_do0_l, wd0001_do01_l, wd0001_do03_l = load_data(
        f'runs0{i+1}/loss', lr, '-loss'
    )

    ax[i][0].plot(wd0001_do0[0], wd0001_do0[1], label='wd: 1e-3, do: 0.0')
    ax[i][0].plot(wd0001_do01[0], wd0001_do01[1], label='wd: 1e-3, do: 0.1')
    ax[i][0].plot(wd0001_do03[0], wd0001_do03[1], label='wd: 1e-3, do: 0.3')

    ax[i][0].set_title(LR_TITLES[i])
    ax[i][0].set_xlabel("Epoch")
    ax[i][0].set_ylabel("Val. Accuracy [%]")

    ax[i][0].set_xlim([0, 150])
    ax[i][0].set_ylim([0, 32])

    if i == 1:
        ax[i][0].legend(loc='best', prop={'size': 9}, ncol=1,
                        handletextpad=0.4, handlelength=1, columnspacing=1,
                        borderpad=0.3, borderaxespad=0.3, labelspacing=0.3)
    else:
        ax[i][0].legend(loc='best', prop={'size': 9}, ncol=2,
                        handletextpad=0.4, handlelength=1, columnspacing=1,
                        borderpad=0.3, borderaxespad=0.3, labelspacing=0.3)

    ax[i][1].plot(wd0001_do0_l[0], wd0001_do0_l[1], label='wd: 1e-3, do: 0.0')
    ax[i][1].plot(wd0001_do01_l[0], wd0001_do01_l[1],
                  label='wd: 1e-3, do: 0.1')
    ax[i][1].plot(wd0001_do03_l[0], wd0001_do03_l[1],
                  label='wd: 1e-3, do: 0.3')

    ax[i][1].set_title(LR_TITLES[i])
    ax[i][1].set_xlabel("Epoch")
    ax[i][1].set_ylabel("Training Loss")

    ax[i][1].set_xlim([0, 150])
    ax[i][1].set_ylim([0, 6])

    ax[i][1].legend(loc='best', prop={'size': 9}, ncol=2, handletextpad=0.4,
                    handlelength=1, columnspacing=1, borderpad=0.3,
                    borderaxespad=0.3, labelspacing=0.3)


plt.show()
