import numpy as np


BASE_PATH = '~/HHU/Bachelor/training_results/grid_search/'
LR = ['0.0001', '0.0004', '0.001', '0.004']
LR_TITLES = ['lr: 1e-4', 'lr: 4e-4', 'lr: 1e-3', 'lr: 4e-3']
RUNS = [
    'wd:0_do:0',
    'wd:0_do:0.1',
    'wd:0_do:0.3',
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


results = {}

for i, lr in enumerate(LR):
    runs = load_data(
        f'runs0{i+1}/accuracy', lr
    )

    for i, run in enumerate(runs):
        key = f'lr{lr}_{RUNS[i]}'

        mean = np.mean(run[1][-10:])
        std = np.std(run[1][-10:])

        results[key] = (mean, std)

best_result = None
best_mean = 0.
for key in results:
    if results[key][0] > best_mean:
        best_mean = results[key][0]
        best_result = {
            key: results[key]
        }

print(best_result)
