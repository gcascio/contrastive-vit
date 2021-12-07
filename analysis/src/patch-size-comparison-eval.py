import numpy as np
import matplotlib.pyplot as plt


BASE_PATH = '~/HHU/Bachelor/training_results/main_runs/'

MODEL_SIZE = 't'
MODELS = [
    f'imagenet_64_vit-{MODEL_SIZE}_patch8_image64',
    f'imagenet_64_vit-{MODEL_SIZE}_patch16_image64',
    f'imagenet_64_vit-{MODEL_SIZE}_patch32_image64',
]
MODEL_TITLES = ['ViT-S', 'ViT-Ti', 'ViT-Xt']
PATCH_SIZES = [8, 16, 32]


def load_data(size: str, path: str, dataset: str, postfix=''):
    out = []
    models = [
        f'imagenet_64_vit-{size}_patch8_image64',
        f'imagenet_64_vit-{size}_patch16_image64',
        f'imagenet_64_vit-{size}_patch32_image64',
    ]

    for model in models:
        file_path = f'{BASE_PATH}{model}/{path}/{dataset}{postfix}.csv'
        _, x, y = np.genfromtxt(
            file_path,
            delimiter=',',
            skip_header=1,
            unpack=True,
        )

        out.append((x, y))

    return out


results = {}
for i, size in enumerate(['s', 't', 'xt']):
    patch_runs = load_data(
        size, '300ep/accuracy', 'imagenet-64'
    )

    for j, patch_run in enumerate(patch_runs):
        key = f'{MODEL_TITLES[i]}'

        mean = np.mean(patch_run[1][-10:])
        std = np.std(patch_run[1][-10:])

        if key not in results:
            results[key] = []

        results[key].append((PATCH_SIZES[j], mean, std))

plt.errorbar(*zip(*results['ViT-S']), linestyle='None', fmt='-^',
             capsize=8, ecolor='black', label='ViT-S')
plt.errorbar(*zip(*results['ViT-Ti']), linestyle='None', fmt='-o',
             capsize=8, ecolor='black', label='ViT-Ti')
plt.errorbar(*zip(*results['ViT-Xt']), linestyle='None', fmt='-s',
             capsize=8, ecolor='black', label='ViT-Xt')

plt.xlabel('Patch Size [px]')
plt.ylabel('Validation Accuracy [%]')
plt.xticks([8, 16, 24, 32])
plt.xlim([5, 35])
plt.legend()
plt.show()
