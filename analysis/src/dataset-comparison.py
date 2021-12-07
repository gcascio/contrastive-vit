import numpy as np
import matplotlib.pyplot as plt


BASE_PATH = '~/HHU/Bachelor/training_results/main_runs/'

MODEL_SIZE = 't'
MODELS = [
    f'imagenet_64_vit-{MODEL_SIZE}_patch8_image64',
]
MODEL_TITLES = ['ViT-S', 'ViT-Ti', 'ViT-Xt']


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


fig, ax = plt.subplots(nrows=3)
fig.set_size_inches(8, 10)

for i, size in enumerate(['s', 't', 'xt']):
    patch8 = load_data(
        size, '300ep/accuracy', 'imagenet-64'
    )[0]

    patch8_tiny = load_data(
        size, '300ep/accuracy', 'tiny-imagenet'
    )[0]

    ax[i].plot(patch8[0], patch8[1], label='Imagenet-64')
    ax[i].plot(patch8_tiny[0], patch8_tiny[1], label='Tiny-Imagenet')

    ax[i].set_title(MODEL_TITLES[i])
    ax[i].set_xlabel("Epoch")
    ax[i].set_ylabel("Validation Accuracy [%]")

    ax[i].set_xlim([0, 300])
    ax[i].set_ylim([0, 54])

    ax[i].legend(loc='lower right', prop={'size': 9}, borderpad=0.3,
                 handlelength=1, handletextpad=0.4, columnspacing=1,
                 borderaxespad=0.3, labelspacing=0.3)

fig.tight_layout()
plt.subplots_adjust(hspace=0.5)

plt.show()
