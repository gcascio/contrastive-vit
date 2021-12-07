import numpy as np
import matplotlib.pyplot as plt


BASE_PATH = '~/HHU/Bachelor/training_results/main_runs/'

MODEL_SIZE = 't'
MODELS = [
    f'imagenet_64_vit-{MODEL_SIZE}_patch8_image64',
    f'imagenet_64_vit-{MODEL_SIZE}_patch16_image64',
    f'imagenet_64_vit-{MODEL_SIZE}_patch32_image64',
]


def load_data(path: str, dataset: str, postfix=''):
    out = []

    for model in MODELS:
        file_path = f'{BASE_PATH}{model}/{path}/{dataset}{postfix}.csv'
        _, x, y = np.genfromtxt(
            file_path,
            delimiter=',',
            skip_header=1,
            unpack=True,
        )

        out.append((x, y))

    return out


fig, ax = plt.subplots(nrows=2)

patch8, patch16, patch32 = load_data(
    '300ep/accuracy', 'imagenet-64'
)

patch8_tiny, patch16_tiny, patch32_tiny = load_data(
    '300ep/accuracy', 'tiny-imagenet'
)

ax[0].plot(patch8[0], patch8[1], label='Patch size 8px')
ax[0].plot(patch16[0], patch16[1], label='Patch size 16px')
ax[0].plot(patch32[0], patch32[1], label='Patch size 32px')

ax[0].set_title('Imagenet-64')
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Validation Accuracy [%]")

ax[0].legend(loc='best', prop={'size': 9}, handletextpad=0.4, handlelength=1,
             columnspacing=1, borderpad=0.3, borderaxespad=0.3,
             labelspacing=0.3)

ax[1].plot(patch8_tiny[0], patch8_tiny[1], label='Patch size 8px')
ax[1].plot(patch16_tiny[0], patch16_tiny[1], label='Patch size 16px')
ax[1].plot(patch32_tiny[0], patch32_tiny[1], label='Patch size 32px')

ax[1].set_title('Tiny-Imagenet')
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Validation Accuracy [%]")

ax[1].legend(loc='best', prop={'size': 9}, handletextpad=0.4, handlelength=1,
             columnspacing=1, borderpad=0.3, borderaxespad=0.3,
             labelspacing=0.3)

fig.tight_layout()
plt.show()
