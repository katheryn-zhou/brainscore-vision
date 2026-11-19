from brainscore_vision.model_helpers.check_submission import check_models
import functools
from pathlib import Path
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
# from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

import torch
from torch import nn

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.


# from seeding import model_seed

# CIFAR-10 normalization used in training -- these must match the Normalize() in your training code
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Every model in this plugin. Each one needs a matching <name>.pt file in this folder
# and its own registration line in __init__.py.
MODEL_NAMES = []
for width in [0.25,0.5,1,2,4,8,16,32]:
    for depth in [4]:
        for seed in [0,1,2]:
            MODEL_NAMES.append(f'e2e_conv{depth}_w{width}_s{seed}')
            MODEL_NAMES.append(f'greedy_conv{depth}_w{width}_s{seed}')

# the input size the models were trained on
IMAGE_SIZE = 32
 
# weight files live next to this file, so they are found no matter which folder you run from
WEIGHTS_DIR = Path(__file__).parent / 'weights'

# base channel counts and downsampling steps for the two network depths
CHANNELS = {4: (32, 64, 128, 256), 8: (32, 32, 64, 64, 128, 128, 256, 256)}
STRIDES = {4: (1, 2, 2, 2), 8: (1, 1, 2, 1, 2, 1, 2, 1)}


class Encoder(nn.Module):
    # stack convolution, batch norm, and relu, then pool the spatial dimensions

    def __init__(self, depth, width):
        super().__init__()
        self.depth = depth
        # widen every layer by the same multiplier
        self.channels = [round(c * width) for c in CHANNELS[depth]]
        self.layers = nn.ModuleList()
        input_channels = 3
        for channels, stride in zip(self.channels, STRIDES[depth]):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, channels, 3, stride, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )
            input_channels = channels

        # Give each block's ReLU a name Brain-Score can find: relu1, relu2, ... reluN.
        # These are extra names for the ReLU modules already inside self.layers, not new modules.
        # ReLU has no weights, so the state_dict and the forward pass are unchanged.
        for index, block in enumerate(self.layers, start=1):
            setattr(self, f'relu{index}', block[2])

    def forward(self, images, stop_layer=None, greedy=False):
        # stop at the active layer during greedy training, or use the full network
        stop_layer = self.depth if stop_layer is None else stop_layer
        maps = images
        if greedy:
            # pass through the frozen layers without tracking their gradients
            with torch.no_grad():
                for layer in self.layers[: stop_layer - 1]:
                    maps = layer(maps)
            # track gradients for the layer we're training now
            maps = self.layers[stop_layer - 1](maps)
        else:
            for layer in self.layers[:stop_layer]:
                maps = layer(maps)
        # average each feature map into one value per channel
        return maps.mean(dim=(2, 3))
    

@functools.lru_cache(maxsize=None)
def load_checkpoint(name):
    # load each checkpoint once, on the CPU, from this folder
    return torch.load(WEIGHTS_DIR / f'{name}.pt', map_location='cpu')


def build_and_load_encoder(name):
    checkpoint = load_checkpoint(name)
    configuration = checkpoint['configuration']
    model = Encoder(depth=configuration['depth'], width=configuration['width'])
    print(f"Model configuration: depth={configuration['depth']}, width={configuration['width']}")
    model.load_state_dict(checkpoint['state_dict'])
    return model
 
 
def get_model_list():
    return MODEL_NAMES


def get_model(name):
    assert name in MODEL_NAMES
    model = build_and_load_encoder(name)
    # preprocessing = functools.partial(load_preprocess_images, image_size=IMAGE_SIZE)
    preprocessing = functools.partial(load_preprocess_images, image_size=IMAGE_SIZE,
                                  normalize_mean=CIFAR10_MEAN, normalize_std=CIFAR10_STD)
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = IMAGE_SIZE
    return wrapper


def get_layers(name):
    assert name in MODEL_NAMES
    # relu1 ... relu4 for depth-4 models, relu1 ... relu8 for depth-8 models
    depth = load_checkpoint(name)['configuration']['depth']
    return [f'relu{index}' for index in range(1, depth + 1)]


def get_bibtex(model_identifier):
    return """"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
