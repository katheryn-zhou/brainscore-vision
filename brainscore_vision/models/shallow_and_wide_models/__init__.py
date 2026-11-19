
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

# model_registry['e2e_conv4_w0.25_s0'] = lambda: ModelCommitment(identifier='e2e_conv4_w0.25_s0',
#                                                                     activations_model=get_model('e2e_conv4_w0.25_s0'),
#                                                                     layers=get_layers('e2e_conv4_w0.25_s0'))

# for width in [1,2,4]:
#     for depth in [4]:
#         for seed in [0]:
#             model_registry[f'e2e_conv{depth}_w{width}_s{seed}'] = lambda depth=depth, width=width, seed=seed: ModelCommitment(
#                 identifier=f'e2e_conv{depth}_w{width}_s{seed}',
#                 activations_model=get_model(f'e2e_conv{depth}_w{width}_s{seed}'),
#                 layers=get_layers(f'e2e_conv{depth}_w{width}_s{seed}')
#             )
#             model_registry[f'greedy_conv{depth}_w{width}_s{seed}'] = lambda depth=depth, width=width, seed=seed: ModelCommitment(
#                 identifier=f'greedy_conv{depth}_w{width}_s{seed}',
#                 activations_model=get_model(f'greedy_conv{depth}_w{width}_s{seed}'),
#                 layers=get_layers(f'greedy_conv{depth}_w{width}_s{seed}')

model_registry['e2e_conv4_w0.25_s0'] = lambda: ModelCommitment(identifier='e2e_conv4_w0.25_s0', activations_model=get_model('e2e_conv4_w0.25_s0'), layers=get_layers('e2e_conv4_w0.25_s0'))
model_registry['greedy_conv4_w0.25_s0'] = lambda: ModelCommitment(identifier='greedy_conv4_w0.25_s0', activations_model=get_model('greedy_conv4_w0.25_s0'), layers=get_layers('greedy_conv4_w0.25_s0'))
model_registry['e2e_conv4_w0.5_s0'] = lambda: ModelCommitment(identifier='e2e_conv4_w0.5_s0', activations_model=get_model('e2e_conv4_w0.5_s0'), layers=get_layers('e2e_conv4_w0.5_s0'))
model_registry['greedy_conv4_w0.5_s0'] = lambda: ModelCommitment(identifier='greedy_conv4_w0.5_s0', activations_model=get_model('greedy_conv4_w0.5_s0'), layers=get_layers('greedy_conv4_w0.5_s0'))
model_registry['e2e_conv4_w1_s0'] = lambda: ModelCommitment(identifier='e2e_conv4_w1_s0', activations_model=get_model('e2e_conv4_w1_s0'), layers=get_layers('e2e_conv4_w1_s0'))
model_registry['greedy_conv4_w1_s0'] = lambda: ModelCommitment(identifier='greedy_conv4_w1_s0', activations_model=get_model('greedy_conv4_w1_s0'), layers=get_layers('greedy_conv4_w1_s0'))
model_registry['e2e_conv4_w2_s0'] = lambda: ModelCommitment(identifier='e2e_conv4_w2_s0', activations_model=get_model('e2e_conv4_w2_s0'), layers=get_layers('e2e_conv4_w2_s0'))
model_registry['greedy_conv4_w2_s0'] = lambda: ModelCommitment(identifier='greedy_conv4_w2_s0', activations_model=get_model('greedy_conv4_w2_s0'), layers=get_layers('greedy_conv4_w2_s0'))
model_registry['e2e_conv4_w4_s0'] = lambda: ModelCommitment(identifier='e2e_conv4_w4_s0', activations_model=get_model('e2e_conv4_w4_s0'), layers=get_layers('e2e_conv4_w4_s0'))
model_registry['greedy_conv4_w4_s0'] = lambda: ModelCommitment(identifier='greedy_conv4_w4_s0', activations_model=get_model('greedy_conv4_w4_s0'), layers=get_layers('greedy_conv4_w4_s0'))
model_registry['e2e_conv4_w8_s0'] = lambda: ModelCommitment(identifier='e2e_conv4_w8_s0', activations_model=get_model('e2e_conv4_w8_s0'), layers=get_layers('e2e_conv4_w8_s0'))
model_registry['greedy_conv4_w8_s0'] = lambda: ModelCommitment(identifier='greedy_conv4_w8_s0', activations_model=get_model('greedy_conv4_w8_s0'), layers=get_layers('greedy_conv4_w8_s0'))
model_registry['e2e_conv4_w16_s0'] = lambda: ModelCommitment(identifier='e2e_conv4_w16_s0', activations_model=get_model('e2e_conv4_w16_s0'), layers=get_layers('e2e_conv4_w16_s0'))
model_registry['greedy_conv4_w16_s0'] = lambda: ModelCommitment(identifier='greedy_conv4_w16_s0', activations_model=get_model('greedy_conv4_w16_s0'), layers=get_layers('greedy_conv4_w16_s0'))
model_registry['e2e_conv4_w32_s0'] = lambda: ModelCommitment(identifier='e2e_conv4_w32_s0', activations_model=get_model('e2e_conv4_w32_s0'), layers=get_layers('e2e_conv4_w32_s0'))
model_registry['greedy_conv4_w32_s0'] = lambda: ModelCommitment(identifier='greedy_conv4_w32_s0', activations_model=get_model('greedy_conv4_w32_s0'), layers=get_layers('greedy_conv4_w32_s0'))
model_registry['e2e_conv4_w0.25_s1'] = lambda: ModelCommitment(identifier='e2e_conv4_w0.25_s1', activations_model=get_model('e2e_conv4_w0.25_s1'), layers=get_layers('e2e_conv4_w0.25_s1'))
model_registry['greedy_conv4_w0.25_s1'] = lambda: ModelCommitment(identifier='greedy_conv4_w0.25_s1', activations_model=get_model('greedy_conv4_w0.25_s1'), layers=get_layers('greedy_conv4_w0.25_s1'))
model_registry['e2e_conv4_w0.5_s1'] = lambda: ModelCommitment(identifier='e2e_conv4_w0.5_s1', activations_model=get_model('e2e_conv4_w0.5_s1'), layers=get_layers('e2e_conv4_w0.5_s1'))
model_registry['greedy_conv4_w0.5_s1'] = lambda: ModelCommitment(identifier='greedy_conv4_w0.5_s1', activations_model=get_model('greedy_conv4_w0.5_s1'), layers=get_layers('greedy_conv4_w0.5_s1'))
model_registry['e2e_conv4_w1_s1'] = lambda: ModelCommitment(identifier='e2e_conv4_w1_s1', activations_model=get_model('e2e_conv4_w1_s1'), layers=get_layers('e2e_conv4_w1_s1'))
model_registry['greedy_conv4_w1_s1'] = lambda: ModelCommitment(identifier='greedy_conv4_w1_s1', activations_model=get_model('greedy_conv4_w1_s1'), layers=get_layers('greedy_conv4_w1_s1'))
model_registry['e2e_conv4_w2_s1'] = lambda: ModelCommitment(identifier='e2e_conv4_w2_s1', activations_model=get_model('e2e_conv4_w2_s1'), layers=get_layers('e2e_conv4_w2_s1'))
model_registry['greedy_conv4_w2_s1'] = lambda: ModelCommitment(identifier='greedy_conv4_w2_s1', activations_model=get_model('greedy_conv4_w2_s1'), layers=get_layers('greedy_conv4_w2_s1'))
model_registry['e2e_conv4_w4_s1'] = lambda: ModelCommitment(identifier='e2e_conv4_w4_s1', activations_model=get_model('e2e_conv4_w4_s1'), layers=get_layers('e2e_conv4_w4_s1'))
model_registry['greedy_conv4_w4_s1'] = lambda: ModelCommitment(identifier='greedy_conv4_w4_s1', activations_model=get_model('greedy_conv4_w4_s1'), layers=get_layers('greedy_conv4_w4_s1'))
model_registry['e2e_conv4_w8_s1'] = lambda: ModelCommitment(identifier='e2e_conv4_w8_s1', activations_model=get_model('e2e_conv4_w8_s1'), layers=get_layers('e2e_conv4_w8_s1'))
model_registry['greedy_conv4_w8_s1'] = lambda: ModelCommitment(identifier='greedy_conv4_w8_s1', activations_model=get_model('greedy_conv4_w8_s1'), layers=get_layers('greedy_conv4_w8_s1'))
model_registry['e2e_conv4_w16_s1'] = lambda: ModelCommitment(identifier='e2e_conv4_w16_s1', activations_model=get_model('e2e_conv4_w16_s1'), layers=get_layers('e2e_conv4_w16_s1'))
model_registry['greedy_conv4_w16_s1'] = lambda: ModelCommitment(identifier='greedy_conv4_w16_s1', activations_model=get_model('greedy_conv4_w16_s1'), layers=get_layers('greedy_conv4_w16_s1'))
model_registry['e2e_conv4_w32_s1'] = lambda: ModelCommitment(identifier='e2e_conv4_w32_s1', activations_model=get_model('e2e_conv4_w32_s1'), layers=get_layers('e2e_conv4_w32_s1'))
model_registry['greedy_conv4_w32_s1'] = lambda: ModelCommitment(identifier='greedy_conv4_w32_s1', activations_model=get_model('greedy_conv4_w32_s1'), layers=get_layers('greedy_conv4_w32_s1'))
model_registry['e2e_conv4_w0.25_s2'] = lambda: ModelCommitment(identifier='e2e_conv4_w0.25_s2', activations_model=get_model('e2e_conv4_w0.25_s2'), layers=get_layers('e2e_conv4_w0.25_s2'))
model_registry['greedy_conv4_w0.25_s2'] = lambda: ModelCommitment(identifier='greedy_conv4_w0.25_s2', activations_model=get_model('greedy_conv4_w0.25_s2'), layers=get_layers('greedy_conv4_w0.25_s2'))
model_registry['e2e_conv4_w0.5_s2'] = lambda: ModelCommitment(identifier='e2e_conv4_w0.5_s2', activations_model=get_model('e2e_conv4_w0.5_s2'), layers=get_layers('e2e_conv4_w0.5_s2'))
model_registry['greedy_conv4_w0.5_s2'] = lambda: ModelCommitment(identifier='greedy_conv4_w0.5_s2', activations_model=get_model('greedy_conv4_w0.5_s2'), layers=get_layers('greedy_conv4_w0.5_s2'))
model_registry['e2e_conv4_w1_s2'] = lambda: ModelCommitment(identifier='e2e_conv4_w1_s2', activations_model=get_model('e2e_conv4_w1_s2'), layers=get_layers('e2e_conv4_w1_s2'))
model_registry['greedy_conv4_w1_s2'] = lambda: ModelCommitment(identifier='greedy_conv4_w1_s2', activations_model=get_model('greedy_conv4_w1_s2'), layers=get_layers('greedy_conv4_w1_s2'))
model_registry['e2e_conv4_w2_s2'] = lambda: ModelCommitment(identifier='e2e_conv4_w2_s2', activations_model=get_model('e2e_conv4_w2_s2'), layers=get_layers('e2e_conv4_w2_s2'))
model_registry['greedy_conv4_w2_s2'] = lambda: ModelCommitment(identifier='greedy_conv4_w2_s2', activations_model=get_model('greedy_conv4_w2_s2'), layers=get_layers('greedy_conv4_w2_s2'))
model_registry['e2e_conv4_w4_s2'] = lambda: ModelCommitment(identifier='e2e_conv4_w4_s2', activations_model=get_model('e2e_conv4_w4_s2'), layers=get_layers('e2e_conv4_w4_s2'))
model_registry['greedy_conv4_w4_s2'] = lambda: ModelCommitment(identifier='greedy_conv4_w4_s2', activations_model=get_model('greedy_conv4_w4_s2'), layers=get_layers('greedy_conv4_w4_s2'))
model_registry['e2e_conv4_w8_s2'] = lambda: ModelCommitment(identifier='e2e_conv4_w8_s2', activations_model=get_model('e2e_conv4_w8_s2'), layers=get_layers('e2e_conv4_w8_s2'))
model_registry['greedy_conv4_w8_s2'] = lambda: ModelCommitment(identifier='greedy_conv4_w8_s2', activations_model=get_model('greedy_conv4_w8_s2'), layers=get_layers('greedy_conv4_w8_s2'))
model_registry['e2e_conv4_w16_s2'] = lambda: ModelCommitment(identifier='e2e_conv4_w16_s2', activations_model=get_model('e2e_conv4_w16_s2'), layers=get_layers('e2e_conv4_w16_s2'))
model_registry['greedy_conv4_w16_s2'] = lambda: ModelCommitment(identifier='greedy_conv4_w16_s2', activations_model=get_model('greedy_conv4_w16_s2'), layers=get_layers('greedy_conv4_w16_s2'))
model_registry['e2e_conv4_w32_s2'] = lambda: ModelCommitment(identifier='e2e_conv4_w32_s2', activations_model=get_model('e2e_conv4_w32_s2'), layers=get_layers('e2e_conv4_w32_s2'))
model_registry['greedy_conv4_w32_s2'] = lambda: ModelCommitment(identifier='greedy_conv4_w32_s2', activations_model=get_model('greedy_conv4_w32_s2'), layers=get_layers('greedy_conv4_w32_s2'))