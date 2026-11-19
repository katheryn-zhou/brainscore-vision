# Scores a list of models on a list of benchmarks and saves the results.
#
# Results go to results/<benchmark_identifier>.json, one file per benchmark. Each file holds a
# dictionary keyed by model identifier, with the score, error, raw score, ceiling, layer
# commitment, model depth/width/seed and a timestamp.
#
# Put this file in your vision folder and run:  python run_benchmarks.py
# Runs that are already saved are skipped; add --force to redo them.

import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
import torch

from brainscore_vision import score as score_model, load_benchmark, load_model
from brainscore_vision.model_helpers.activations.pca import LayerPCA
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

MODELS = []
for seed in [0, 1, 2]: #0]:
    for depth in [4]:
        for width in [0.25, 0.5, 1, 2, 4, 8, 16, 32]:
            for method in ['e2e', 'greedy']:
                MODELS.append(f'{method}_conv{depth}_w{width}_s{seed}')

BENCHMARKS = [
    'FreemanZiemba2013public.V1-pls',
    'FreemanZiemba2013public.V2-pls',
    'MajajHong2015public.V4-pls',
    'MajajHong2015public.IT-pls',
]

RESULTS_DIR = Path('results')
WEIGHTS_DIR = Path(__file__).parent.parent /  'brainscore_vision' / 'models' / 'shallow_and_wide_models' / 'weights'

# Per-layer scores to save alongside the main score:
# - selection: the scores Brain-Score used to pick the best layer (quick, read from its cache)
# - benchmark: every layer scored on the benchmark itself, exactly like the main score (slower)
SAVE_SELECTION_LAYER_SCORES = True
SAVE_BENCHMARK_LAYER_SCORES = True

# the benchmarks Brain-Score uses to choose a layer for each region (from ModelCommitment)
SELECTION_BENCHMARKS = {
    'V1': 'FreemanZiemba2013public.V1-pls',
    'V2': 'FreemanZiemba2013public.V2-pls',
    'V4': 'MajajHong2015public.V4-pls',
    'IT': 'MajajHong2015public.IT-pls',
}


def to_float(value):
    # Brain-Score stores numbers as small xarray objects; turn them into plain floats
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(value.values.mean())
        except Exception:
            return None


def model_settings(model_identifier):
    # depth, width and seed: from the checkpoint if possible, otherwise from the name
    settings = {'method': None, 'depth': None, 'width': None, 'seed': None}
    # match = re.search(r'conv(\d+)_w(\d+(?:[.p]\d+)?)_s(\d+)', model_identifier)
    # if match:
    #     settings = {'learning_rule': None,
    #                 'depth': int(match.group(1)),
    #                 'width': float(match.group(2).replace('p', '.')),
    #                 'seed': int(match.group(3))}
    # try:
        # from brainscore_vision.models.shallow_e2e.model import load_checkpoint
        # configuration = load_checkpoint(model_identifier)['configuration']
        # settings['depth'] = configuration['depth']
        # settings['width'] = configuration['width']
    checkpoint = torch.load(WEIGHTS_DIR / f'{model_identifier}.pt', map_location='cpu')
    configuration = checkpoint['configuration']
    try:
        settings = {'method': configuration.get('method'),
                    'depth': configuration.get('depth'),
                    'width': configuration.get('width'),
                    'seed': configuration.get('seed')}
    except Exception:
        pass
    return settings


def selection_layer_scores(model, region):
    # Repeats Brain-Score's own layer-selection call (1000 PCA components, same cache keys),
    # so the scores it computed while choosing a layer are read back from the cache.
    region_layer_map = model.layer_model.region_layer_map
    if not hasattr(region_layer_map, 'layer_selection') or region not in region_layer_map.region_benchmarks:
        return None  # no automatic layer selection for this model or region
    selection = region_layer_map.layer_selection
    activations_model = selection._layer_scoring._activations_model
    handle = None
    if not LayerPCA.is_hooked(activations_model):
        handle = LayerPCA.hook(activations_model, n_components=1000)
        identifier = activations_model.identifier
        activations_model.identifier = identifier + '-pca_1000'
    try:
        layer_scores = selection._layer_scoring(benchmark=region_layer_map.region_benchmarks[region],
                                                benchmark_identifier=region, layers=selection.layers,
                                                prerun=True)
    finally:
        if handle is not None:
            handle.remove()
            activations_model.identifier = identifier
    return {str(layer): to_float(layer_scores.sel(layer=layer)) for layer in layer_scores['layer'].values}


def benchmark_layer_scores(model_identifier, model, benchmark):
    # Scores every layer on the benchmark itself, the same way the main score is computed
    scores = {}
    for layer in model.layers:
        layer_model = ModelCommitment(identifier=model_identifier, activations_model=model.activations_model,
                                      layers=model.layers, region_layer_map={benchmark.region: layer})
        scores[layer] = to_float(benchmark(layer_model))
        print(f'  {layer}: {scores[layer]:.4f}', flush=True)
    return scores


def layer_results(model_identifier, benchmark_identifier):
    results = {}
    benchmark = load_benchmark(benchmark_identifier)
    region = getattr(benchmark, 'region', None)
    model = load_model(model_identifier)
    if SAVE_SELECTION_LAYER_SCORES:
        results['selection_layer_scores'] = selection_layer_scores(model, region)
        results['selection_benchmark'] = SELECTION_BENCHMARKS.get(region)
    if SAVE_BENCHMARK_LAYER_SCORES:
        print('scoring each layer on the benchmark...', flush=True)
        results['layer_scores'] = benchmark_layer_scores(model_identifier, model, benchmark)
    return results


def load_results(benchmark_identifier):
    path = RESULTS_DIR / f'{benchmark_identifier}.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return {}


def save_results(benchmark_identifier, results):
    # write to a temporary file first, so an interrupted run can't corrupt the results file
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f'{benchmark_identifier}.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(results, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def main():
    force = '--force' in sys.argv
    for benchmark_identifier in BENCHMARKS:
        for model_identifier in MODELS:
            results = load_results(benchmark_identifier)
            if model_identifier in results and not force:
                if SAVE_BENCHMARK_LAYER_SCORES and 'layer_scores' not in results[model_identifier]:
                    print(f'\nper-layer scores missing for {model_identifier} on {benchmark_identifier}; rescoring')
                elif SAVE_SELECTION_LAYER_SCORES and 'selection_layer_scores' not in results[model_identifier]:
                    print(f'\nselection-layer scores missing for {model_identifier} on {benchmark_identifier}; rescoring')
                else:
                    print(f'skip  {model_identifier} on {benchmark_identifier} (already saved)')
                    continue
            print(f'\n=== {model_identifier} on {benchmark_identifier}', flush=True)
            try:
                settings = model_settings(model_identifier)
                score = score_model(model_identifier, benchmark_identifier)
            except Exception:
                traceback.print_exc()
                print(f'FAILED  {model_identifier} on {benchmark_identifier}, continuing')
                continue
            attrs = score.attrs
            results[model_identifier] = {
                'score': to_float(score),
                'error': to_float(attrs.get('error')),
                'raw': to_float(attrs.get('raw')),
                'ceiling': to_float(attrs.get('ceiling')),
                'model_identifier': model_identifier,
                'benchmark_identifier': benchmark_identifier,
                'comment': str(attrs.get('comment', '')),
                **settings,
                'timestamp': datetime.now().isoformat(timespec='seconds'),
            }
            # per-layer scores are extras: if they fail, the main score is still saved
            try:
                results[model_identifier].update(layer_results(model_identifier, benchmark_identifier))
            except Exception:
                traceback.print_exc()
                results[model_identifier]['layer_scores_error'] = traceback.format_exc(limit=1)
                print('per-layer scores failed; saving the main score anyway')
            save_results(benchmark_identifier, results)
            print(f"saved {model_identifier} on {benchmark_identifier}: score={results[model_identifier]['score']:.4f}")


if __name__ == '__main__':
    main()
