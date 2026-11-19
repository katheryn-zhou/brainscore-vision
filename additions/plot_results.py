# Plots benchmark score against model width for one model depth, from the saved results.
#
# Run from your vision folder, e.g.:
#   python plot_results.py MajajHong2015public.IT-pls 4
# The plot is saved to plots/<benchmark>_depth<depth>.png and also shown on screen.
# If several seeds of the same width are saved, it plots their mean, with error bars
# showing the standard deviation across seeds.

import json
import sys
from collections import defaultdict
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path('results')
PLOTS_DIR = Path('plots')


def main():
    # if len(sys.argv) != 3:
    #     print('usage: python plot_results.py <benchmark_identifier> <depth>')
    #     sys.exit(1)
    # benchmark_identifier, depth = sys.argv[1], int(sys.argv[2])

    for BENCHMARK_IDENTIFIER in ['FreemanZiemba2013public.V1-pls', 
                                 'FreemanZiemba2013public.V2-pls', 
                                 'MajajHong2015public.V4-pls', 
                                 'MajajHong2015public.IT-pls']:
        print(f'\n===== {BENCHMARK_IDENTIFIER} =====')
        DEPTH = 4

        figure, axes = plt.subplots(figsize=(6, 4))

        results = json.loads((RESULTS_DIR / f'{BENCHMARK_IDENTIFIER}.json').read_text(encoding='utf-8'))

        for method in ['e2e', 'greedy']:
            scores_by_width = defaultdict(list)
            for model_identifier, result in results.items():
                if result.get('depth') == DEPTH and result.get('method') == method and result.get('width') is not None and result.get('score') is not None:
                    scores_by_width[result['width']].append(result['score'])
            if not scores_by_width:
                print(f'no saved results for depth {DEPTH} on {BENCHMARK_IDENTIFIER}')
                sys.exit(1)

            widths = sorted(scores_by_width)
            means = [np.mean(scores_by_width[w]) for w in widths]
            spreads = [np.std(scores_by_width[w]) if len(scores_by_width[w]) > 1 else 0 for w in widths]
            for w, m in zip(widths, means):
                print(f'width {w:g}: score {m:.4f}  ({len(scores_by_width[w])} model(s))')

            axes.errorbar(widths, means, yerr=spreads, marker='o', capsize=3, label=f'{method} depth {DEPTH}')

        axes.set_xscale('log', base=2)  # widths double each step, so space them evenly
        axes.set_xticks(widths)
        axes.set_xticklabels([f'{w:g}' for w in widths])
        axes.set_xlabel('model width multiplier')
        axes.set_ylabel('score')
        axes.set_title(f'{BENCHMARK_IDENTIFIER} vs width, depth {DEPTH} models (n=3)', fontsize=11)
        axes.legend()
        figure.tight_layout()

        PLOTS_DIR.mkdir(exist_ok=True)
        output = PLOTS_DIR / f'{BENCHMARK_IDENTIFIER}_depth{DEPTH}.png'
        figure.savefig(output, dpi=150)
        print(f'saved {output}')
        # plt.show()

        # plot the per-layer scores for each method
        # e2e will have circles and solid lines, greedy will have squares and dashed lines
        # models with different widths will have different colors, but the same color for both methods. the colors will increase in brightness with width.
        # There will be one line for each model width and method, with error bars showing the standard deviation across seeds.
        # the x-axis is the layer index (1,2,3,4), and the y-axis is the score. The legend will show the method and width for each line.

        figure, axes = plt.subplots(figsize=(10, 4))
        scores_by_width = defaultdict(list)
        for method in ['e2e', 'greedy']:
            for model_identifier, result in results.items():
                # print(model_identifier, result['method'])
                if result.get('depth') == DEPTH and result.get('method') == method and result.get('width') is not None and result.get('score') is not None:
                    # print(model_identifier, result['method'])
                    if "layer_scores" not in result.keys() and "selection_layer_scores" not in result.keys():
                        print(f'no layer scores for {model_identifier} on {BENCHMARK_IDENTIFIER}')
                        continue
                    elif "layer_scores" in result.keys():
                        key = 'layer_scores'
                    elif "selection_layer_scores" in result.keys():
                        key = 'selection_layer_scores'
                    scores_by_width['depth'].append(result['depth'])
                    scores_by_width['method'].append(result['method'])
                    scores_by_width['width'].append(result['width'])
                    scores_by_width['seed'].append(result['seed'])
                    scores_by_width['score_type'].append(key)
                    layer_scores = result[key] # list of scores, one for each layer
                    for layer_name in layer_scores.keys():
                        scores_by_width[f'{layer_name}'].append(layer_scores[layer_name])
        scores_df = pd.DataFrame(scores_by_width)

        # plot the per-layer scores for each method and width, with error bars showing the standard deviation across seeds
        for method in ['e2e', 'greedy']:
            for width in sorted(scores_df['width'].unique()):
                subset = scores_df[(scores_df['method'] == method) & (scores_df['width'] == width)]
                # display(subset)
                if subset.empty:
                    continue
                means = subset.groupby('method').mean(numeric_only=True)
                stds = subset.groupby('method').std(numeric_only=True)
                layer_names = [f'relu{layer_i}' for layer_i in range(1, DEPTH+1)]
                means = means[layer_names]
                stds = stds[layer_names]
                # plot the means and stds for each score type (layer_scores or selection_layer_scores)
                # the color will be the same for both methods, but different for each width. The color will be a shade of blue for e2e and a shade of orange for greedy, with the brightness increasing with width.
                if method == 'e2e':
                    color = (0, 0, 1, 0.2 + 0.8 * (width / max(scores_df['width']))) # blue with alpha increasing with width
                else:
                    color = (1, 0.5, 0, 0.2 + 0.8 * (width / max(scores_df['width']))) # orange with alpha increasing with width
                for score_type in means.index:
                    axes.errorbar(range(1, len(means.columns)+1), means.loc[score_type], yerr=stds.loc[score_type], marker='o' if method == 'e2e' else 's', 
                                  linestyle='-', label=f'{method} width {width} {score_type}',
                                  color=color)
        plt.xticks(range(1, DEPTH+1), [f'relu{layer_i}' for layer_i in range(1, DEPTH+1)])
        plt.xlabel('layer')
        plt.ylabel('score')
        plt.title(f'{BENCHMARK_IDENTIFIER} per-layer scores, depth {DEPTH} models (n=3)', fontsize=11)
        # move legend outside of plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.legend()
        figure.tight_layout()
        # save plot
        output = PLOTS_DIR / f'{BENCHMARK_IDENTIFIER}_depth{DEPTH}_per_layer.png'
        figure.savefig(output, dpi=150)
        print(f'saved {output}')
        

    # display(scores_df)
    print(scores_df.shape)


if __name__ == '__main__':
    main()
