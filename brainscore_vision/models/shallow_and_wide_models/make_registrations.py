for seed in [0, 1, 2]:
    for depth in [4]:
        for width in [0.25, 0.5, 1, 2, 4, 8, 16, 32]:
            for method in ['e2e', 'greedy']:
                name = f'{method}_conv{depth}_w{width}_s{seed}'
                print(f"model_registry['{name}'] = lambda: ModelCommitment(identifier='{name}', "
                      f"activations_model=get_model('{name}'), layers=get_layers('{name}'))")