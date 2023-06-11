import numpy as np
import pandas as pd
import os
import torch
import datasets
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score


MODEL_N_LAYERS = {
    'pythia-70m': 6,
    'pythia-160m': 12,
    'pythia-410m': 24,
    'pythia-1b': 16,
    'pythia-1.4b': 24,
    'pythia-2.8b': 32,
    'pythia-6.9b': 32
}


def load_dataset(args):
    ds = datasets.load_from_disk(f'data/{args.feature_dataset}')
    label_key = 'language' if args.feature_dataset == 'europarl_lang' else 'distribution'
    labels = np.array(ds[label_key])

    all_acts = []
    for i in range(MODEL_N_LAYERS[args.model]):
        save_path = f'cached_activations/{args.model}/{args.feature_dataset}/{i}.mean.pt'
        acts = torch.load(save_path)
        acts = acts.dequantize().numpy()
        all_acts.append(acts)
    all_acts = np.concatenate(all_acts, axis=1)

    return all_acts, labels


def probe_context_neurons(args, activations, labels):
    d_mlp = activations.shape[1] // MODEL_N_LAYERS[args.model]
    unique_labels = np.unique(labels)

    results = []
    for ix, label in enumerate(unique_labels):
        distr_seqs = labels == label
        distr_acts = activations[distr_seqs].mean(axis=0)
        not_distr_acts = activations[~distr_seqs].mean(axis=0)
        distr_diff = distr_acts - not_distr_acts

        max_diff_neurons = np.argsort(np.abs(distr_diff))[-args.top_k:]

        for nix in max_diff_neurons:
            lr = LogisticRegression(class_weight='balanced')
            lr.fit(activations[:, nix].reshape(-1, 1), distr_seqs)
            preds = lr.predict(activations[:, nix].reshape(-1, 1))
            f1 = f1_score(distr_seqs, preds)
            precision = precision_score(distr_seqs, preds)
            recall = recall_score(distr_seqs, preds)

            results.append({
                'label': label,
                'layer': nix // d_mlp,
                'neuron': nix % d_mlp,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'mean_dif': distr_diff[nix],
            })

    result_df = pd.DataFrame(results).sort_values(
        'f1', ascending=False).reset_index(drop=True)

    return result_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', default='pythia-70m',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--feature_dataset',
        help='Name of feature dataset')
    parser.add_argument(
        '--activation_aggregation', default='mean',
        help='Average activations across all tokens in a sequence')
    parser.add_argument(
        '--output_dir', default='context_neurons')
    parser.add_argument(
        '--top_k', default=10, type=int,
        help='Number of neurons to probe')

    args = parser.parse_args()

    activations, labels = load_dataset(args)
    results = probe_context_neurons(args, activations, labels)

    save_path = os.path.join(args.output_dir, args.model)
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f'{args.feature_dataset}_neurons.csv')
    results.to_csv(save_file, index=False)
