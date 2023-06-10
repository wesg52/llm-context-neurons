import os
import time
import tqdm
import torch
import einops
import datasets
import argparse
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer


def quantize_neurons(activation_tensor, output_precision=8):
    activation_tensor = activation_tensor.to(torch.float32)
    min_vals = activation_tensor.min(dim=0)[0]
    max_vals = activation_tensor.max(dim=0)[0]
    num_quant_levels = 2**output_precision
    scale = (max_vals - min_vals) / (num_quant_levels - 1)
    zero_point = torch.round(-min_vals / scale)
    return torch.quantize_per_channel(
        activation_tensor, scale, zero_point, 1, torch.qint8)


def save_activation(tensor, hook):
    hook.ctx['activation'] = tensor.detach().cpu().to(torch.float16)


def get_activations(args, model, dataset, device):
    # preallocate memory for activations
    n, d = dataset['tokens'].shape
    n_layers = model.cfg.n_layers
    activation_rows = n \
        if args.activation_aggregation is not None \
        else n * d
    layer_activations = {
        l: torch.zeros(activation_rows, model.cfg.d_mlp, dtype=torch.float16)
        for l in range(n_layers)
    }

    # define hooks to save activations from each layer
    hooks = [
        (f'blocks.{layer_ix}.{args.save_location}', save_activation)
        for layer_ix in range(n_layers)
    ]

    dataloader = DataLoader(
        dataset['tokens'], batch_size=args.batch_size, shuffle=False)

    offset = 0
    for step, batch in enumerate(tqdm.tqdm(dataloader)):
        model.run_with_hooks(
            batch.to(device),
            fwd_hooks=hooks,
        )

        for lix, (hook_pt, _) in enumerate(hooks):
            batch_activations = model.hook_dict[hook_pt].ctx['activation']
            if args.activation_aggregation == 'mean':
                batch_activations = batch_activations.mean(dim=1)
            elif args.activation_aggregation == 'max':
                batch_activations = batch_activations.max(dim=1).values
            else:
                batch_activations = einops.rearrange(
                    batch_activations, 'b c d -> (b c) d')
            layer_activations[lix][offset:offset +
                                   batch_activations.shape[0]] = batch_activations

        offset += batch_activations.shape[0]
        model.reset_hooks()

    # save activations
    save_path = os.path.join(
        args.output_dir,
        args.model,
        args.feature_dataset
    )
    os.makedirs(save_path, exist_ok=True)
    agg = 'none' if args.activation_aggregation is None else args.activation_aggregation
    for layer_ix, activations in layer_activations.items():
        torch.save(
            quantize_neurons(activations),
            os.path.join(save_path, f'{layer_ix}.{agg}.pt')
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', default='pythia-70m',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--feature_dataset',
        help='Name of cached feature dataset')
    parser.add_argument(
        '--save_location', default='mlp.hook_post',
        help='Model component to probe')
    parser.add_argument(
        '--activation_aggregation', default='mean',
        help='Average activations across all tokens in a sequence')
    parser.add_argument(
        '--output_dir', default='cached_activations')
    parser.add_argument(
        '--batch_size', default=32, type=int,
    )

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HookedTransformer.from_pretrained(args.model, device='cpu')
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    tokenized_dataset = datasets.load_from_disk(
        os.path.join('data', args.feature_dataset))

    get_activations(args, model, tokenized_dataset, device)
