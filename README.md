# llm-context-neurons
A minimal replication of finding context neurons in Pythia models. See [Finding Neurons in a Haystack: Case Studies with Sparse Probing](https://arxiv.org/abs/2305.01610) for more details.

The results of this repo are available in the `context_neurons` directory.

## Instructions
This repo contains two datasets: `europarl_lang` and `pile_data_source` containing samples of parliment documents in many different European languages, and samples of the Pile dataset from different subdistributions.

After installing the required dependencies, save activations for any of the pythia models via 
```
python activations.py --model pythia-70m --feature_dataset pile_data_source
python activations.py --model pythia-70m --feature_dataset europarl_lang
```

Then, run the following to find context neurons:
```
python find_context_neurons.py --model pythia-70m --feature_dataset pile_data_source
python find_context_neurons.py --model pythia-70m --feature_dataset europarl_lang
```

This will dump a csv containing the top 10 most predictive neurons for each context feature in the dataset. Precomputed results are available in the `context_neurons` directory.


### Caveats
- Assumes you can store all activations in CPU RAM before caching
- Datasets are tokenized according to the pythia tokenizer so this won't work for other model families without modifying the datasets.


## Cite
If this was helpful, please cite our work:
```
@article{gurnee2023finding,
  title={Finding Neurons in a Haystack: Case Studies with Sparse Probing},
  author={Gurnee, Wes and Nanda, Neel and Pauly, Matthew and Harvey, Katherine and Troitskii, Dmitrii and Bertsimas, Dimitris},
  journal={arXiv preprint arXiv:2305.01610},
  year={2023}
}
```