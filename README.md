# Meta Spirit LM: Interleaved Spoken and Written Language Model

This repository contains the model weights, inference code and evaluation scripts for the Spirit LM [paper](https://arxiv.org/pdf/2402.05755.pdf). You can find more generation samples on our [demo page](https://speechbot.github.io/spiritlm/).

## Spirit LM Model Overview
<img src="assets/spiritlm_overview.png">

## Installation Setup
### Conda
```
conda env create -f env.yml
pip install -e '.[eval]'

```
### Pip
```
pip install -e '.[eval]'
```

### Dev
(Optionally, use only if you want to run the tests.)
```
pip install -e '.[dev]'
```

## Checkpoints Setup
See [checkpoints/README.md](checkpoints/README.md)

## Quick Start
### Speech Tokenization
See [spiritlm/speech_tokenizer/README.md](spiritlm/speech_tokenizer/README.md)
### Spirit LM Generation
See [spiritlm/model/README.md](spiritlm/model/README.md)
### Speech-Text Sentiment Preservation benchmark (STSP)
See [spiritlm/eval/README.md](spiritlm/eval/README.md)

## Model Card
More details of the model can be found in [MODEL_CARD.md](MODEL_CARD.md).

## License
The present code is provided under the **FAIR Noncommercial Research License** found in [LICENSE](LICENSE).

## Citation
```
@misc{nguyen2024spiritlminterleavedspokenwritten,
      title={SpiRit-LM: Interleaved Spoken and Written Language Model},
      author={Tu Anh Nguyen and Benjamin Muller and Bokai Yu and Marta R. Costa-jussa and Maha Elbayad and Sravya Popuri and Paul-Ambroise Duquenne and Robin Algayres and Ruslan Mavlyutov and Itai Gat and Gabriel Synnaeve and Juan Pino and Benoit Sagot and Emmanuel Dupoux},
      year={2024},
      eprint={2402.05755},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.05755},
}
```

