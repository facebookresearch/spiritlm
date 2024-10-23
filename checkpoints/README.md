# Spirit LM Checkpoints

## Download Checkpoints
To access and download Spirit LM Checkpoints, please request the model artifacts in this link:

[https://ai.meta.com/resources/models-and-libraries/spirit-lm-downloads/](https://ai.meta.com/resources/models-and-libraries/spirit-lm-downloads/)

Upon approval, you will then receive an email with download links to each model artifact.

Please note that Spirit LM is made available under the **FAIR Noncommercial Research License**
found in the [LICENSE](../LICENSE) file in the root directory of this source tree and Acceptable Use Policy.

## Structure
The checkpoints directory should look like this:
```
checkpoints/
├── README.md
├── speech_tokenizer
│   ├── hifigan_spiritlm_base
│   │   ├── config.json
│   │   ├── generator.pt
│   │   ├── speakers.txt
│   │   └── styles.txt
│   ├── hifigan_spiritlm_expressive_w2v2
│   │   ├── config.json
│   │   ├── generator.pt
│   │   └── speakers.txt
│   ├── hubert_25hz
│   │   ├── L11_quantizer_500.pt
│   │   └── mhubert_base_25hz.pt
│   ├── style_encoder_w2v2
│   │   ├── config.json
│   │   └── pytorch_model.bin
│   └── vqvae_f0_quantizer
│       ├── config.yaml
│       └── model.pt
└── spiritlm_model
    ├── spirit-lm-base-7b
    │   ├── config.json
    │   ├── generation_config.json
    │   ├── pytorch_model.bin
    │   ├── special_tokens_map.json
    │   ├── tokenizer_config.json
    │   └── tokenizer.model
    └── spirit-lm-expressive-7b
        ├── config.json
        ├── generation_config.json
        ├── pytorch_model.bin
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        └── tokenizer.model
```
You can export `SPIRITLM_CHECKPOINTS_DIR` to point to a differnt directory where you downloaded checkpoints.