# SpiritLM Checkpoints

## Locations (TODO: remove it once open sourced)
### H2
`/private/home/ntuanh/Projects/parrot/packages/spiritlm/checkpoints`
### AWS
`/data/home/bokai/codes/spiritlm/checkpoints`
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
    │   ├── pytorch_model-00001-of-00002.bin
    │   ├── pytorch_model-00002-of-00002.bin
    │   ├── pytorch_model.bin.index.json
    │   ├── special_tokens_map.json
    │   ├── tokenizer.json
    │   ├── tokenizer.model
    │   └── tokenizer_config.json
    └── spirit-lm-expressive-7b
        ├── config.json
        ├── generation_config.json
        ├── pytorch_model-00001-of-00002.bin
        ├── pytorch_model-00002-of-00002.bin
        ├── pytorch_model.bin.index.json
        ├── special_tokens_map.json
        ├── tokenizer.json
        ├── tokenizer.model
        └── tokenizer_config.json
```
