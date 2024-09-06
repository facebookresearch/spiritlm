# Model for Spirit-LM
This repo includes the Spirit-LM model wraper.

## Usage examples

### Model Loading
```python
from spiritlm.model.spiritlm_model import Spiritlm

# Spirit-LM-base 7B
spirit_lm = Spiritlm("spirit-lm-base-7b")

# Spirit-LM-expressive 7B
spirit_lm = Spiritlm("spirit-lm-expressive-7b")
```

### Generation examples
```python
from spiritlm.model.spiritlm_model import OutputModality, GenerationInput, ContentType
from transformers import GenerationConfig

# Generate only text
spirit_lm.generate(
    output_modality=OutputModality.TEXT,
    interleaved_inputs=[
        GenerationInput(
            content="The largest country in the world is",
            content_type=ContentType.TEXT,
        )
        ],
    generation_config=GenerationConfig(
        temperature=0.9,
        top_p=0.95,
        max_new_tokens=50,
        do_sample=True,
    ),
)

# Expected output format:
# [GenerationOuput(content='Russia, with an area of ...', content_type=<ContentType.TEXT: 'TEXT'>)]

# Generate only speech
spirit_lm.generate(
    output_modality=OutputModality.SPEECH,
    interleaved_inputs=[
        GenerationInput(
            content="examples/audio/7143-88743-0029.flac",
            content_type=ContentType.SPEECH,
        )
    ],
    generation_config=GenerationConfig(
        temperature=0.9,
        top_p=0.95,
        max_new_tokens=200,
        do_sample=True,
    ),
)

# Expected output format:
# [GenerationOuput(content=array([ 3.6673620e-05,  2.6468514e-04,  1.0735081e-03, ...,], dtype=float32), content_type=<ContentType.SPEECH: 'SPEECH'>)]


# Arbitrary generation
spirit_lm.generate(
    output_modality=OutputModality.ARBITRARY,
    interleaved_inputs=[
        GenerationInput(
            content="examples/audio/7143-88743-0029.flac",
            content_type=ContentType.SPEECH,
        )
    ],
    generation_config=GenerationConfig(
        temperature=0.9,
        top_p=0.95,
        max_new_tokens=200,
        do_sample=True,
    ),
)
# Expected output format is a list of GenerationOuput where content type could be `ContentType.TEXT' or `ContentType.SPEECH`:
# [GenerationOuput(content='xxx', content_type=<ContentType.TEXT: 'TEXT'>), GenerationOuput(content=array([ 0.00553902, -0.03210586, ... ], dtype=float32), content_type=<ContentType.SPEECH: 'SPEECH'>), GenerationOuput(content='yyy', content_type=<ContentType.TEXT: 'TEXT'>), GenerationOuput(content=array([0.04051103, 0.03596291, 0.03381396, ..., 0.05103811, 0.05429034, ..,,], dtype=float32), content_type=<ContentType.SPEECH: 'SPEECH'>)]
```
See more examples with other modalites in [examples/speech_generation/spirit_model.ipynb](../../examples/speech_generation/spirit_model.ipynb).