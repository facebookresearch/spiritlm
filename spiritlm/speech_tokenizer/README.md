# Speech Tokenization for Spirit-LM

This repo contains the speech encoder/decoder used for the Spirit-LM.

Here is an example of how to use spiritlm_tokenizer

```python
import IPython.display as ipd
from spiritlm.speech_tokenizer import spiritlm_base, spiritlm_expressive

tokenizer = spiritlm_base() # base version, only has hubert units
# tokenizer = spiritlm_expressive() # expressive version, with pitch & style units

# Input audio
audio = "examples/audio/7143-88743-0029.flac"
print('Original audio:')
ipd.display(ipd.Audio(audio))

## encode_units
print('\nEncode audio into units (not deduplicated) \n', '-'*20)
units = tokenizer.encode_units(audio)
print(units)
# > {'audio': '.../audio/7143-88743-0029.flac', 'hubert': '99 49 38 149 149 71...'}

## encode_string
print('\nEncode audio into string (deduplicated and sorted units) \n', '-'*20)
string_tokens = tokenizer.encode_string(audio)
print(string_tokens)
# > '[Hu99][Hu49][Hu38][Hu149][Hu71]...'

## decode from units
print('\nDecode back to audio from units (not deduplicated) \n', '-'*20)
resyn_wav = tokenizer.decode(units, speaker_id=2, dur_pred=False)
ipd.display(ipd.Audio(resyn_wav, rate=16000))

## decode from string
print('\nDecode back to audio from string (deduplicated and sorted units) \n', '-'*20)
resyn_dedup_wav = tokenizer.decode(string_tokens, speaker_id=2)
ipd.display(ipd.Audio(resyn_dedup_wav, rate=16000))
```