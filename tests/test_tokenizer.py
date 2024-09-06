import torchaudio
from spiritlm.speech_tokenizer import spiritlm_expressive, spiritlm_base
import pytest


@pytest.fixture
def spiritlm_expressive_tokenizer():
    return spiritlm_expressive()


@pytest.fixture
def spiritlm_base_tokenizer():
    return spiritlm_base()


def test_expressive_tokenizer_encode_units(spiritlm_expressive_tokenizer):
    audio = "examples/audio/7143-88743-0029.flac"
    units = spiritlm_expressive_tokenizer.encode_units(audio)
    expected = {
        "hubert": "99 49 38 149 149 71 423 427 492 288 315 153 153 389 497 412 247 354 7 96 452 452 176 266 266 77 248 336 336 211 166 65 94 224 224 148 492 191 440 440 41 41 457 79 382 451 332 216 114 340 478 74 79 370 272 370 370 53 477 65 171 60 258 111 111 111 111 338 338 23 23 338 23 338 338 338 7 338 338 149 406 7 361 361 361 99 99 99 99 99 99 99 209 209 209 209 209 479 50 50 7 149 149 35 35 130 130 169 169 72 434 119 272 4 249 245 245 433 159 294 139 359 343 269 302 226 370 216 459 424 424 226 382 7 58 138 428 397 350 350 306 306 306 84 11 171 171 60 314 227 227 355 9 58 138 226 370 272 382 334 330 176 176 307 145 248 493 64 44 388 7 111 111 111 111 23 23 481 149 149 80 70 431 457 79 79 249 249 245 245 245 433 433 316 316 180 458 458 458 86 86 225 103 60 96 119 119 129 356 218 4 259 259 392 490 75 488 166 65 171 60 7 54 54 85 85 361 361",
        "pitch": "39 57 48 48 56 44 42 59 51 44 43 0 54 13 39 25 58 26 44 26 13 20 46 26 26 40 26 56 51 14 46 41 41 40 40 39 39 39 57 57 59 59 59 59 59 20 20 20 20 35 35 55 13 3 52 9 9 20 57 56 56 56 56 32 39 57 46 27 20 44 32 59 59 42 48 48 56 44 25 56 26 56 46 56 40 39 39 59 59 13 15 47 13 27 55 54 36 53 3 26 43 49 3 39 41 14 56 46 14 46 40 40 39 40",
        "style": "71 71 71 71 71 71 71 71 71 83",
    }
    for token_key in ["hubert", "pitch", "style"]:
        assert expected[token_key] == units[token_key], f"{token_key} expected {expected[token_key]}, got {units[token_key]}"


def test_expressive_tokenizer_encode_units_with_tensor_input(spiritlm_expressive_tokenizer):
    wav = torchaudio.load("examples/audio/7143-88743-0029.flac")[0].squeeze(0)
    units = spiritlm_expressive_tokenizer.encode_units(wav)
    expected = {
        "hubert": "99 49 38 149 149 71 423 427 492 288 315 153 153 389 497 412 247 354 7 96 452 452 176 266 266 77 248 336 336 211 166 65 94 224 224 148 492 191 440 440 41 41 457 79 382 451 332 216 114 340 478 74 79 370 272 370 370 53 477 65 171 60 258 111 111 111 111 338 338 23 23 338 23 338 338 338 7 338 338 149 406 7 361 361 361 99 99 99 99 99 99 99 209 209 209 209 209 479 50 50 7 149 149 35 35 130 130 169 169 72 434 119 272 4 249 245 245 433 159 294 139 359 343 269 302 226 370 216 459 424 424 226 382 7 58 138 428 397 350 350 306 306 306 84 11 171 171 60 314 227 227 355 9 58 138 226 370 272 382 334 330 176 176 307 145 248 493 64 44 388 7 111 111 111 111 23 23 481 149 149 80 70 431 457 79 79 249 249 245 245 245 433 433 316 316 180 458 458 458 86 86 225 103 60 96 119 119 129 356 218 4 259 259 392 490 75 488 166 65 171 60 7 54 54 85 85 361 361",
        "pitch": "39 57 48 48 56 44 42 59 51 44 43 0 54 13 39 25 58 26 44 26 13 20 46 26 26 40 26 56 51 14 46 41 41 40 40 39 39 39 57 57 59 59 59 59 59 20 20 20 20 35 35 55 13 3 52 9 9 20 57 56 56 56 56 32 39 57 46 27 20 44 32 59 59 42 48 48 56 44 25 56 26 56 46 56 40 39 39 59 59 13 15 47 13 27 55 54 36 53 3 26 43 49 3 39 41 14 56 46 14 46 40 40 39 40",
        "style": "71 71 71 71 71 71 71 71 71 83",
    }
    for token_key in ["hubert", "pitch", "style"]:
        assert expected[token_key] == units[token_key], f"{token_key} expected {expected[token_key]}, got {units[token_key]}"


def test_base_tokenizer_encode_units(spiritlm_base_tokenizer):
    audio = "examples/audio/7143-88743-0029.flac"
    units = spiritlm_base_tokenizer.encode_units(audio)
    expected_hubert = "99 49 38 149 149 71 423 427 492 288 315 153 153 389 497 412 247 354 7 96 452 452 176 266 266 77 248 336 336 211 166 65 94 224 224 148 492 191 440 440 41 41 457 79 382 451 332 216 114 340 478 74 79 370 272 370 370 53 477 65 171 60 258 111 111 111 111 338 338 23 23 338 23 338 338 338 7 338 338 149 406 7 361 361 361 99 99 99 99 99 99 99 209 209 209 209 209 479 50 50 7 149 149 35 35 130 130 169 169 72 434 119 272 4 249 245 245 433 159 294 139 359 343 269 302 226 370 216 459 424 424 226 382 7 58 138 428 397 350 350 306 306 306 84 11 171 171 60 314 227 227 355 9 58 138 226 370 272 382 334 330 176 176 307 145 248 493 64 44 388 7 111 111 111 111 23 23 481 149 149 80 70 431 457 79 79 249 249 245 245 245 433 433 316 316 180 458 458 458 86 86 225 103 60 96 119 119 129 356 218 4 259 259 392 490 75 488 166 65 171 60 7 54 54 85 85 361 361"
    assert expected_hubert == units["hubert"]


def test_expressive_tokenizer_encode_string(spiritlm_expressive_tokenizer):
    audio = "examples/audio/7143-88743-0029.flac"
    encoded_string = spiritlm_expressive_tokenizer.encode_string(audio)
    expected = "[St71][Pi39][Hu99][Hu49][Pi57][Hu38][Hu149][Pi48][Hu71][Hu423][Hu427][Pi56][Hu492][Hu288][Pi44][Hu315][Hu153][Pi42][Hu389][Pi59][Hu497][Hu412][Pi51][Hu247][Hu354][Pi44][Hu7][Hu96][Pi43][Hu452][Pi0][Hu176][Hu266][Pi54][St71][Hu77][Pi13][Hu248][Hu336][Pi39][Hu211][Pi25][Hu166][Hu65][Pi58][Hu94][Hu224][Pi26][Hu148][Pi44][Hu492][Hu191][Pi26][Hu440][Pi13][Hu41][Pi20][Hu457][Hu79][Pi46][Hu382][Hu451][Pi26][Hu332][Hu216][Hu114][Hu340][St71][Pi40][Hu478][Hu74][Pi26][Hu79][Hu370][Pi56][Hu272][Hu370][Pi51][Hu53][Pi14][Hu477][Hu65][Pi46][Hu171][Hu60][Pi41][Hu258][Hu111][Pi40][Hu338][Hu23][Pi39][Hu338][Hu23][Hu338][St71][Pi57][Hu7][Hu338][Hu149][Pi59][Hu406][Hu7][Hu361][Hu99][Pi20][Hu209][Hu479][Pi35][Hu50][St71][Hu7][Hu149][Pi55][Hu35][Pi13][Hu130][Pi3][Hu169][Pi52][Hu72][Pi9][Hu434][Hu119][Hu272][Hu4][Pi20][Hu249][Hu245][Pi57][Hu433][Pi56][Hu159][Hu294][Hu139][Hu359][Hu343][Hu269][Hu302][St71][Hu226][Pi32][Hu370][Hu216][Pi39][Hu459][Hu424][Pi57][Hu226][Pi46][Hu382][Hu7][Pi27][Hu58][Hu138][Pi20][Hu428][Hu397][Pi44][Hu350][Pi32][Hu306][Pi59][Hu84][Hu11][Hu171][Pi42][Hu60][Pi48][Hu314][Hu227][St71][Hu355][Pi56][Hu9][Hu58][Pi44][Hu138][Hu226][Pi25][Hu370][Hu272][Pi56][Hu382][Hu334][Pi26][Hu330][Hu176][Pi56][Hu307][Pi46][Hu145][Hu248][Pi56][Hu493][Hu64][Pi40][Hu44][Hu388][Pi39][Hu7][Hu111][Pi59][St71][Hu23][Hu481][Pi13][Hu149][Pi15][Hu80][Hu70][Pi47][Hu431][Hu457][Pi13][Hu79][Pi27][Hu249][Pi55][Hu245][Pi54][Hu433][Pi36][Hu316][Pi53][Hu180][Pi3][Hu458][Pi26][Hu86][St71][Pi43][Hu225][Pi49][Hu103][Hu60][Pi3][Hu96][Hu119][Pi39][Hu129][Pi41][Hu356][Hu218][Pi14][Hu4][Hu259][Pi56][Hu392][Pi46][Hu490][Hu75][Pi14][Hu488][Hu166][Pi46][Hu65][Hu171][Pi40][Hu60][Hu7][Hu54][Pi39][Hu85][St83][Pi40][Hu361]"
    assert encoded_string == expected


def test_base_tokenizer_encode_string(spiritlm_base_tokenizer):
    audio = "examples/audio/7143-88743-0029.flac"
    encoded_string = spiritlm_base_tokenizer.encode_string(audio)
    expected = "[Hu99][Hu49][Hu38][Hu149][Hu71][Hu423][Hu427][Hu492][Hu288][Hu315][Hu153][Hu389][Hu497][Hu412][Hu247][Hu354][Hu7][Hu96][Hu452][Hu176][Hu266][Hu77][Hu248][Hu336][Hu211][Hu166][Hu65][Hu94][Hu224][Hu148][Hu492][Hu191][Hu440][Hu41][Hu457][Hu79][Hu382][Hu451][Hu332][Hu216][Hu114][Hu340][Hu478][Hu74][Hu79][Hu370][Hu272][Hu370][Hu53][Hu477][Hu65][Hu171][Hu60][Hu258][Hu111][Hu338][Hu23][Hu338][Hu23][Hu338][Hu7][Hu338][Hu149][Hu406][Hu7][Hu361][Hu99][Hu209][Hu479][Hu50][Hu7][Hu149][Hu35][Hu130][Hu169][Hu72][Hu434][Hu119][Hu272][Hu4][Hu249][Hu245][Hu433][Hu159][Hu294][Hu139][Hu359][Hu343][Hu269][Hu302][Hu226][Hu370][Hu216][Hu459][Hu424][Hu226][Hu382][Hu7][Hu58][Hu138][Hu428][Hu397][Hu350][Hu306][Hu84][Hu11][Hu171][Hu60][Hu314][Hu227][Hu355][Hu9][Hu58][Hu138][Hu226][Hu370][Hu272][Hu382][Hu334][Hu330][Hu176][Hu307][Hu145][Hu248][Hu493][Hu64][Hu44][Hu388][Hu7][Hu111][Hu23][Hu481][Hu149][Hu80][Hu70][Hu431][Hu457][Hu79][Hu249][Hu245][Hu433][Hu316][Hu180][Hu458][Hu86][Hu225][Hu103][Hu60][Hu96][Hu119][Hu129][Hu356][Hu218][Hu4][Hu259][Hu392][Hu490][Hu75][Hu488][Hu166][Hu65][Hu171][Hu60][Hu7][Hu54][Hu85][Hu361]"
    assert encoded_string == expected
