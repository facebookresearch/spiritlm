from unittest.mock import Mock, patch
import pytest

from spiritlm.model.spiritlm_model import Spiritlm
from spiritlm.model.utils import (
    does_end_with_speech_token,
    does_start_with_speech_token,
    find_prompt_last_speech_start_position,
)


@pytest.mark.parametrize(
    "content,expected",
    [
        (
            "abc[Hubert][St1][Pi234][Hu123][Hu45][Text]hello world[",
            [("abc", "t"), ("[St1][Pi234][Hu123][Hu45]", "s"), ("hello world[", "t")],
        ),
        (
            "[St1][Pi234][Hu123][Hu45]",
            [("[St1][Pi234][Hu123][Hu45]", "s")],
        ),
        (
            "abc",
            [("abc", "t")],
        ),
        (
            "abc[]",
            [("abc[]", "t")],
        ),
        (
            "[St1][Pi234][Hu123][Hu45][Text][abc",
            [("[St1][Pi234][Hu123][Hu45]", "s"), ("[abc", "t")],
        ),
        (
            "abc[Text]def",
            [("abcdef", "t")],
        ),
    ],
)
def test_parse_speech_and_text(content, expected):
    with patch(
        "spiritlm.model.spiritlm_model.Spiritlm.__init__", Mock(return_value=None)
    ):
        mock_spiritlm_model = Spiritlm("spirit-lm-base-7b")
        mock_spiritlm_model.speech_prompt_prefix = "[Hubert]"
        assert mock_spiritlm_model._parse_speech_and_text(content) == expected


@pytest.mark.parametrize(
    "content,expected",
    [
        (
            "[Hu338][Text] and they went out together[HubertPitchStyle][Hu431][Pi0][Hu457][Hu79][Pi11][Hu258][Hu85][Hu28][Hu50][Text] and mrs johnson  shoes except in mourning[HubertPitchStyle][Pi59][Hu32][Pi20][Hu453][Pi35][Pi26][Hu166]",
            [
                ("[Hu338]", "s"),
                (" and they went out together", "t"),
                ("[Hu431][Pi0][Hu457][Hu79][Pi11][Hu258][Hu85][Hu28][Hu50]", "s"),
                (" and mrs johnson  shoes except in mourning", "t"),
                ("[Pi59][Hu32][Pi20][Hu453][Pi35][Pi26][Hu166]", "s"),
            ],
        )
    ],
)
def test_parse_speech_and_text_with_expressive_tokens(content, expected):
    with patch(
        "spiritlm.model.spiritlm_model.Spiritlm.__init__", Mock(return_value=None)
    ):
        mock_spiritlm_model = Spiritlm("spirit-lm-base-7b")
        mock_spiritlm_model.speech_prompt_prefix = "[HubertPitchStyle]"
        assert mock_spiritlm_model._parse_speech_and_text(content) == expected


@pytest.mark.parametrize(
    "encoded_string,expected",
    [
        (
            "]]",
            False,
        ),
        (
            "[]",
            False,
        ),
        (
            "[Hu100]",
            True,
        ),
        ("abc[]", False),
        (
            "[St1][Pi234][Hu123][Hu45][Text][abc]",
            False,
        ),
        (
            "abc[Text]def",
            False,
        ),
        (
            "[Pi9]",
            True,
        ),
        (
            "[St0]",
            True,
        ),
    ],
)
def test_does_prompt_end_by_speech(encoded_string, expected):
    assert does_end_with_speech_token(encoded_string) == expected


@pytest.mark.parametrize(
    "encoded_string,expected",
    [
        (
            "abc[Hu123][Hu456][Pi23][St2]",
            3,
        ),
        (
            "[Hu123]abc[Hu123][Hu456][Pi23][St2]",
            10,
        ),
        (
            "[Hu123][Hu456][Pi23][St2]",
            0,
        ),
        (
            "abc",
            None,
        ),
        (
            "[HubertPitchStyle][St71][Pi39][Hu99][Hu49][Pi57][Hu38][Hu149][Pi48][Hu71][Hu423][Hu427][Pi56][Hu492][Hu288][Pi44][Hu315][Hu153][Pi42][Hu389][Pi59][Hu497][Hu412][Pi51][Hu247][Hu354][Pi44][Hu7][Hu96][Pi43][Hu452][Pi0][Hu176][Hu266][Pi54][St71][Hu77][Pi13][Hu248][Hu336][Pi39][Hu211][Pi25][Hu166][Hu65][Pi58][Hu94][Hu224][Pi26][Hu148][Pi44][Hu492][Hu191][Pi26][Hu440][Pi13][Hu41][Pi20][Hu457][Hu79][Pi46][Hu382][Hu451][Pi26][Hu332][Hu216][Hu114][Hu340][St71][Pi40][Hu478][Hu74][Pi26][Hu79][Hu370][Pi56][Hu272][Hu370][Pi51][Hu53][Pi14][Hu477][Hu65][Pi46][Hu171][Hu60][Pi41][Hu258][Hu111][Pi40][Hu338][Hu23][Pi39][Hu338][Hu23][Hu338][St71][Pi57][Hu7][Hu338][Hu149][Pi59][Hu406][Hu7][Hu361][Hu99][Pi20][Hu209][Hu479][Pi35][Hu50][St71][Hu7][Hu149][Pi55][Hu35][Pi13][Hu130][Pi3][Hu169][Pi52][Hu72][Pi9][Hu434][Hu119][Hu272][Hu4][Pi20][Hu249][Hu245][Pi57][Hu433][Pi56][Hu159][Hu294][Hu139][Hu359][Hu343][Hu269][Hu302][St71][Hu226][Pi32][Hu370][Hu216][Pi39][Hu459][Hu424][Pi57][Hu226][Pi46][Hu382][Hu7][Pi27][Hu58][Hu138][Pi20][Hu428][Hu397][Pi44][Hu350][Pi32][Hu306][Pi59][Hu84][Hu11][Hu171][Pi42][Hu60][Pi48][Hu314][Hu227][St71][Hu355][Pi56][Hu9][Hu58][Pi44][Hu138][Hu226][Pi25][Hu370][Hu272][Pi56][Hu382][Hu334][Pi26][Hu330][Hu176][Pi56][Hu307][Pi46][Hu145][Hu248][Pi56][Hu493][Hu64][Pi40][Hu44][Hu388][Pi39][Hu7][Hu111][Pi59][St71][Hu23][Hu481][Pi13][Hu149][Pi15][Hu80][Hu70][Pi47][Hu431][Hu457][Pi13][Hu79][Pi27][Hu249][Pi55][Hu245][Pi54][Hu433][Pi36][Hu316][Pi53][Hu180][Pi3][Hu458][Pi26][Hu86][St71][Pi43][Hu225][Pi49][Hu103][Hu60][Pi3][Hu96][Hu119][Pi39][Hu129][Pi41][Hu356][Hu218][Pi14][Hu4][Hu259][Pi56][Hu392][Pi46][Hu490][Hu75][Pi14][Hu488][Hu166][Pi46][Hu65][Hu171][Pi40][Hu60][Hu7][Hu54][Pi39][Hu85][St83][Pi40][Hu361]",
            18,
        ),
    ],
)
def test_find_prompt_last_speech_start_position(encoded_string, expected):
    assert find_prompt_last_speech_start_position(encoded_string) == expected


@pytest.mark.parametrize(
    "encoded_string,expected",
    [
        (
            "[[",
            False,
        ),
        (
            "[]",
            False,
        ),
        (
            "[Hu100]",
            True,
        ),
        ("abc[]", False),
        (
            "[St1][Pi234][Hu123][Hu45][Text][abc]",
            True,
        ),
        (
            "abc[Text]def",
            False,
        ),
        (
            "[Pi9]",
            True,
        ),
        (
            "[St0]",
            True,
        ),
    ],
)
def test_does_start_with_speech_token(encoded_string, expected):
    assert does_start_with_speech_token(encoded_string) == expected
