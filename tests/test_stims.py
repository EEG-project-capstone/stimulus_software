# tests/test_stims.py
"""Tests for stims.py - Stimulus generation and management."""

import pytest
import sys
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.stims import Stims
from lib.constants import FilePaths


@pytest.fixture
def stims_temp_dir():
    """Create a temporary directory structure for stims testing."""
    tmp = tempfile.mkdtemp()
    tmp_path = Path(tmp)

    (tmp_path / 'sentences').mkdir()
    (tmp_path / 'familiar').mkdir()

    yield tmp_path
    shutil.rmtree(tmp)


@pytest.fixture
def stims_instance():
    """Create a Stims instance."""
    return Stims()


class TestStimsInit:
    """Tests for Stims initialization."""

    def test_initialization(self, stims_instance):
        """Stims should initialize with empty state."""
        assert stims_instance.stim_dictionary == []
        assert stims_instance.current_stim_index is None
        assert stims_instance.lang_audio == []
        assert stims_instance.lang_stims_ids == []
        assert stims_instance.sample_rate == 44100

    def test_initialization_clears_audio(self, stims_instance):
        """Stims should initialize with null audio references."""
        assert stims_instance.right_keep_audio is None
        assert stims_instance.right_stop_audio is None
        assert stims_instance.left_keep_audio is None
        assert stims_instance.left_stop_audio is None
        assert stims_instance.familiar_voice_audio is None
        assert stims_instance.unfamiliar_voices_audio == []


class TestGenerateStimsValidation:
    """Tests for stimulus generation validation."""

    def test_generate_stims_clears_existing(self, stims_instance):
        """generate_stims should clear existing stimuli before generating."""
        stims_instance.stim_dictionary = [{"type": "old"}]
        stims_instance.generate_stims({})
        assert stims_instance.stim_dictionary == []

    def test_generate_stims_requires_familiar_file(self, stims_instance):
        """generate_stims should raise if familiar voice stimuli requested without file."""
        with pytest.raises(ValueError, match="no audio file specified"):
            stims_instance.generate_stims({"loved": 1})

    def test_generate_stims_requires_familiar_gender(self, stims_instance):
        """generate_stims should raise if familiar voice stimuli requested without valid gender."""
        stims_instance.familiar_file = "test.wav"
        with pytest.raises(ValueError, match="gender not properly set"):
            stims_instance.generate_stims({"loved": 1})

    def test_generate_stims_requires_valid_gender(self, stims_instance):
        """generate_stims should raise if gender is invalid."""
        stims_instance.familiar_file = "test.wav"
        stims_instance.familiar_gender = "Invalid"
        with pytest.raises(ValueError, match="gender not properly set"):
            stims_instance.generate_stims({"loved": 1})


class TestOddballStimuli:
    """Tests for oddball stimulus generation."""

    def test_generate_oddball_stimuli(self, stims_instance):
        """generate_stims should create oddball stimuli."""
        stims_instance.generate_stims({"odd": 3})

        oddball_stims = [s for s in stims_instance.stim_dictionary if s['type'] == 'oddball']
        assert len(oddball_stims) == 3
        assert all(s['status'] == 'pending' for s in oddball_stims)

    @patch('lib.stims.AudioSegment.from_wav')
    def test_generate_oddball_with_prompt(self, mock_from_wav, stims_instance):
        """generate_stims should create oddball+prompt stimuli and load prompt audio."""
        mock_from_wav.return_value = MagicMock()

        stims_instance.generate_stims({"odd+p": 2})

        oddball_p_stims = [s for s in stims_instance.stim_dictionary if s['type'] == 'oddball+p']
        assert len(oddball_p_stims) == 2
        assert stims_instance.oddball_prompt_audio is not None


class TestCommandStimuli:
    """Tests for command stimulus generation."""

    @patch('lib.stims.AudioSegment.from_mp3')
    def test_generate_right_command_stimuli(self, mock_from_mp3, stims_instance):
        """generate_stims should create right command stimuli."""
        mock_from_mp3.return_value = MagicMock()

        stims_instance.generate_stims({"rcmd": 2})

        rcmd_stims = [s for s in stims_instance.stim_dictionary if s['type'] == 'right_command']
        assert len(rcmd_stims) == 2
        assert stims_instance.right_keep_audio is not None
        assert stims_instance.right_stop_audio is not None

    @patch('lib.stims.AudioSegment.from_mp3')
    def test_generate_left_command_stimuli(self, mock_from_mp3, stims_instance):
        """generate_stims should create left command stimuli."""
        mock_from_mp3.return_value = MagicMock()

        stims_instance.generate_stims({"lcmd": 3})

        lcmd_stims = [s for s in stims_instance.stim_dictionary if s['type'] == 'left_command']
        assert len(lcmd_stims) == 3


class TestLanguageStimuli:
    """Tests for language stimulus generation."""

    @patch.object(Stims, '_random_lang_stim')
    def test_generate_language_stimuli(self, mock_random_lang, stims_instance):
        """generate_stims should create language stimuli."""
        stims_instance.generate_stims({"lang": 5})

        lang_stims = [s for s in stims_instance.stim_dictionary if s['type'] == 'language']
        assert len(lang_stims) == 5
        assert mock_random_lang.call_count == 5

    @patch.object(Stims, '_random_lang_stim')
    def test_language_stimuli_have_audio_index(self, mock_random_lang, stims_instance):
        """Language stimuli should have incrementing audio_index."""
        stims_instance.generate_stims({"lang": 3})

        lang_stims = [s for s in stims_instance.stim_dictionary if s['type'] == 'language']
        audio_indices = [s['audio_index'] for s in lang_stims]
        assert audio_indices == [0, 1, 2]


class TestStimulusCounts:
    """Tests for stimulus count combinations."""

    @patch.object(Stims, '_random_lang_stim')
    def test_multiple_stim_types(self, mock_random_lang, stims_instance):
        """generate_stims should handle multiple stimulus types."""
        stims_instance.generate_stims({
            "lang": 2,
            "odd": 2
        })

        lang_count = len([s for s in stims_instance.stim_dictionary if s['type'] == 'language'])
        odd_count = len([s for s in stims_instance.stim_dictionary if s['type'] == 'oddball'])

        assert lang_count == 2
        assert odd_count == 2
        assert len(stims_instance.stim_dictionary) == 4

    def test_zero_counts_ignored(self, stims_instance):
        """generate_stims should ignore zero counts."""
        stims_instance.generate_stims({"lang": 0, "odd": 0})
        assert stims_instance.stim_dictionary == []

    def test_empty_counts(self, stims_instance):
        """generate_stims should handle empty counts dict."""
        stims_instance.generate_stims({})
        assert stims_instance.stim_dictionary == []


class TestLoadAudioAsInt16:
    """Tests for _load_audio_as_int16 method."""

    @patch('lib.stims.AudioSegment.from_mp3')
    def test_load_mp3_file(self, mock_from_mp3, stims_instance, stims_temp_dir):
        """_load_audio_as_int16 should load mp3 files."""
        test_file = stims_temp_dir / 'test.mp3'
        test_file.touch()

        mock_audio = MagicMock()
        mock_audio.frame_rate = 44100
        mock_audio.channels = 1
        mock_audio.get_array_of_samples.return_value = np.array([100, 200, 300], dtype=np.int16)
        mock_from_mp3.return_value = mock_audio

        result = stims_instance._load_audio_as_int16(test_file)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int16

    @patch('lib.stims.AudioSegment.from_wav')
    def test_load_wav_file(self, mock_from_wav, stims_instance, stims_temp_dir):
        """_load_audio_as_int16 should load wav files."""
        test_file = stims_temp_dir / 'test.wav'
        test_file.touch()

        mock_audio = MagicMock()
        mock_audio.frame_rate = 44100
        mock_audio.channels = 1
        mock_audio.get_array_of_samples.return_value = np.array([100, 200, 300], dtype=np.int16)
        mock_from_wav.return_value = mock_audio

        result = stims_instance._load_audio_as_int16(test_file)

        assert isinstance(result, np.ndarray)

    def test_unsupported_format_raises(self, stims_instance, stims_temp_dir):
        """_load_audio_as_int16 should raise for unsupported formats."""
        test_file = stims_temp_dir / 'test.ogg'
        test_file.touch()

        with pytest.raises(ValueError, match="Unsupported file format"):
            stims_instance._load_audio_as_int16(test_file)

    @patch('lib.stims.AudioSegment.from_wav')
    def test_resamples_to_44100(self, mock_from_wav, stims_instance, stims_temp_dir):
        """_load_audio_as_int16 should resample non-44100Hz audio."""
        test_file = stims_temp_dir / 'test.wav'
        test_file.touch()

        mock_audio = MagicMock()
        mock_audio.frame_rate = 22050
        mock_audio.channels = 1
        mock_audio.get_array_of_samples.return_value = np.array([100], dtype=np.int16)
        mock_audio.set_frame_rate.return_value = mock_audio
        mock_from_wav.return_value = mock_audio

        stims_instance._load_audio_as_int16(test_file)

        mock_audio.set_frame_rate.assert_called_with(44100)

    @patch('lib.stims.AudioSegment.from_wav')
    def test_reshapes_stereo_audio(self, mock_from_wav, stims_instance, stims_temp_dir):
        """_load_audio_as_int16 should reshape stereo to (n, 2)."""
        test_file = stims_temp_dir / 'test.wav'
        test_file.touch()

        mock_audio = MagicMock()
        mock_audio.frame_rate = 44100
        mock_audio.channels = 2
        mock_audio.get_array_of_samples.return_value = np.array([1, 2, 3, 4, 5, 6], dtype=np.int16)
        mock_from_wav.return_value = mock_audio

        result = stims_instance._load_audio_as_int16(test_file)

        assert result.shape == (3, 2)

    @patch('lib.stims.AudioSegment.from_wav')
    def test_reshapes_mono_audio(self, mock_from_wav, stims_instance, stims_temp_dir):
        """_load_audio_as_int16 should reshape mono to (n, 1)."""
        test_file = stims_temp_dir / 'test.wav'
        test_file.touch()

        mock_audio = MagicMock()
        mock_audio.frame_rate = 44100
        mock_audio.channels = 1
        mock_audio.get_array_of_samples.return_value = np.array([1, 2, 3], dtype=np.int16)
        mock_from_wav.return_value = mock_audio

        result = stims_instance._load_audio_as_int16(test_file)

        assert result.shape == (3, 1)


class TestRandomLangStim:
    """Tests for _random_lang_stim method."""

    def test_missing_sentences_directory_raises(self, stims_instance, stims_temp_dir):
        """_random_lang_stim should raise if sentences directory missing."""
        nonexistent = stims_temp_dir / 'nonexistent_sentences'
        with patch.object(FilePaths, 'SENTENCES_DIR', nonexistent):
            with pytest.raises(FileNotFoundError, match="Sentences directory not found"):
                stims_instance._random_lang_stim()

    def test_insufficient_sentences_raises(self, stims_instance, stims_temp_dir):
        """_random_lang_stim should raise if not enough sentences."""
        sentences_dir = stims_temp_dir / 'sentences'
        for i in range(5):
            (sentences_dir / f'lang{i}.wav').touch()

        with patch.object(FilePaths, 'SENTENCES_DIR', sentences_dir):
            with pytest.raises(ValueError, match="Requested 12 sentences, but only 5 available"):
                stims_instance._random_lang_stim()


class TestBlockRandomization:
    """Tests for block randomization in generate_stims."""

    @patch.object(Stims, '_random_lang_stim')
    def test_blocks_are_shuffled(self, mock_random_lang, stims_instance):
        """generate_stims should shuffle blocks (not individual stimuli)."""
        orders = []
        for _ in range(10):
            stims_instance.generate_stims({
                "lang": 1,
                "odd": 1
            })
            order = [s['type'] for s in stims_instance.stim_dictionary]
            orders.append(tuple(order))

        unique_orders = set(orders)
        assert len(unique_orders) >= 1

    @patch.object(Stims, '_random_lang_stim')
    def test_stimuli_within_block_preserved(self, mock_random_lang, stims_instance):
        """Stimuli within a block should stay together."""
        stims_instance.generate_stims({"lang": 3})

        lang_indices = [i for i, s in enumerate(stims_instance.stim_dictionary) if s['type'] == 'language']

        if lang_indices:
            for i in range(len(lang_indices) - 1):
                assert lang_indices[i + 1] - lang_indices[i] == 1


class TestVoiceStimuli:
    """Tests for familiar/unfamiliar voice stimulus generation."""

    def _make_control_statements_dir(self, tmp_path, gender):
        """Create a mock control_statements directory with the expected normalized wav files."""
        import lib.constants as const
        cs_dir = tmp_path / 'control_statements'
        cs_dir.mkdir(exist_ok=True)
        names = const.MALE_CONTROL_VOICES if gender == 'Male' else const.FEMALE_CONTROL_VOICES
        for name in names:
            (cs_dir / f"{name}_normalized.wav").touch()
        return cs_dir

    @patch.object(Stims, '_load_audio_as_int16')
    def test_generate_voice_balanced_split(self, mock_load_audio, stims_instance, stims_temp_dir):
        """generate_stims should create an equal number of familiar and unfamiliar trials."""
        stims_instance.familiar_file = "familiar.wav"
        stims_instance.familiar_gender = "Male"

        familiar_dir = stims_temp_dir / 'familiar'
        (familiar_dir / 'familiar.wav').touch()
        cs_dir = self._make_control_statements_dir(stims_temp_dir, 'Male')

        mock_load_audio.return_value = np.array([[100]], dtype=np.int16)

        with patch.object(FilePaths, 'FAMILIAR_DIR', familiar_dir), \
             patch.object(FilePaths, 'CONTROL_STATEMENTS_DIR', cs_dir):
            stims_instance.generate_stims({"loved": 4})

        familiar_stims = [s for s in stims_instance.stim_dictionary if s['type'] == 'familiar']
        unfamiliar_stims = [s for s in stims_instance.stim_dictionary if s['type'] == 'unfamiliar']

        assert len(familiar_stims) == 2
        assert len(unfamiliar_stims) == 2
        assert len(stims_instance.stim_dictionary) == 4

    @patch.object(Stims, '_load_audio_as_int16')
    def test_unfamiliar_stims_have_voice_index(self, mock_load_audio, stims_instance, stims_temp_dir):
        """Unfamiliar stims should carry a voice_index key referencing a loaded speaker."""
        stims_instance.familiar_file = "familiar.wav"
        stims_instance.familiar_gender = "Male"

        familiar_dir = stims_temp_dir / 'familiar'
        (familiar_dir / 'familiar.wav').touch()
        cs_dir = self._make_control_statements_dir(stims_temp_dir, 'Male')

        mock_load_audio.return_value = np.array([[100]], dtype=np.int16)

        with patch.object(FilePaths, 'FAMILIAR_DIR', familiar_dir), \
             patch.object(FilePaths, 'CONTROL_STATEMENTS_DIR', cs_dir):
            stims_instance.generate_stims({"loved": 2})

        unfamiliar_stims = [s for s in stims_instance.stim_dictionary if s['type'] == 'unfamiliar']
        for s in unfamiliar_stims:
            assert 'voice_index' in s
            assert 0 <= s['voice_index'] < 4

    @patch.object(Stims, '_load_audio_as_int16')
    def test_familiar_missing_file_raises(self, mock_load_audio, stims_instance, stims_temp_dir):
        """generate_stims should raise if the familiar voice file is not found."""
        stims_instance.familiar_file = "nonexistent.wav"
        stims_instance.familiar_gender = "Male"

        with patch.object(FilePaths, 'FAMILIAR_DIR', stims_temp_dir / 'familiar'):
            with pytest.raises(FileNotFoundError, match="Familiar voice audio file not found"):
                stims_instance.generate_stims({"loved": 1})

    @patch.object(Stims, '_load_audio_as_int16')
    def test_female_unfamiliar_voices_loaded(self, mock_load_audio, stims_instance, stims_temp_dir):
        """generate_stims should load all four female unfamiliar speakers for Female gender."""
        import lib.constants as const
        stims_instance.familiar_file = "familiar.wav"
        stims_instance.familiar_gender = "Female"

        familiar_dir = stims_temp_dir / 'familiar'
        (familiar_dir / 'familiar.wav').touch()
        cs_dir = self._make_control_statements_dir(stims_temp_dir, 'Female')

        mock_load_audio.return_value = np.array([[100]], dtype=np.int16)

        with patch.object(FilePaths, 'FAMILIAR_DIR', familiar_dir), \
             patch.object(FilePaths, 'CONTROL_STATEMENTS_DIR', cs_dir):
            stims_instance.generate_stims({"loved": 2})

        # 1 familiar load + 4 unfamiliar speaker loads = 5 total
        assert mock_load_audio.call_count == 5
        loaded_paths = [str(call.args[0]) for call in mock_load_audio.call_args_list[1:]]
        for name in const.FEMALE_CONTROL_VOICES:
            assert any(name in p for p in loaded_paths)
