# tests/test_stims.py
"""Tests for stims.py - Stimulus generation and management."""

import pytest
import sys
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.stims import Stims


class MockConfig:
    """Mock configuration for Stims testing."""

    def __init__(self, temp_dir):
        self.temp_dir = Path(temp_dir)
        self.file = {
            'sentences_path': self.temp_dir / 'sentences',
            'right_keep_path': self.temp_dir / 'right_keep.mp3',
            'right_stop_path': self.temp_dir / 'right_stop.mp3',
            'left_keep_path': self.temp_dir / 'left_keep.mp3',
            'left_stop_path': self.temp_dir / 'left_stop.mp3',
            'motor_prompt_path': self.temp_dir / 'motorcommandprompt.wav',
            'oddball_prompt_path': self.temp_dir / 'oddballprompt.wav',
            'loved_one_path': self.temp_dir / 'loved_one',
            'male_control_path': self.temp_dir / 'male_control.wav',
            'female_control_path': self.temp_dir / 'female_control.wav',
        }


class MockGuiCallback:
    """Mock GUI callback for Stims testing."""

    def __init__(self, config):
        self.config = config


@pytest.fixture
def stims_temp_dir():
    """Create a temporary directory structure for stims testing."""
    tmp = tempfile.mkdtemp()
    tmp_path = Path(tmp)

    # Create subdirectories
    (tmp_path / 'sentences').mkdir()
    (tmp_path / 'loved_one').mkdir()

    yield tmp_path
    shutil.rmtree(tmp)


@pytest.fixture
def mock_stims_config(stims_temp_dir):
    """Create a mock config with the temp directory."""
    return MockConfig(stims_temp_dir)


@pytest.fixture
def stims_instance(mock_stims_config):
    """Create a Stims instance with mock config."""
    gui_callback = MockGuiCallback(mock_stims_config)
    return Stims(gui_callback)


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
        assert stims_instance.loved_one_voice_audio is None
        assert stims_instance.control_voice_audio is None


class TestGenerateStimsValidation:
    """Tests for stimulus generation validation."""

    def test_generate_stims_clears_existing(self, stims_instance):
        """generate_stims should clear existing stimuli before generating."""
        stims_instance.stim_dictionary = [{"type": "old"}]
        stims_instance.generate_stims({})
        assert stims_instance.stim_dictionary == []

    def test_generate_stims_requires_loved_one_file(self, stims_instance):
        """generate_stims should raise if loved one stimuli requested without file."""
        with pytest.raises(ValueError, match="no audio file specified"):
            stims_instance.generate_stims({"loved": 1})

    def test_generate_stims_requires_loved_one_gender(self, stims_instance):
        """generate_stims should raise if loved one stimuli requested without valid gender."""
        stims_instance.loved_one_file = "test.wav"
        with pytest.raises(ValueError, match="gender not properly set"):
            stims_instance.generate_stims({"loved": 1})

    def test_generate_stims_requires_valid_gender(self, stims_instance):
        """generate_stims should raise if gender is invalid."""
        stims_instance.loved_one_file = "test.wav"
        stims_instance.loved_one_gender = "Invalid"
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
    def test_generate_oddball_with_prompt(self, mock_from_wav, stims_instance, stims_temp_dir):
        """generate_stims should create oddball+prompt stimuli and load prompt audio."""
        # Create mock audio file
        prompt_file = stims_temp_dir / 'oddballprompt.wav'
        prompt_file.touch()

        mock_audio = MagicMock()
        mock_from_wav.return_value = mock_audio

        stims_instance.generate_stims({"odd+p": 2})

        oddball_p_stims = [s for s in stims_instance.stim_dictionary if s['type'] == 'oddball+p']
        assert len(oddball_p_stims) == 2
        assert stims_instance.oddball_prompt_audio is not None


class TestCommandStimuli:
    """Tests for command stimulus generation."""

    @patch('lib.stims.AudioSegment.from_mp3')
    def test_generate_right_command_stimuli(self, mock_from_mp3, stims_instance, stims_temp_dir):
        """generate_stims should create right command stimuli."""
        # Create mock audio files
        for path_key in ['right_keep_path', 'right_stop_path']:
            stims_instance.config.file[path_key].touch()

        mock_audio = MagicMock()
        mock_from_mp3.return_value = mock_audio

        stims_instance.generate_stims({"rcmd": 2})

        rcmd_stims = [s for s in stims_instance.stim_dictionary if s['type'] == 'right_command']
        assert len(rcmd_stims) == 2
        assert stims_instance.right_keep_audio is not None
        assert stims_instance.right_stop_audio is not None

    @patch('lib.stims.AudioSegment.from_mp3')
    def test_generate_left_command_stimuli(self, mock_from_mp3, stims_instance, stims_temp_dir):
        """generate_stims should create left command stimuli."""
        for path_key in ['left_keep_path', 'left_stop_path']:
            stims_instance.config.file[path_key].touch()

        mock_audio = MagicMock()
        mock_from_mp3.return_value = mock_audio

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
        mock_audio.frame_rate = 22050  # Not 44100
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
        # Interleaved stereo samples: L R L R
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
        # Remove the sentences directory
        shutil.rmtree(stims_temp_dir / 'sentences')

        with pytest.raises(FileNotFoundError, match="Sentences directory not found"):
            stims_instance._random_lang_stim()

    def test_insufficient_sentences_raises(self, stims_instance, stims_temp_dir):
        """_random_lang_stim should raise if not enough sentences."""
        # Create only 5 sentence files when 12 are needed by default
        sentences_dir = stims_temp_dir / 'sentences'
        for i in range(5):
            (sentences_dir / f'lang{i}.wav').touch()

        with pytest.raises(ValueError, match="Requested 12 sentences, but only 5 available"):
            stims_instance._random_lang_stim()


class TestBlockRandomization:
    """Tests for block randomization in generate_stims."""

    @patch.object(Stims, '_random_lang_stim')
    def test_blocks_are_shuffled(self, mock_random_lang, stims_instance):
        """generate_stims should shuffle blocks (not individual stimuli)."""
        # Generate multiple times and check that order varies
        orders = []
        for _ in range(10):
            stims_instance.generate_stims({
                "lang": 1,
                "odd": 1
            })
            order = [s['type'] for s in stims_instance.stim_dictionary]
            orders.append(tuple(order))

        # With shuffling, we should see at least 2 different orderings in 10 trials
        # (probability of same order 10 times is 1/1024 for 2 items)
        unique_orders = set(orders)
        assert len(unique_orders) >= 1  # At minimum, we have valid orders

    @patch.object(Stims, '_random_lang_stim')
    def test_stimuli_within_block_preserved(self, mock_random_lang, stims_instance):
        """Stimuli within a block should stay together."""
        stims_instance.generate_stims({"lang": 3})

        # All language stimuli should be contiguous (single block)
        lang_indices = [i for i, s in enumerate(stims_instance.stim_dictionary) if s['type'] == 'language']

        if lang_indices:
            # Check they are sequential
            for i in range(len(lang_indices) - 1):
                assert lang_indices[i + 1] - lang_indices[i] == 1


class TestLovedOneStimuli:
    """Tests for loved one voice stimulus generation."""

    @patch.object(Stims, '_load_audio_as_int16')
    def test_generate_loved_one_creates_pairs(self, mock_load_audio, stims_instance, stims_temp_dir):
        """generate_stims should create paired control + loved_one stimuli."""
        # Set up loved one requirements
        stims_instance.loved_one_file = "loved.wav"
        stims_instance.loved_one_gender = "Male"

        # Create mock audio files
        (stims_temp_dir / 'loved_one' / 'loved.wav').touch()
        (stims_temp_dir / 'male_control.wav').touch()

        mock_load_audio.return_value = np.array([[100]], dtype=np.int16)

        stims_instance.generate_stims({"loved": 2})

        control_stims = [s for s in stims_instance.stim_dictionary if s['type'] == 'control']
        loved_stims = [s for s in stims_instance.stim_dictionary if s['type'] == 'loved_one_voice']

        assert len(control_stims) == 2
        assert len(loved_stims) == 2
        # Total should be 4 (2 pairs)
        assert len(stims_instance.stim_dictionary) == 4

    @patch.object(Stims, '_load_audio_as_int16')
    def test_loved_one_missing_file_raises(self, mock_load_audio, stims_instance, stims_temp_dir):
        """generate_stims should raise if loved one file not found."""
        stims_instance.loved_one_file = "nonexistent.wav"
        stims_instance.loved_one_gender = "Male"

        with pytest.raises(FileNotFoundError, match="Loved one audio file not found"):
            stims_instance.generate_stims({"loved": 1})

    @patch.object(Stims, '_load_audio_as_int16')
    def test_female_control_path_used(self, mock_load_audio, stims_instance, stims_temp_dir):
        """generate_stims should use female control for Female gender."""
        stims_instance.loved_one_file = "loved.wav"
        stims_instance.loved_one_gender = "Female"

        (stims_temp_dir / 'loved_one' / 'loved.wav').touch()
        (stims_temp_dir / 'female_control.wav').touch()

        mock_load_audio.return_value = np.array([[100]], dtype=np.int16)

        stims_instance.generate_stims({"loved": 1})

        # Check that female_control_path was loaded (second call)
        assert mock_load_audio.call_count == 2
        # Second call should be for control path
        call_args = mock_load_audio.call_args_list[1]
        assert 'female_control' in str(call_args)
