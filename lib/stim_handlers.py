# lib/stim_handlers.py

"""
Modular stimulus handlers using improved base class.
Eliminates code duplication and provides consistent behavior.
Fixed for Python 3.9 compatibility - no lambda in callbacks.
"""

import random
import numpy as np
import logging
from lib.base_stim_handler import BaseStimHandler
from lib.constants import CommandStimParams, OddballStimParams, AudioParams

logger = logging.getLogger('eeg_stimulus.handlers')


class LanguageStimHandler(BaseStimHandler):
    """Handler for language stimuli."""
    
    def start(self, stim: dict):
        """Start a language stimulus.
        
        Args:
            stim: Stimulus dictionary
        """
        self.is_active = True
        n = stim.get('audio_index', 0)
        logger.debug(f"Starting language stimulus with audio index {n}")
        
        if 0 <= n < len(self.audio_stim.stims.lang_audio):
            audio_segment = self.audio_stim.stims.lang_audio[n]
            samples = self.reshape_audio_samples(audio_segment)
            
            # Log metadata
            self.log_event('language_stim_meta', {
                'audio_index': n,
                'sentence_ids': (self.audio_stim.stims.lang_stims_ids[n] 
                               if n < len(self.audio_stim.stims.lang_stims_ids) else None),
                'duration_sec': len(samples) / AudioParams.SAMPLE_RATE
            })

            # play_audio_safe includes a built-in watchdog
            self.play_audio_safe(
                samples=samples,
                sample_rate=AudioParams.SAMPLE_RATE,
                on_finish=self.safe_finish,
                log_label="language_audio"
            )
        else:
            logger.error(f"Invalid language audio index: {n}")
            self.audio_stim.finish_current_stim()
    
    def continue_stim(self):
        """Language stimuli don't have continuation logic."""
        pass


class CommandStimHandler(BaseStimHandler):
    """Handler for a single motor command keep+stop pair.

    Each instance of this handler runs exactly one keep phase followed by one
    stop phase.  The cycle loop has been moved to the stim_dictionary level —
    each pair is its own entry — so pause/resume restarts only the current
    pair, not the whole run.
    """

    def start(self, stim: dict):
        """Start one keep+stop pair."""
        self.is_active = True
        stim_type = stim.get('type', '')
        self.state = {
            'side': stim.get('side', 'right' if 'right' in stim_type else 'left'),
            'has_prompt': stim.get('has_prompt', False),
            'cycle_num': stim.get('cycle_num', 0),
            'total_cycles': stim.get('total_cycles', CommandStimParams.TOTAL_CYCLES),
        }
        logger.info(
            f"Starting {self.state['side']} command pair "
            f"{self.state['cycle_num'] + 1}/{self.state['total_cycles']} "
            f"(prompt={self.state['has_prompt']})"
        )
        if self.state['has_prompt']:
            self._play_prompt()
        else:
            self._play_keep_command()

    def _play_prompt(self):
        """Play the motor command prompt (first pair of a run only)."""
        if not self.is_active:
            return
        prompt_seg = self.audio_stim.stims.motor_prompt_audio
        samples = self.reshape_audio_samples(prompt_seg)
        self.play_audio_safe(
            samples=samples,
            sample_rate=AudioParams.SAMPLE_RATE,
            on_finish=self._after_prompt,
            log_label="motor_prompt",
        )

    def _after_prompt(self):
        self.safe_schedule(CommandStimParams.PROMPT_DELAY_MS, self._play_keep_command)

    def _play_keep_command(self):
        if not self.is_active:
            return
        logger.debug(
            f"Command pair {self.state['cycle_num'] + 1}/"
            f"{self.state['total_cycles']}: KEEP phase"
        )
        audio = (self.audio_stim.stims.right_keep_audio
                 if self.state['side'] == 'right'
                 else self.audio_stim.stims.left_keep_audio)
        samples = self.reshape_audio_samples(audio)
        self.play_audio_safe(
            samples=samples,
            sample_rate=AudioParams.SAMPLE_RATE,
            on_finish=self._after_keep_command,
            log_label=f"{self.state['side']}_keep",
        )

    def _after_keep_command(self):
        self.safe_schedule(CommandStimParams.KEEP_PAUSE_MS, self._play_stop_command)

    def _play_stop_command(self):
        if not self.is_active:
            return
        logger.debug(
            f"Command pair {self.state['cycle_num'] + 1}/"
            f"{self.state['total_cycles']}: STOP phase"
        )
        audio = (self.audio_stim.stims.right_stop_audio
                 if self.state['side'] == 'right'
                 else self.audio_stim.stims.left_stop_audio)
        samples = self.reshape_audio_samples(audio)
        self.play_audio_safe(
            samples=samples,
            sample_rate=AudioParams.SAMPLE_RATE,
            on_finish=self._after_stop_command,
            log_label=f"{self.state['side']}_stop",
        )

    def _after_stop_command(self):
        # Wait out the rest period, then the pair is complete.
        self.safe_schedule(CommandStimParams.STOP_PAUSE_MS, self.safe_finish)

    def continue_stim(self):
        """Not used — pairs are driven by the stim_dictionary."""
        pass


class OddballStimHandler(BaseStimHandler):
    """Handler for oddball stimuli using pre-generated buffer for precise timing.

    This handler generates all tones as a single continuous audio buffer,
    providing sample-accurate 1000ms onset-to-onset timing (±0.02ms at 44100Hz).
    """

    def start(self, stim: dict):
        """Start an oddball stimulus.

        Args:
            stim: Stimulus dictionary
        """
        self.is_active = True
        stim_type = stim.get('type', '')

        # Initialize state
        self.state = {
            'has_prompt': '+p' in stim_type,
        }

        logger.info(f"Starting oddball stimulus (prompt={self.state['has_prompt']})")

        if self.state['has_prompt']:
            self._play_prompt()
        else:
            self._start_sequence()

    def _play_prompt(self):
        """Play the oddball prompt."""
        if not self.is_active:
            return

        prompt_seg = self.audio_stim.stims.oddball_prompt_audio
        samples = self.reshape_audio_samples(prompt_seg)

        self.play_audio_safe(
            samples=samples,
            sample_rate=AudioParams.SAMPLE_RATE,
            on_finish=self._after_prompt,
            log_label="oddball_prompt"
        )

    def _after_prompt(self):
        """Called after prompt finishes."""
        self.safe_schedule(OddballStimParams.PROMPT_DELAY_MS, self._start_sequence)

    def _start_sequence(self):
        """Generate and play the complete oddball sequence as a single buffer."""
        if not self.is_active:
            return

        # Generate the complete sequence with all tones
        audio_samples, tone_events = self.audio_stim._generate_oddball_sequence(
            AudioParams.SAMPLE_RATE
        )

        # Store each tone's raw sample offset from the buffer start.
        # _save_oddball_results() converts to seconds and adds the buffer's
        # dac_onset_time to get each tone's precise DAC time.
        for event in tone_events:
            label = 'rare_tone' if event['type'] == 'rare' else 'standard_tone'
            self.audio_stim.current_stim_sentences.append({
                'event': label,
                'onset_sample': event['onset_sample'],
            })

        logger.info(f"Playing oddball sequence: {len(tone_events)} tones, "
                   f"duration={len(audio_samples)/AudioParams.SAMPLE_RATE:.2f}s")

        # log_label="oddball_sequence" attaches on_onset to capture the true DAC
        # time of the buffer start, used in _save_oddball_results() for per-tone times.
        self.play_audio_safe(
            samples=audio_samples,
            sample_rate=AudioParams.SAMPLE_RATE,
            on_finish=self._finish_stim,
            log_label="oddball_sequence"
        )

    def continue_stim(self):
        """Not used in pre-generated buffer approach."""
        pass

    def _finish_stim(self):
        """Finish the oddball stimulus."""
        if not self.is_active:
            return

        logger.debug(f"Oddball stimulus sequence completed "
                    f"({OddballStimParams.INITIAL_TONES + OddballStimParams.MAIN_TONES} tones)")
        self.audio_stim.finish_current_stim()


class VoiceStimHandler(BaseStimHandler):
    """Handler for familiar and unfamiliar voice stimuli."""

    def start(self, stim: dict):
        """Start a voice stimulus.

        Args:
            stim: Stimulus dictionary
        """
        self.is_active = True
        stim_type = stim.get('type', 'unfamiliar')
        is_unfamiliar = stim_type == 'unfamiliar'
        logger.info(f"Starting {'unfamiliar' if is_unfamiliar else 'familiar'} voice stimulus")

        # Get audio data based on stim type
        if is_unfamiliar:
            voice_index = stim.get('voice_index', 0)
            audio_data = self.audio_stim.stims.unfamiliar_voices_audio[voice_index]
        else:
            audio_data = self.audio_stim.stims.familiar_voice_audio

        if audio_data is None:
            logger.error(f"{stim_type} audio data is None, skipping stimulus")
            self.audio_stim.finish_current_stim()
            return

        # Ensure int16 format
        if audio_data.dtype != np.int16:
            logger.warning(f"{stim_type} audio has dtype {audio_data.dtype}, converting...")
            if np.issubdtype(audio_data.dtype, np.floating):
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)

        # Play audio
        self.play_audio_safe(
            samples=audio_data,
            sample_rate=AudioParams.SAMPLE_RATE,
            on_finish=self.safe_finish,
            log_label="voice_audio"
        )

        # play_audio_safe includes a built-in watchdog

    def continue_stim(self):
        """Voice stimuli don't have continuation logic."""
        pass