# lib/stim_handlers.py

"""
Modular stimulus handlers using improved base class.
Eliminates code duplication and provides consistent behavior.
Fixed for Python 3.9 compatibility - no lambda in callbacks.
"""

import random
import time
import numpy as np
import logging
from lib.base_stim_handler import BaseStimHandler
from lib.constants import CommandStimParams, OddballStimParams

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
                'duration_sec': len(samples) / audio_segment.frame_rate
            })
            
            # Play audio with safe finish
            self.play_audio_safe(
                samples=samples,
                sample_rate=audio_segment.frame_rate,
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
    """Handler for motor command stimuli."""
    
    def start(self, stim: dict):
        """Start a command stimulus.
        
        Args:
            stim: Stimulus dictionary
        """
        self.is_active = True
        stim_type = stim.get('type', '')
        
        # Initialize state
        self.state = {
            'side': 'right' if 'right' in stim_type else 'left',
            'has_prompt': '+p' in stim_type,
            'cycle': 0,
            'phase': 'keep'
        }
        
        logger.info(f"Starting {self.state['side']} command stimulus "
                   f"(prompt={self.state['has_prompt']})")
        
        self.log_event('command_stim_start', {
            'side': self.state['side'],
            'prompt': self.state['has_prompt']
        })
        
        if self.state['has_prompt']:
            self._play_prompt()
        else:
            self.continue_stim()
    
    def _play_prompt(self):
        """Play the motor command prompt."""
        if not self.is_active:
            return

        prompt_seg = self.audio_stim.stims.motor_prompt_audio
        samples = self.reshape_audio_samples(prompt_seg)

        self.play_audio_safe(
            samples=samples,
            sample_rate=prompt_seg.frame_rate,
            on_finish=self._after_prompt,
            log_label="motor_prompt"
        )
    
    def _after_prompt(self):
        """Called after prompt finishes."""
        self.safe_schedule(CommandStimParams.PROMPT_DELAY_MS, self.continue_stim)
    
    def continue_stim(self):
        """Continue the command stimulus cycle."""
        if not self.should_continue():
            return
        
        if self.state['cycle'] >= CommandStimParams.TOTAL_CYCLES:
            self._finish_stim()
            return
        
        phase = self.state['phase']
        
        if phase == 'keep':
            self._play_keep_command()
        elif phase == 'pause_after_keep':
            self.safe_schedule(CommandStimParams.KEEP_PAUSE_MS, self._start_stop_phase)
        elif phase == 'stop':
            self._play_stop_command()
        elif phase == 'pause_after_stop':
            self.safe_schedule(CommandStimParams.STOP_PAUSE_MS, self._next_cycle)
    
    def _play_keep_command(self):
        """Play the 'keep' command audio."""
        if not self.is_active:
            return
        
        logger.debug(f"Command stimulus cycle {self.state['cycle'] + 1}/"
                    f"{CommandStimParams.TOTAL_CYCLES}: KEEP phase")
        
        self.log_event('command_cycle', {
            'cycle': self.state['cycle'] + 1,
            'phase': 'keep',
            'side': self.state['side']
        })
        
        audio = (self.audio_stim.stims.right_keep_audio 
                if self.state['side'] == 'right'
                else self.audio_stim.stims.left_keep_audio)
        
        samples = np.array(audio.get_array_of_samples(), dtype=np.int16).reshape(-1, 1)
        
        # Use method reference instead of lambda
        self.play_audio_safe(
            samples=samples,
            sample_rate=audio.frame_rate,
            on_finish=self._after_keep_command,
            log_label=f"{self.state['side']}_keep"
        )
    
    def _after_keep_command(self):
        """Called after keep command finishes."""
        self._set_phase('pause_after_keep')
    
    def _play_stop_command(self):
        """Play the 'stop' command audio."""
        if not self.is_active:
            return
        
        logger.debug(f"Command stimulus cycle {self.state['cycle'] + 1}/"
                    f"{CommandStimParams.TOTAL_CYCLES}: STOP phase")
        
        self.log_event('command_cycle', {
            'cycle': self.state['cycle'] + 1,
            'phase': 'stop',
            'side': self.state['side']
        })
        
        audio = (self.audio_stim.stims.right_stop_audio 
                if self.state['side'] == 'right'
                else self.audio_stim.stims.left_stop_audio)
        
        samples = np.array(audio.get_array_of_samples(), dtype=np.int16).reshape(-1, 1)
        
        # Use method reference instead of lambda
        self.play_audio_safe(
            samples=samples,
            sample_rate=audio.frame_rate,
            on_finish=self._after_stop_command,
            log_label=f"{self.state['side']}_stop"
        )
    
    def _after_stop_command(self):
        """Called after stop command finishes."""
        self._set_phase('pause_after_stop')
    
    def _start_stop_phase(self):
        """Transition to stop phase."""
        self._set_phase('stop')
    
    def _set_phase(self, phase: str):
        """Set the current phase and continue.
        
        Args:
            phase: New phase name
        """
        if not self.is_active:
            return
        self.state['phase'] = phase
        self.continue_stim()
    
    def _next_cycle(self):
        """Move to the next cycle."""
        if not self.is_active:
            return
        self.state['cycle'] += 1
        self.state['phase'] = 'keep'
        self.continue_stim()
    
    def _finish_stim(self):
        """Finish the command stimulus."""
        if not self.is_active:
            return
        
        logger.debug(f"Command stimulus completed: {self.state['side']}, "
                    f"{CommandStimParams.TOTAL_CYCLES} cycles")
        
        self.log_event('command_stim_end', {
            'side': self.state['side'],
            'total_cycles': CommandStimParams.TOTAL_CYCLES
        })
        
        self.audio_stim.finish_current_stim()


class OddballStimHandler(BaseStimHandler):
    """Handler for oddball stimuli."""
    
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
            'tone_count': 0,
            'phase': 'initial_standard',
            'last_tone_time': None  # Track for double-beep detection
        }
        
        logger.info(f"Starting oddball stimulus (prompt={self.state['has_prompt']})")
        self.log_event('oddball_stim_start', {'prompt': self.state['has_prompt']})
        
        if self.state['has_prompt']:
            self._play_prompt()
        else:
            self.continue_stim()
    
    def _play_prompt(self):
        """Play the oddball prompt."""
        if not self.is_active:
            return

        prompt_seg = self.audio_stim.stims.oddball_prompt_audio
        samples = self.reshape_audio_samples(prompt_seg)

        self.play_audio_safe(
            samples=samples,
            sample_rate=prompt_seg.frame_rate,
            on_finish=self._after_prompt,
            log_label="oddball_prompt"
        )
    
    def _after_prompt(self):
        """Called after prompt finishes."""
        self.safe_schedule(OddballStimParams.PROMPT_DELAY_MS, self.continue_stim)
    
    def continue_stim(self):
        """Continue the oddball stimulus."""
        if not self.should_continue():
            return

        phase = self.state['phase']
        if phase == 'initial_standard':
            self._handle_initial_phase()
        elif phase == 'main_sequence':
            self._handle_main_phase()

    def _handle_initial_phase(self):
        """Handle the initial standard tones phase."""
        if not self.is_active:
            return

        if self.state['tone_count'] < OddballStimParams.INITIAL_TONES:
            self.state['tone_count'] += 1
            self._play_tone(OddballStimParams.STANDARD_FREQ, 'standard_tone')
        else:
            self.log_event('oddball_phase_change', {'from': 'initial_standard', 'to': 'main_sequence'})
            self.state['phase'] = 'main_sequence'
            self.state['tone_count'] = 0
            self.continue_stim()

    def _handle_main_phase(self):
        """Handle the main sequence phase with rare tones."""
        if not self.is_active:
            return

        if self.state['tone_count'] < OddballStimParams.MAIN_TONES:
            self.state['tone_count'] += 1
            is_rare = random.random() < OddballStimParams.RARE_PROBABILITY
            frequency = OddballStimParams.RARE_FREQ if is_rare else OddballStimParams.STANDARD_FREQ
            label = "rare_tone" if is_rare else "standard_tone"
            self._play_tone(frequency, label)
        else:
            self._finish_stim()

    def _play_tone(self, frequency: int, label: str):
        """Play a single tone."""
        if not self.is_active:
            return

        # Record tone onset time for accurate inter-tone timing
        self.state['last_tone_onset'] = time.time()

        tone_samples = self.audio_stim._generate_tone(frequency, OddballStimParams.TONE_DURATION_MS)
        self.play_audio_safe(
            samples=tone_samples,
            sample_rate=44100,
            on_finish=self._after_tone,
            log_label=label,
            # Offset onset time by leading padding so CSV records actual tone start
            onset_offset_ms=OddballStimParams.TONE_PADDING_MS
        )

    def _after_tone(self):
        """Called after tone finishes. Schedule next tone for 1000ms from onset."""
        # Calculate remaining time to achieve 1000ms onset-to-onset
        last_onset = self.state.get('last_tone_onset', time.time())
        elapsed_ms = (time.time() - last_onset) * 1000
        remaining_ms = max(0, 1000 - elapsed_ms)
        self.safe_schedule(int(remaining_ms), self.continue_stim)
    
    def _finish_stim(self):
        """Finish the oddball stimulus."""
        if not self.is_active:
            return
        
        logger.debug("Oddball stimulus sequence completed")
        self.log_event('oddball_stim_end', {
            'total_tones': OddballStimParams.INITIAL_TONES + OddballStimParams.MAIN_TONES
        })
        
        self.audio_stim.finish_current_stim()


class VoiceStimHandler(BaseStimHandler):
    """Handler for voice stimuli (control and loved one)."""

    def start(self, stim: dict):
        """Start a voice stimulus.

        Args:
            stim: Stimulus dictionary
        """
        self.is_active = True
        stim_type = stim.get('type', 'control')
        is_control = stim_type == 'control'
        logger.info(f"Starting {'control' if is_control else 'loved_one'} voice stimulus")

        self.log_event('voice_stim_start', {'type': stim_type})

        # Get audio data based on stim type
        audio_data = (self.audio_stim.stims.control_voice_audio
                     if is_control
                     else self.audio_stim.stims.loved_one_voice_audio)
        
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
            sample_rate=self.audio_stim.stims.sample_rate,
            on_finish=self.safe_finish,
            log_label=stim_type
        )
    
    def continue_stim(self):
        """Voice stimuli don't have continuation logic."""
        pass