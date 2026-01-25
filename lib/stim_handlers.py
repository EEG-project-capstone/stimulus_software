# lib/stim_handlers.py

"""
Modular stimulus handlers for different stimulus types.
Each handler encapsulates the logic for a specific stimulus type.
"""

import random
import numpy as np
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger('eeg_stimulus.stim_handlers')


class StimHandler(ABC):
    """Abstract base class for stimulus handlers."""
    
    def __init__(self, auditory_stimulator):
        self.audio_stim = auditory_stimulator
        self.state = {}
        self.is_active = False  # Track if handler is actively running
    
    @abstractmethod
    def start(self, stim: dict):
        """Start the stimulus."""
        pass
    
    @abstractmethod
    def continue_stim(self):
        """Continue the stimulus (for multi-phase stimuli)."""
        pass
    
    def reset(self):
        """Reset handler state."""
        self.state = {}
        self.is_active = False
    
    def stop(self):
        """Stop the handler immediately."""
        self.is_active = False
        self.state = {}


class LanguageStimHandler(StimHandler):
    """Handler for language stimuli."""
    
    def start(self, stim: dict):
        """Start a language stimulus."""
        self.is_active = True
        n = stim.get('audio_index', 0)
        logger.debug(f"Starting language stimulus with audio index {n}")
        
        if 0 <= n < len(self.audio_stim.stims.lang_audio):
            audio_segment = self.audio_stim.stims.lang_audio[n]
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
            
            if audio_segment.channels == 2:
                samples = samples.reshape(-1, 2)
            else:
                samples = samples.reshape(-1, 1)
            
            # Log metadata
            self.audio_stim._log_event('language_stim_meta', {
                'audio_index': n,
                'sentence_ids': self.audio_stim.stims.lang_stims_ids[n] if n < len(self.audio_stim.stims.lang_stims_ids) else None,
                'duration_sec': len(samples) / audio_segment.frame_rate
            })
            
            self.audio_stim.play_audio(
                samples=samples,
                sample_rate=audio_segment.frame_rate,
                callback=self._safe_finish,
                log_label="language_audio"
            )
        else:
            logger.error(f"Invalid language audio index: {n}")
            self.audio_stim.finish_current_stim()
    
    def _safe_finish(self):
        """Safely finish only if still active."""
        if self.is_active:
            self.audio_stim.finish_current_stim()
    
    def continue_stim(self):
        """Language stimuli don't have continuation logic."""
        pass


class CommandStimHandler(StimHandler):
    """Handler for motor command stimuli."""
    
    TOTAL_CYCLES = 8
    KEEP_PAUSE_MS = 10000
    STOP_PAUSE_MS = 10000
    PROMPT_DELAY_MS = 2000
    
    def start(self, stim: dict):
        """Start a command stimulus."""
        self.is_active = True
        stim_type = stim.get('type', '')
        self.state['side'] = 'right' if 'right' in stim_type else 'left'
        self.state['has_prompt'] = '+p' in stim_type
        self.state['cycle'] = 0
        self.state['phase'] = 'keep'
        
        logger.info(f"Starting {self.state['side']} command stimulus (prompt={self.state['has_prompt']})")
        self.audio_stim._log_event('command_stim_start', {
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
        samples = np.array(prompt_seg.get_array_of_samples(), dtype=np.int16)
        
        if prompt_seg.channels == 2:
            samples = samples.reshape(-1, 2)
        else:
            samples = samples.reshape(-1, 1)
        
        self.audio_stim.play_audio(
            samples=samples,
            sample_rate=prompt_seg.frame_rate,
            callback=lambda: self._safe_schedule(self.PROMPT_DELAY_MS, self.continue_stim),
            log_label="motor_prompt"
        )
    
    def continue_stim(self):
        """Continue the command stimulus cycle."""
        if not self.is_active or not self._should_continue():
            return
        
        if self.state['cycle'] >= self.TOTAL_CYCLES:
            self._finish_stim()
            return
        
        phase = self.state['phase']
        
        if phase == 'keep':
            self._play_keep_command()
        elif phase == 'pause_after_keep':
            self._safe_schedule(self.KEEP_PAUSE_MS, lambda: self._set_phase('stop'))
        elif phase == 'stop':
            self._play_stop_command()
        elif phase == 'pause_after_stop':
            self._safe_schedule(self.STOP_PAUSE_MS, self._next_cycle)
    
    def _should_continue(self) -> bool:
        """Check if stimulus should continue."""
        if not self.is_active:
            return False
        if self.audio_stim.gui_callback.playback_state != "playing" or self.audio_stim.is_paused:
            self._safe_schedule(100, self.continue_stim)
            return False
        return True
    
    def _play_keep_command(self):
        """Play the 'keep' command audio."""
        if not self.is_active:
            return
            
        logger.debug(f"Command stimulus cycle {self.state['cycle'] + 1}/{self.TOTAL_CYCLES}: KEEP phase")
        self.audio_stim._log_event('command_cycle', {
            'cycle': self.state['cycle'] + 1,
            'phase': 'keep',
            'side': self.state['side']
        })
        
        audio = (self.audio_stim.stims.right_keep_audio if self.state['side'] == 'right'
                else self.audio_stim.stims.left_keep_audio)
        
        samples = np.array(audio.get_array_of_samples(), dtype=np.int16).reshape(-1, 1)
        self.audio_stim.play_audio(
            samples=samples,
            sample_rate=audio.frame_rate,
            callback=lambda: self._set_phase('pause_after_keep'),
            log_label=f"{self.state['side']}_keep"
        )
    
    def _play_stop_command(self):
        """Play the 'stop' command audio."""
        if not self.is_active:
            return
            
        logger.debug(f"Command stimulus cycle {self.state['cycle'] + 1}/{self.TOTAL_CYCLES}: STOP phase")
        self.audio_stim._log_event('command_cycle', {
            'cycle': self.state['cycle'] + 1,
            'phase': 'stop',
            'side': self.state['side']
        })
        
        audio = (self.audio_stim.stims.right_stop_audio if self.state['side'] == 'right'
                else self.audio_stim.stims.left_stop_audio)
        
        samples = np.array(audio.get_array_of_samples(), dtype=np.int16).reshape(-1, 1)
        self.audio_stim.play_audio(
            samples=samples,
            sample_rate=audio.frame_rate,
            callback=lambda: self._set_phase('pause_after_stop'),
            log_label=f"{self.state['side']}_stop"
        )
    
    def _set_phase(self, phase: str):
        """Set the current phase and continue."""
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
        logger.debug(f"Command stimulus completed: {self.state['side']}, {self.TOTAL_CYCLES} cycles")
        self.audio_stim._log_event('command_stim_end', {
            'side': self.state['side'],
            'total_cycles': self.TOTAL_CYCLES
        })
        self.audio_stim.finish_current_stim()
    
    def _safe_schedule(self, delay_ms: int, callback):
        """Schedule a callback only if handler is still active."""
        if not self.is_active:
            logger.debug(f"Skipping schedule - handler inactive: {self.__class__.__name__}")
            return
        
        def wrapped_callback():
            if self.is_active:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in scheduled callback: {e}", exc_info=True)
            else:
                logger.debug(f"Skipped callback - handler became inactive: {self.__class__.__name__}")
        
        try:
            self.audio_stim._schedule(delay_ms, wrapped_callback)
        except Exception as e:
            logger.error(f"Failed to schedule callback: {e}", exc_info=True)


class OddballStimHandler(StimHandler):
    """Handler for oddball stimuli."""
    
    INITIAL_TONES = 5
    MAIN_TONES = 20
    TONE_DURATION_MS = 100
    INTER_TONE_INTERVAL_MS = 900
    STANDARD_FREQ = 1000
    RARE_FREQ = 2000
    RARE_PROBABILITY = 0.2
    PROMPT_DELAY_MS = 2000
    
    def start(self, stim: dict):
        """Start an oddball stimulus."""
        self.is_active = True
        stim_type = stim.get('type', '')
        self.state['has_prompt'] = '+p' in stim_type
        self.state['tone_count'] = 0
        self.state['phase'] = 'initial_standard'
        
        logger.info(f"Starting oddball stimulus (prompt={self.state['has_prompt']})")
        self.audio_stim._log_event('oddball_stim_start', {'prompt': self.state['has_prompt']})
        
        if self.state['has_prompt']:
            self._play_prompt()
        else:
            self.continue_stim()
    
    def _play_prompt(self):
        """Play the oddball prompt."""
        if not self.is_active:
            return
            
        prompt_seg = self.audio_stim.stims.oddball_prompt_audio
        samples = np.array(prompt_seg.get_array_of_samples(), dtype=np.int16)
        
        if prompt_seg.channels == 2:
            samples = samples.reshape(-1, 2)
        else:
            samples = samples.reshape(-1, 1)
        
        self.audio_stim.play_audio(
            samples=samples,
            sample_rate=prompt_seg.frame_rate,
            callback=lambda: self._safe_schedule(self.PROMPT_DELAY_MS, self.continue_stim),
            log_label="oddball_prompt"
        )
    
    def continue_stim(self):
        """Continue the oddball stimulus."""
        if not self.is_active or not self._should_continue():
            return
        
        phase = self.state['phase']
        
        if phase == 'initial_standard':
            self._handle_initial_phase()
        elif phase == 'main_sequence':
            self._handle_main_phase()
    
    def _should_continue(self) -> bool:
        """Check if stimulus should continue."""
        if not self.is_active:
            return False
        if self.audio_stim.gui_callback.playback_state != "playing" or self.audio_stim.is_paused:
            self._safe_schedule(100, self.continue_stim)
            return False
        return True
    
    def _handle_initial_phase(self):
        """Handle the initial standard tones phase."""
        if not self.is_active:
            return
            
        if self.state['tone_count'] < self.INITIAL_TONES:
            self.state['tone_count'] += 1
            logger.debug(f"Oddball initial tone {self.state['tone_count']}/{self.INITIAL_TONES}")
            
            self._play_tone(self.STANDARD_FREQ, 'standard_tone')
        else:
            logger.debug("Oddball switching to main sequence")
            self.audio_stim._log_event('oddball_phase_change', {
                'from': 'initial_standard',
                'to': 'main_sequence'
            })
            self.state['phase'] = 'main_sequence'
            self.state['tone_count'] = 0
            self.continue_stim()
    
    def _handle_main_phase(self):
        """Handle the main sequence phase with rare tones."""
        if not self.is_active:
            return
            
        if self.state['tone_count'] < self.MAIN_TONES:
            self.state['tone_count'] += 1
            
            is_rare = random.random() < self.RARE_PROBABILITY
            frequency = self.RARE_FREQ if is_rare else self.STANDARD_FREQ
            label = "rare_tone" if is_rare else "standard_tone"
            
            logger.debug(f"Oddball main tone {self.state['tone_count']}/{self.MAIN_TONES}: {label}")
            
            self._play_tone(frequency, label)
        else:
            self._finish_stim()
    
    def _play_tone(self, frequency: int, label: str):
        """Play a single tone."""
        if not self.is_active:
            return
            
        tone_samples = self.audio_stim._generate_tone(frequency, self.TONE_DURATION_MS)
        self.audio_stim.play_audio(
            samples=tone_samples,
            sample_rate=44100,
            callback=lambda: self._safe_schedule(self.INTER_TONE_INTERVAL_MS, self.continue_stim),
            log_label=label
        )
    
    def _finish_stim(self):
        """Finish the oddball stimulus."""
        if not self.is_active:
            return
        logger.debug("Oddball stimulus sequence completed")
        self.audio_stim._log_event('oddball_stim_end', {
            'total_tones': self.INITIAL_TONES + self.MAIN_TONES
        })
        self.audio_stim.finish_current_stim()
    
    def _safe_schedule(self, delay_ms: int, callback):
        """Schedule a callback only if handler is still active."""
        if not self.is_active:
            logger.debug(f"Skipping oddball schedule - handler inactive")
            return
        
        def wrapped_callback():
            if self.is_active:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in oddball scheduled callback: {e}", exc_info=True)
            else:
                logger.debug(f"Skipped oddball callback - handler became inactive")
        
        try:
            self.audio_stim._schedule(delay_ms, wrapped_callback)
        except Exception as e:
            logger.error(f"Failed to schedule oddball callback: {e}", exc_info=True)


class VoiceStimHandler(StimHandler):
    """Handler for voice stimuli (control and loved one)."""
    
    def start(self, stim: dict):
        """Start a voice stimulus."""
        self.is_active = True
        voice_type = stim.get('voice_type', 'control')
        logger.info(f"Starting {voice_type} voice stimulus")
        self.audio_stim._log_event('voice_stim_start', {'voice_type': voice_type})
        
        audio_data = (self.audio_stim.stims.control_voice_audio if voice_type == "control"
                     else self.audio_stim.stims.loved_one_voice_audio)
        
        if audio_data is None:
            logger.error(f"{voice_type} audio data is None, skipping stimulus")
            self.audio_stim.finish_current_stim()
            return
        
        # Ensure int16 format
        if audio_data.dtype != np.int16:
            logger.warning(f"{voice_type} audio has dtype {audio_data.dtype}, converting...")
            if np.issubdtype(audio_data.dtype, np.floating):
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)
        
        self.audio_stim._log_event('voice_stim_meta', {
            'voice_type': voice_type,
            'duration_sec': len(audio_data) / self.audio_stim.stims.sample_rate,
            'shape': audio_data.shape
        })
        
        self.audio_stim.play_audio(
            samples=audio_data,
            sample_rate=self.audio_stim.stims.sample_rate,
            callback=self._safe_finish,
            log_label=f"{voice_type}_voice"
        )
    
    def _safe_finish(self):
        """Safely finish only if still active."""
        if self.is_active:
            self.audio_stim.finish_current_stim()
    
    def continue_stim(self):
        """Voice stimuli don't have continuation logic."""
        pass