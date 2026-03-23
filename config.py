import threading


class Config:
    """Thread-safe shared configuration for processing and UI."""

    def __init__(self):
        self._lock = threading.RLock()
        self._active = {
            'volume': 0.7,
            'noise_reduction': True,
            'noise_reduction_strength': 1.4,
            'nr_band_low_hz': 120.0,
            'nr_band_high_hz': 6000.0,
            'nr_cepstral_smoothing': True,
            'nr_attack_release_split': True,
            'nr_wind_mode': False,
            'nr_wind_sensitivity': 0.6,
            'compression_strength': 1.0,
            'aec_enabled': True,
            'aec_strength': 1.2,
            'aec_delay_blocks': 4,
            'near_end_suppression_mode': False,
            'near_end_suppression_strength': 0.6,
            'feedback_cancellation': True,
            'stt_enabled': True,
            'stt_input_gain': 1.0,
        }
        self._pending = self._active.copy()
        self._needs_swap = False

    def get(self, key):
        with self._lock:
            return self._active.get(key)

    def set(self, key, value):
        """Update config immediately for UI visibility and runtime use."""
        with self._lock:
            # Backward compatibility for older UI/client key.
            if key == 'feedback_cancellation':
                key = 'aec_enabled'

            if key == 'volume':
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = self._active.get('volume', 0.7)
                value = max(0.0, min(3.0, value))
            if key == 'noise_reduction_strength':
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = self._active.get('noise_reduction_strength', 1.4)
                value = max(0.3, min(3.0, value))
            if key == 'nr_band_low_hz':
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = self._active.get('nr_band_low_hz', 120.0)
                value = max(0.0, min(7900.0, value))
            if key == 'nr_band_high_hz':
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = self._active.get('nr_band_high_hz', 6000.0)
                value = max(100.0, min(8000.0, value))
            if key == 'nr_wind_sensitivity':
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = self._active.get('nr_wind_sensitivity', 0.6)
                value = max(0.0, min(1.0, value))
            if key == 'aec_strength':
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = self._active.get('aec_strength', 1.2)
                value = max(0.2, min(3.0, value))
            if key == 'aec_delay_blocks':
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    value = self._active.get('aec_delay_blocks', 4)
                value = max(1, min(16, value))
            if key == 'near_end_suppression_strength':
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = self._active.get('near_end_suppression_strength', 0.6)
                value = max(0.0, min(1.0, value))
            if key == 'stt_input_gain':
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = self._active.get('stt_input_gain', 1.0)
                value = max(0.5, min(8.0, value))
            if key in ('noise_reduction', 'nr_cepstral_smoothing', 'nr_attack_release_split', 'nr_wind_mode', 'aec_enabled', 'near_end_suppression_mode', 'stt_enabled'):
                value = bool(value)

            # Keep legacy key mirrored for existing UI/client code.
            if key == 'aec_enabled':
                self._pending['feedback_cancellation'] = bool(value)
                self._active['feedback_cancellation'] = bool(value)

            self._pending[key] = value
            self._active[key] = value

            # Keep band limits ordered.
            low = float(self._active.get('nr_band_low_hz', 120.0))
            high = float(self._active.get('nr_band_high_hz', 6000.0))
            if low > high - 100.0:
                if key == 'nr_band_low_hz':
                    high = min(8000.0, low + 100.0)
                else:
                    low = max(0.0, high - 100.0)
                self._active['nr_band_low_hz'] = low
                self._active['nr_band_high_hz'] = high
                self._pending['nr_band_low_hz'] = low
                self._pending['nr_band_high_hz'] = high

            self._needs_swap = False

    def apply(self):
        """Compatibility hook for existing processing loop."""
        with self._lock:
            if self._needs_swap:
                self._active = self._pending.copy()
                self._needs_swap = False
                return True
        return False

    def get_all(self):
        with self._lock:
            return self._active.copy()
