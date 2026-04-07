import threading


class Config:
    """Thread-safe shared configuration for processing and UI."""

    def __init__(self):
        self._lock = threading.RLock()
        self._active = {
            'volume': 0.55,
            'audio_blocksize': 512,
            'bypass_all': False,
            'noise_reduction': True,
            'noise_reduction_strength': 1.1,
            'nr_band_low_hz': 120.0,
            'nr_band_high_hz': 6000.0,
            'nr_cepstral_smoothing': True,
            'nr_attack_release_split': True,
            'compression_strength': 1.0,
            'eq_enabled': True,
            'eq_bass_db': -3.0,
            'eq_presence_db': 2.5,
            'eq_treble_db': -0.5,
            'stt_enabled': True,
            'stt_input_gain': 1.1,
            'stt_sensitivity': 'normal',
            'stt_model_root': 'models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20',
        }

    def get(self, key):
        with self._lock:
            return self._active.get(key)

    def set(self, key, value):
        """Update config immediately for UI visibility and runtime use."""
        with self._lock:
            if key == 'volume':
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = self._active.get('volume', 0.7)
                value = max(0.0, min(6.0, value))
            if key == 'audio_blocksize':
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    value = self._active.get('audio_blocksize', 128)
                value = max(32, min(2048, value))
                # Keep to sensible powers-of-two-ish values for audio callbacks.
                value = int(round(value / 32.0) * 32)
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
            if key == 'stt_input_gain':
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = self._active.get('stt_input_gain', 1.0)
                value = max(0.5, min(8.0, value))
            if key in ('eq_bass_db', 'eq_presence_db', 'eq_treble_db'):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = self._active.get(key, 0.0)
                value = max(-12.0, min(12.0, value))
            if key == 'stt_model_root':
                if value is None:
                    value = 'models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20'
                value = str(value).strip()
                if not value or value.lower() == 'none':
                    value = 'models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20'
            if key in ('noise_reduction', 'nr_cepstral_smoothing', 'nr_attack_release_split', 'stt_enabled', 'eq_enabled', 'bypass_all'):
                value = bool(value)

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

    def apply(self):
        """Compatibility hook for existing processing loop (no-op with immediate updates)."""
        return False

    def get_all(self):
        with self._lock:
            return self._active.copy()
