# main.py
import threading
import time
import numpy as np
import queue
from collections import deque

from config import Config
from audio_io import AudioIO
from processing import NoiseReducer, Compressor, FeedbackCanceller
from stt import StreamingSTT
from server import start_server

# ----------------------------------------------------------------------
# Global shared queues for UI updates
meter_queue = queue.Queue(maxsize=5)          # (left_rms, right_rms)
transcription_queue = queue.Queue()
quality_queue = queue.Queue(maxsize=5)

# ----------------------------------------------------------------------
def audio_processing_thread(config_obj, audio_io):
    """
    Runs the real‑time audio processing pipeline.
    """
    # Initialize per-channel algorithm instances so channel state is not mixed.
    noise_reducer_l = NoiseReducer()
    noise_reducer_r = NoiseReducer()
    compressor_l = Compressor()
    compressor_r = Compressor()
    feedback_canceller_l = FeedbackCanceller(filter_length=1024, mu=0.025)
    feedback_canceller_r = FeedbackCanceller(filter_length=1024, mu=0.025)
    spk_hist_l = deque(maxlen=24)
    spk_hist_r = deque(maxlen=24)

    def pick_best_reference(mic_block, history, fallback_len, delay_blocks=4):
        if not history:
            return np.zeros(fallback_len, dtype=np.float32)

        best = history[-1]
        best_score = -1.0
        mic = mic_block.astype(np.float32, copy=False)
        mic_norm = float(np.linalg.norm(mic)) + 1e-8

        delay_blocks = int(np.clip(delay_blocks, 1, max(1, len(history))))
        base_idx = len(history) - delay_blocks
        start_idx = max(0, base_idx - 2)
        end_idx = min(len(history) - 1, base_idx + 2)
        candidates = [history[i] for i in range(start_idx, end_idx + 1)]

        mic_rms = float(np.sqrt(np.mean(mic * mic)) + 1e-8)
        for ref in candidates:
            if ref.shape[0] != fallback_len:
                continue
            ref_rms = float(np.sqrt(np.mean(ref * ref)) + 1e-8)
            if ref_rms < 1e-5:
                continue
            ref_norm = float(np.linalg.norm(ref)) + 1e-8
            score = abs(float(np.dot(mic, ref))) / (mic_norm * ref_norm)
            # Prefer references with realistic energy match to avoid wrong-lag locks.
            energy_penalty = min(ref_rms / mic_rms, mic_rms / ref_rms)
            score *= energy_penalty
            if score > best_score:
                best_score = score
                best = ref
        return best

    # Optional: load audiogram (example)
    # frequencies = [250, 500, 1000, 2000, 4000, 8000]
    # thresholds = [45, 50, 55, 60, 65, 70]
    # compressor.set_audiogram(frequencies, thresholds)

    # STT instance (managed dynamically from config).
    stt = None
    last_stt_init_attempt = 0.0

    def ensure_stt_state():
        nonlocal stt, last_stt_init_attempt
        enabled = bool(config_obj.get('stt_enabled'))

        if enabled and stt is None:
            now = time.monotonic()
            # Retry model init periodically in case files are added at runtime.
            if now - last_stt_init_attempt < 5.0:
                return
            last_stt_init_attempt = now
            try:
                stt = StreamingSTT(model_root="models", sample_rate=16000)
                stt.start()
                transcription_queue.put_nowait("STT ready")
            except Exception as e:
                msg = f"STT disabled: {e}"
                print(msg)
                try:
                    transcription_queue.put_nowait(msg)
                except queue.Full:
                    pass
                stt = None
        elif not enabled and stt is not None:
            stt.stop()
            stt = None

    # For output processing, we need to maintain an output buffer
    # We'll use a simple approach: process each input chunk and put result into output queue

    # To measure meter level, compute RMS of input chunk
    def compute_rms(chunk):
        # chunk shape: (frames, channels)
        if chunk.ndim == 1:
            return (np.sqrt(np.mean(chunk**2)), 0)
        else:
            left = chunk[:, 0]
            right = chunk[:, 1]
            return (np.sqrt(np.mean(left**2)), np.sqrt(np.mean(right**2)))

    def enqueue_quality(indata_chunk, output_chunk):
        if indata_chunk.ndim == 1:
            mic = indata_chunk.astype(np.float32, copy=False)
            out = output_chunk.reshape(-1).astype(np.float32, copy=False)
        else:
            mic = np.mean(indata_chunk.astype(np.float32, copy=False), axis=1)
            out = np.mean(output_chunk.astype(np.float32, copy=False), axis=1)

        eps = 1e-10
        mic_rms = float(np.sqrt(np.mean(mic * mic)) + eps)
        out_rms = float(np.sqrt(np.mean(out * out)) + eps)
        attenuation_db = float(20.0 * np.log10(out_rms / mic_rms))

        spec_in = np.abs(np.fft.rfft(mic * np.hanning(mic.size)))
        spec_out = np.abs(np.fft.rfft(out * np.hanning(out.size)))
        freqs = np.fft.rfftfreq(mic.size, d=1.0 / 16000.0)

        low_hz = float(config_obj.get('nr_band_low_hz') or 120.0)
        high_hz = float(config_obj.get('nr_band_high_hz') or 6000.0)
        band = (freqs >= low_hz) & (freqs <= high_hz)
        if np.any(band):
            in_band = float(np.mean((spec_in[band] ** 2)) + eps)
            out_band = float(np.mean((spec_out[band] ** 2)) + eps)
            reduction_db = float(10.0 * np.log10(in_band / out_band))
        else:
            reduction_db = 0.0

        nplot = min(64, spec_in.size)
        plot_in = spec_in[:nplot]
        plot_out = spec_out[:nplot]
        max_v = float(max(np.max(plot_in), np.max(plot_out), eps))
        plot_in = (plot_in / max_v).astype(float).tolist()
        plot_out = (plot_out / max_v).astype(float).tolist()

        payload = {
            'in_rms': mic_rms,
            'out_rms': out_rms,
            'attenuation_db': attenuation_db,
            'reduction_db': reduction_db,
            'band_low_hz': low_hz,
            'band_high_hz': high_hz,
            'spectrum_in': plot_in,
            'spectrum_out': plot_out,
        }
        try:
            quality_queue.put_nowait(payload)
        except queue.Full:
            pass

    last_quality_emit = 0.0

    # Main loop
    while True:
        ensure_stt_state()

        # Get next audio chunk
        try:
            indata = audio_io.get_input(block=True, timeout=0.1)
        except queue.Empty:
            continue

        # Feed STT as early as possible to minimize end-to-end latency.
        if indata.ndim == 1:
            stt_input = indata
        else:
            stt_input = indata[:, 0]

        if stt and config_obj.get('stt_enabled'):
            stt.feed_audio(stt_input)
            while True:
                text = stt.get_transcript(block=False)
                if not text:
                    break
                try:
                    transcription_queue.put_nowait(text)
                except queue.Full:
                    pass

        # Send meter update (max 5 times per second is enough)
        meter = compute_rms(indata)
        try:
            meter_queue.put_nowait(meter)
        except queue.Full:
            pass  # drop if UI can't keep up

        # Process each channel separately or stereo? For simplicity, treat each channel independently.
        # For a real hearing aid, you'd use binaural processing.
        # We'll process left and right separately.
        if indata.ndim == 1:
            # Mono
            aec_delay = int(config_obj.get('aec_delay_blocks') or 4)
            spk_ref = pick_best_reference(indata, spk_hist_l, indata.shape[0], delay_blocks=aec_delay)
            processed = process_channel(
                indata,
                config_obj,
                noise_reducer_l,
                compressor_l,
                feedback_canceller_l,
                spk_ref,
            )
            output = processed.reshape(-1, 1)
        else:
            left = indata[:, 0]
            right = indata[:, 1]
            aec_delay = int(config_obj.get('aec_delay_blocks') or 4)
            ref_l = pick_best_reference(left, spk_hist_l, left.shape[0], delay_blocks=aec_delay)
            ref_r = pick_best_reference(right, spk_hist_r, right.shape[0], delay_blocks=aec_delay)

            left_out = process_channel(
                left,
                config_obj,
                noise_reducer_l,
                compressor_l,
                feedback_canceller_l,
                ref_l,
            )
            right_out = process_channel(
                right,
                config_obj,
                noise_reducer_r,
                compressor_r,
                feedback_canceller_r,
                ref_r,
            )
            output = np.column_stack((left_out, right_out))

        # Apply output gain (supports amplification above 1.0).
        volume = float(config_obj.get('volume') or 1.0)
        volume = float(np.clip(volume, 0.0, 3.0))
        output = np.clip(output * volume, -0.98, 0.98)

        # Keep latest speaker signal as acoustic reference for next block.
        if output.ndim == 1 or output.shape[1] == 1:
            spk_hist_l.append(output.reshape(-1).astype(np.float32, copy=True))
        else:
            spk_hist_l.append(output[:, 0].astype(np.float32, copy=True))
            spk_hist_r.append(output[:, 1].astype(np.float32, copy=True))

        now = time.monotonic()
        if now - last_quality_emit >= 0.16:
            enqueue_quality(indata, output)
            last_quality_emit = now

        # Put processed audio into output queue
        audio_io.put_output(output)

        # Periodically apply config changes (double‑buffering)
        if config_obj.apply():
            # Update algorithm parameters if needed
            # For demo, just print
            print("Config updated")

def process_channel(audio, config_obj, noise_reducer, compressor, feedback_canceller, speaker_reference):
    """Apply the processing chain to one channel."""
    # 1. AEC using previous speaker output as reference.
    aec_enabled = bool(config_obj.get('aec_enabled'))
    aec_strength = float(config_obj.get('aec_strength') or 1.2)
    if aec_enabled:
        audio = feedback_canceller.process_block(audio, speaker_reference, strength=aec_strength)
    # 2. Noise reduction
    if config_obj.get('noise_reduction'):
        nr_strength = float(config_obj.get('noise_reduction_strength') or 1.0)
        nr_low = float(config_obj.get('nr_band_low_hz') or 120.0)
        nr_high = float(config_obj.get('nr_band_high_hz') or 6000.0)
        audio = noise_reducer.process(
            audio,
            strength=nr_strength,
            cepstral_smoothing=bool(config_obj.get('nr_cepstral_smoothing')),
            attack_release_split=bool(config_obj.get('nr_attack_release_split')),
            wind_mode=bool(config_obj.get('nr_wind_mode')),
            wind_sensitivity=float(config_obj.get('nr_wind_sensitivity') or 0.6),
            band_low_hz=nr_low,
            band_high_hz=nr_high,
        )
    # 3. Compression
    strength = float(config_obj.get('compression_strength') or 1.0)
    compressed = compressor.process(audio)
    if strength <= 1.0:
        # Dry/wet mix when strength is <= 1.
        audio = (1.0 - strength) * audio + strength * compressed
    else:
        # Extra compression emphasis when strength is > 1.
        audio = compressed + (strength - 1.0) * (compressed - audio)
    return audio

# ----------------------------------------------------------------------
def meter_publisher_thread():
    """Read meter queue and send to UI via server (which uses its own queue)."""
    # The server has its own background thread that reads from meter_queue.
    # This thread just passes values from our internal queue to the global one.
    # Actually, the server already reads from meter_queue. So we just need to ensure the queue is the same.
    pass  # Not needed; server's background thread reads meter_queue directly.

# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create shared config
    cfg = Config()

    # Create audio I/O
    audio = AudioIO(samplerate=16000, blocksize=64, channels=2)
    audio.start()

    # Start processing thread
    proc_thread = threading.Thread(target=audio_processing_thread, args=(cfg, audio), daemon=True)
    proc_thread.start()

    # Start Flask server with WebSocket (runs in main thread)
    # Pass the global queues and config
    start_server(config_obj=cfg, meter_q=meter_queue, trans_q=transcription_queue, quality_q=quality_queue)