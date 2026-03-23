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


meter_queue = queue.Queue(maxsize=5)
transcription_queue = queue.Queue()
quality_queue = queue.Queue(maxsize=5)


def process_channel(audio, config_obj, noise_reducer, compressor, feedback_canceller, speaker_reference):
    aec_enabled = bool(config_obj.get('aec_enabled'))
    aec_strength = float(config_obj.get('aec_strength') or 1.2)
    near_end_mode = bool(config_obj.get('near_end_suppression_mode'))
    near_end_strength = float(config_obj.get('near_end_suppression_strength') or 0.6)
    if aec_enabled:
        audio = feedback_canceller.process_block(
            audio,
            speaker_reference,
            strength=aec_strength,
            near_end_mode=near_end_mode,
            near_end_strength=near_end_strength,
        )

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

    strength = float(config_obj.get('compression_strength') or 1.0)
    compressed = compressor.process(audio)
    if strength <= 1.0:
        audio = (1.0 - strength) * audio + strength * compressed
    else:
        audio = compressed + (strength - 1.0) * (compressed - audio)
    return audio


def audio_processing_thread(config_obj, audio_io):
    noise_reducer_l = NoiseReducer()
    noise_reducer_r = NoiseReducer()
    compressor_l = Compressor()
    compressor_r = Compressor()
    feedback_canceller_l = FeedbackCanceller(filter_length=1024, mu=0.025, sample_rate=16000)
    feedback_canceller_r = FeedbackCanceller(filter_length=1024, mu=0.025, sample_rate=16000)
    spk_hist_l = deque(maxlen=24)
    spk_hist_r = deque(maxlen=24)

    stt = None
    last_stt_init_attempt = 0.0
    last_quality_emit = 0.0

    def is_mono_chunk(chunk):
        return chunk.ndim == 1 or (chunk.ndim == 2 and chunk.shape[1] == 1)

    def pick_best_reference(mic_block, history, fallback_len, delay_blocks=4):
        if not history:
            return np.zeros(fallback_len, dtype=np.float32)
        mic = mic_block.astype(np.float32, copy=False)
        mic_norm = float(np.linalg.norm(mic)) + 1e-8
        mic_rms = float(np.sqrt(np.mean(mic * mic)) + 1e-8)

        delay_blocks = int(np.clip(delay_blocks, 1, max(1, len(history))))
        base_idx = len(history) - delay_blocks
        start_idx = max(0, base_idx - 2)
        end_idx = min(len(history) - 1, base_idx + 2)

        best = history[max(0, min(len(history) - 1, base_idx))]
        best_score = -1.0
        for i in range(start_idx, end_idx + 1):
            ref = history[i]
            if ref.shape[0] != fallback_len:
                continue
            ref_rms = float(np.sqrt(np.mean(ref * ref)) + 1e-8)
            if ref_rms < 1e-5:
                continue
            ref_norm = float(np.linalg.norm(ref)) + 1e-8
            score = abs(float(np.dot(mic, ref))) / (mic_norm * ref_norm)
            score *= min(ref_rms / mic_rms, mic_rms / ref_rms)
            if score > best_score:
                best_score = score
                best = ref
        return best

    def compute_rms(chunk):
        if is_mono_chunk(chunk):
            mono = chunk.reshape(-1)
            return (float(np.sqrt(np.mean(mono * mono))), 0.0)
        left = chunk[:, 0]
        right = chunk[:, 1]
        return (float(np.sqrt(np.mean(left * left))), float(np.sqrt(np.mean(right * right))))

    def prepare_stt_audio(chunk):
        if is_mono_chunk(chunk):
            mono = chunk.reshape(-1).astype(np.float32, copy=False)
        else:
            # Mix channels so STT still works when voice is mostly on one side.
            mono = np.mean(chunk.astype(np.float32, copy=False), axis=1)

        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
        rms = float(np.sqrt(np.mean(mono * mono) + 1e-12))
        if rms < 1e-4:
            return mono

        # Adaptive pre-gain improves recognition on quiet PC microphones.
        target_rms = 0.08
        gain = float(np.clip(target_rms / (rms + 1e-9), 1.0, 12.0))
        user_gain = float(config_obj.get('stt_input_gain') or 1.0)
        user_gain = float(np.clip(user_gain, 0.5, 8.0))
        return np.clip(mono * gain * user_gain, -1.0, 1.0)

    def ensure_stt_state():
        nonlocal stt, last_stt_init_attempt
        enabled = bool(config_obj.get('stt_enabled'))
        if enabled and stt is None:
            now = time.monotonic()
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

    def enqueue_quality(indata_chunk, output_chunk):
        if is_mono_chunk(indata_chunk):
            mic = indata_chunk.reshape(-1).astype(np.float32, copy=False)
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
            in_band = float(np.mean(spec_in[band] ** 2) + eps)
            out_band = float(np.mean(spec_out[band] ** 2) + eps)
            reduction_db = float(10.0 * np.log10(in_band / out_band))
        else:
            reduction_db = 0.0

        nplot = min(64, spec_in.size)
        plot_in = (spec_in[:nplot] / max(float(np.max(spec_in[:nplot])), 1e-10)).astype(float).tolist()
        plot_out = (spec_out[:nplot] / max(float(np.max(spec_out[:nplot])), 1e-10)).astype(float).tolist()

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

    while True:
        ensure_stt_state()

        try:
            indata = audio_io.get_input(block=True, timeout=0.1)
        except queue.Empty:
            continue

        stt_input = prepare_stt_audio(indata)
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

        try:
            meter_queue.put_nowait(compute_rms(indata))
        except queue.Full:
            pass

        aec_delay = int(config_obj.get('aec_delay_blocks') or 4)
        if is_mono_chunk(indata):
            mono = indata.reshape(-1)
            spk_ref = pick_best_reference(mono, spk_hist_l, mono.shape[0], delay_blocks=aec_delay)
            processed = process_channel(mono, config_obj, noise_reducer_l, compressor_l, feedback_canceller_l, spk_ref)
            output = processed.reshape(-1, 1)
        else:
            left = indata[:, 0]
            right = indata[:, 1]
            ref_l = pick_best_reference(left, spk_hist_l, left.shape[0], delay_blocks=aec_delay)
            ref_r = pick_best_reference(right, spk_hist_r, right.shape[0], delay_blocks=aec_delay)
            left_out = process_channel(left, config_obj, noise_reducer_l, compressor_l, feedback_canceller_l, ref_l)
            right_out = process_channel(right, config_obj, noise_reducer_r, compressor_r, feedback_canceller_r, ref_r)
            output = np.column_stack((left_out, right_out))

        volume = float(np.clip(float(config_obj.get('volume') or 1.0), 0.0, 3.0))
        output = np.clip(output * volume, -0.98, 0.98)

        if output.ndim == 1 or output.shape[1] == 1:
            spk_hist_l.append(output.reshape(-1).astype(np.float32, copy=True))
        else:
            spk_hist_l.append(output[:, 0].astype(np.float32, copy=True))
            spk_hist_r.append(output[:, 1].astype(np.float32, copy=True))

        now = time.monotonic()
        if now - last_quality_emit >= 0.16:
            enqueue_quality(indata, output)
            last_quality_emit = now

        audio_io.put_output(output)
        if config_obj.apply():
            print("Config updated")


if __name__ == "__main__":
    cfg = Config()
    audio = AudioIO(samplerate=16000, blocksize=64, input_channels=1, output_channels=2)
    audio.start()
    proc_thread = threading.Thread(target=audio_processing_thread, args=(cfg, audio), daemon=True)
    proc_thread.start()
    start_server(config_obj=cfg, meter_q=meter_queue, trans_q=transcription_queue, quality_q=quality_queue)