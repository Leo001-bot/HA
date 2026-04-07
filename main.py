import threading
import time
import os
import platform
import numpy as np
import queue
import traceback
from pathlib import Path

from config import Config
from processing import NoiseReducer, Compressor, OutputLimiter, SpeechEQ
from stt import create_streaming_stt
from cpp_bridge import CppRealtimeBridge
from server import start_server


meter_queue = queue.Queue(maxsize=5)
transcription_queue = queue.Queue()
quality_queue = queue.Queue(maxsize=5)
runtime_diag = {
    'stt': {},
    'audio_restarts': 0,
    'last_audio_restart_reason': '',
    'last_input_age_s': None,
    'input_frames_seen': 0,
    'output_frames_sent': 0,
    'audio_blocksize_in_use': 512,
    'processing_thread_alive': False,
    'processing_thread_error': '',
    'bridge_mode': False,
    'bridge': {},
}


def process_channel(audio, config_obj, noise_reducer, compressor, eq):
    nr_applied = False
    if bool(config_obj.get('noise_reduction')):
        nr_strength = float(config_obj.get('noise_reduction_strength') or 1.0)
        nr_low = float(config_obj.get('nr_band_low_hz') or 120.0)
        nr_high = float(config_obj.get('nr_band_high_hz') or 6000.0)
        audio = noise_reducer.process(
            audio,
            strength=nr_strength,
            cepstral_smoothing=bool(config_obj.get('nr_cepstral_smoothing')),
            attack_release_split=bool(config_obj.get('nr_attack_release_split')),
            band_low_hz=nr_low,
            band_high_hz=nr_high,
        )
        nr_applied = True

    if bool(config_obj.get('eq_enabled')):
        audio = eq.process(
            audio,
            bass_db=float(config_obj.get('eq_bass_db') or 0.0),
            presence_db=float(config_obj.get('eq_presence_db') or 0.0),
            treble_db=float(config_obj.get('eq_treble_db') or 0.0),
        )

    strength = float(config_obj.get('compression_strength') or 1.0)
    compressed = compressor.process(audio)
    if strength <= 1.0:
        audio = (1.0 - strength) * audio + strength * compressed
    else:
        audio = compressed + (strength - 1.0) * (compressed - audio)
    return audio, nr_applied


def audio_processing_thread(config_obj, audio_io):
    try:
        noise_reducer_l = NoiseReducer()
        noise_reducer_r = NoiseReducer()
        compressor_l = Compressor()
        compressor_r = Compressor()
        eq_l = SpeechEQ(sample_rate=16000)
        eq_r = SpeechEQ(sample_rate=16000)
        output_limiter = OutputLimiter(threshold=0.95, attack_ms=2.0, release_ms=50.0, sample_rate=16000)
    except Exception as init_err:
        runtime_diag['processing_thread_alive'] = False
        runtime_diag['processing_thread_error'] = f'Init failed: {init_err}'
        print(f"Audio processing init failed: {init_err}")
        traceback.print_exc()
        try:
            transcription_queue.put_nowait(f"Audio processing init failed: {init_err}")
        except queue.Full:
            pass
        return

    stt = None
    stt_model_root_in_use = None
    last_stt_init_attempt = 0.0
    stt_init_thread = None
    stt_init_queue = queue.Queue(maxsize=1)
    stt_init_target_root = None
    last_blocksize_in_use = int(audio_io.blocksize)
    last_quality_emit = 0.0
    last_idle_emit = 0.0
    hp_prev_x_l = 0.0
    hp_prev_y_l = 0.0
    hp_prev_x_r = 0.0
    hp_prev_y_r = 0.0
    stt_prev_x = 0.0
    stt_prev_y = 0.0
    input_seen_logged = False
    last_input_seen_ts = time.monotonic()
    last_audio_restart_attempt = 0.0

    print(f"Audio processing thread started (audio stream running: {audio_io._running})")
    runtime_diag['processing_thread_alive'] = True
    runtime_diag['processing_thread_error'] = ''
    if not audio_io._running:
        msg = "Audio I/O failed to start; microphone/speaker path unavailable"
        print(msg)
        try:
            transcription_queue.put_nowait(msg)
        except queue.Full:
            pass
    
    def is_mono_chunk(chunk):
        return chunk.ndim == 1 or (chunk.ndim == 2 and chunk.shape[1] == 1)

    def compute_rms(chunk):
        if is_mono_chunk(chunk):
            mono = chunk.reshape(-1)
            # Mirror mono level to both sides so UI does not look left-only.
            level = float(np.sqrt(np.mean(mono * mono)))
            return (level, level)
        left = chunk[:, 0]
        right = chunk[:, 1]
        return (float(np.sqrt(np.mean(left * left))), float(np.sqrt(np.mean(right * right))))

    def prepare_stt_audio(chunk):
        nonlocal stt_prev_x, stt_prev_y
        if is_mono_chunk(chunk):
            mono = chunk.reshape(-1).astype(np.float32, copy=False)
        else:
            # Prefer L+R averaging for voice naturalness; fallback to stronger side if cancellation is detected.
            stereo = chunk.astype(np.float32, copy=False)
            left = stereo[:, 0]
            right = stereo[:, 1]
            l_rms = float(np.sqrt(np.mean(left * left) + 1e-12))
            r_rms = float(np.sqrt(np.mean(right * right) + 1e-12))
            mixed = 0.5 * (left + right)
            m_rms = float(np.sqrt(np.mean(mixed * mixed) + 1e-12))
            dominant = max(l_rms, r_rms)
            if dominant > 1e-6 and m_rms < 0.72 * dominant:
                mono = left if l_rms >= r_rms else right
            else:
                mono = mixed

        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
        # Gentle DC blocker to suppress hum while preserving voice body.
        if mono.size:
            y = np.empty_like(mono)
            r = 0.996
            px = stt_prev_x
            py = stt_prev_y
            for i, x in enumerate(mono):
                yi = x - px + r * py
                y[i] = yi
                px = float(x)
                py = float(yi)
            stt_prev_x = px
            stt_prev_y = py
            mono = y

        rms = float(np.sqrt(np.mean(mono * mono) + 1e-12))
        # Keep STT path permissive for low-level speech from distant/quiet mics.
        target_rms = 0.070
        auto_gain_max = 16.0 if rms < 0.0030 else 8.0
        gain = float(np.clip(target_rms / (rms + 1e-9), 1.0, auto_gain_max))
        if rms < 0.00035:
            # Avoid over-amplifying pure hiss while still allowing weak speech through.
            gain *= 0.6
        user_gain = float(config_obj.get('stt_input_gain') or 1.0)
        user_gain = float(np.clip(user_gain, 0.5, 8.0))
        stt_audio = mono * gain * user_gain
        # Very gentle limiter; preserves phoneme detail better than hard clipping.
        return np.tanh(stt_audio * 1.05).astype(np.float32)

    def ensure_stt_state():
        nonlocal stt, stt_model_root_in_use, last_stt_init_attempt
        nonlocal stt_init_thread, stt_init_target_root

        def resolve_model_root(raw_value):
            if raw_value is None:
                raw_value = "models"
            value = str(raw_value).strip()
            if not value or value.lower() == "none":
                value = "models"
            path = Path(value)
            if not path.is_absolute():
                path = Path.cwd() / path
            return str(path.resolve())

        def _stt_init_worker(target_root):
            try:
                stt_backend = os.environ.get("STT_BACKEND", "auto")
                stt_obj = create_streaming_stt(model_root=target_root, sample_rate=16000, backend=stt_backend)
                stt_obj.start()
                stt_init_queue.put_nowait((True, stt_obj, target_root, ""))
            except Exception as init_err:
                stt_init_queue.put_nowait((False, None, target_root, str(init_err)))

        # Consume finished background init results first.
        try:
            ok, stt_obj, target_root, err_text = stt_init_queue.get_nowait()
            stt_init_thread = None
            if ok:
                # Apply only if root is still desired and STT is still enabled.
                if bool(config_obj.get('stt_enabled')) and resolve_model_root(config_obj.get('stt_model_root')) == target_root:
                    if stt is not None:
                        try:
                            stt.stop()
                        except Exception:
                            pass
                    stt = stt_obj
                    stt_model_root_in_use = target_root
                    runtime_diag['stt'] = stt.get_diagnostics()
                    try:
                        transcription_queue.put_nowait(f"STT ready ({target_root})")
                    except queue.Full:
                        pass
                else:
                    try:
                        stt_obj.stop()
                    except Exception:
                        pass
            else:
                stt_model_root_in_use = None
                msg = f"STT disabled (path={target_root}): {err_text}"
                print(msg)
                runtime_diag['stt'] = {'running': False, 'last_error': err_text, 'model_root': target_root}
                try:
                    transcription_queue.put_nowait(msg)
                except queue.Full:
                    pass
        except queue.Empty:
            pass

        enabled = bool(config_obj.get('stt_enabled'))

        desired_root = resolve_model_root(config_obj.get('stt_model_root'))

        if enabled and stt is not None and stt_model_root_in_use != desired_root:
            try:
                stt.stop()
            except Exception:
                pass
            stt = None
            stt_model_root_in_use = None
            try:
                transcription_queue.put_nowait(f"STT restarting with model path: {desired_root}")
            except queue.Full:
                pass

        if enabled and stt is None:
            now = time.monotonic()
            if now - last_stt_init_attempt < 5.0:
                return
            # Don't block the audio loop while STT model is loading.
            if stt_init_thread is None or (not stt_init_thread.is_alive()):
                last_stt_init_attempt = now
                stt_init_target_root = desired_root
                stt_init_thread = threading.Thread(
                    target=_stt_init_worker,
                    args=(desired_root,),
                    daemon=True,
                )
                stt_init_thread.start()
                runtime_diag['stt'] = {
                    'running': False,
                    'initializing': True,
                    'model_root': desired_root,
                }
                try:
                    transcription_queue.put_nowait(f"STT initializing ({desired_root})")
                except queue.Full:
                    pass
        elif not enabled and stt is not None:
            stt.stop()
            stt = None
            stt_model_root_in_use = None
            runtime_diag['stt'] = {'running': False, 'enabled': False}

    def ensure_audio_blocksize():
        nonlocal last_blocksize_in_use
        desired = int(config_obj.get('audio_blocksize') or audio_io.blocksize)
        desired = max(32, min(2048, desired))
        desired = int(round(desired / 32.0) * 32)
        runtime_diag['audio_blocksize_in_use'] = int(last_blocksize_in_use)
        if desired == last_blocksize_in_use:
            return

        runtime_diag['audio_blocksize_in_use'] = int(desired)
        try:
            audio_io.stop()
            audio_io.blocksize = desired
            time.sleep(0.15)
            audio_io.start()
            last_blocksize_in_use = desired
            runtime_diag['audio_blocksize_in_use'] = int(desired)
            runtime_diag['audio_restarts'] = int(runtime_diag.get('audio_restarts', 0)) + 1
            runtime_diag['last_audio_restart_reason'] = f'Audio blocksize changed to {desired}'
            try:
                transcription_queue.put_nowait(f'Audio block size changed to {desired}')
            except queue.Full:
                pass
        except Exception as e:
            runtime_diag['last_audio_restart_reason'] = f'Audio blocksize restart failed: {e}'

    def enqueue_quality(indata_chunk, output_chunk, nr_applied):
        if is_mono_chunk(indata_chunk):
            mic = indata_chunk.reshape(-1).astype(np.float32, copy=False)
            out = output_chunk.reshape(-1).astype(np.float32, copy=False)
        else:
            mic = np.mean(indata_chunk.astype(np.float32, copy=False), axis=1)
            out = np.mean(output_chunk.astype(np.float32, copy=False), axis=1)

        # Keep spectra aligned even if callback/input and output frame lengths differ.
        n = int(min(mic.size, out.size))
        if n < 8:
            return
        mic = mic[:n]
        out = out[:n]

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
        if bool(nr_applied) and np.any(band):
            in_band = float(np.mean(spec_in[band] ** 2) + eps)
            out_band = float(np.mean(spec_out[band] ** 2) + eps)
            reduction_db = float(10.0 * np.log10(in_band / out_band))
        else:
            reduction_db = 0.0

        nplot = min(64, spec_in.size)
        plot_in = (spec_in[:nplot] / max(float(np.max(spec_in[:nplot])), 1e-10)).astype(float).tolist()
        if bool(nr_applied):
            plot_out = (spec_out[:nplot] / max(float(np.max(spec_out[:nplot])), 1e-10)).astype(float).tolist()
        else:
            # Mirror input when NR is off so effectiveness chart doesn't imply filtering is active.
            plot_out = list(plot_in)

        payload = {
            'in_rms': mic_rms,
            'out_rms': out_rms,
            'attenuation_db': attenuation_db,
            'reduction_db': reduction_db,
            'band_low_hz': low_hz,
            'band_high_hz': high_hz,
            'nr_active': bool(nr_applied),
            'spectrum_in': plot_in,
            'spectrum_out': plot_out,
        }
        try:
            quality_queue.put_nowait(payload)
        except queue.Full:
            pass

    def enqueue_idle_visuals():
        """Keep UI meters/graphs alive when no input frames are available."""
        try:
            meter_queue.put_nowait((0.0, 0.0))
        except queue.Full:
            pass

        payload = {
            'in_rms': 0.0,
            'out_rms': 0.0,
            'attenuation_db': 0.0,
            'reduction_db': 0.0,
            'band_low_hz': float(config_obj.get('nr_band_low_hz') or 120.0),
            'band_high_hz': float(config_obj.get('nr_band_high_hz') or 6000.0),
            'nr_active': bool(config_obj.get('noise_reduction')),
            'spectrum_in': [0.0] * 64,
            'spectrum_out': [0.0] * 64,
        }
        try:
            quality_queue.put_nowait(payload)
        except queue.Full:
            pass

    while True:
        try:
            ensure_stt_state()
            ensure_audio_blocksize()
            bypass_all = bool(config_obj.get('bypass_all'))
            # Direct monitor path: raw input is copied to output callback without DSP queueing.
            audio_io.set_passthrough(bypass_all, 1.0)
            if stt is not None:
                runtime_diag['stt'] = stt.get_diagnostics()

            try:
                indata = audio_io.get_input(block=True, timeout=0.1)
            except queue.Empty:
                now = time.monotonic()
                runtime_diag['last_input_age_s'] = float(now - last_input_seen_ts)
                if now - last_idle_emit >= 0.25:
                    enqueue_idle_visuals()
                    last_idle_emit = now

                # Attempt to recover audio devices if callbacks stop delivering frames.
                if audio_io._running and (now - last_input_seen_ts) > 4.0 and (now - last_audio_restart_attempt) > 10.0:
                    last_audio_restart_attempt = now
                    try:
                        audio_io.stop()
                        time.sleep(0.2)
                        audio_io.start()
                        runtime_diag['audio_restarts'] = int(runtime_diag.get('audio_restarts', 0)) + 1
                        runtime_diag['last_audio_restart_reason'] = 'No input frames for >4s; restarted audio stream'
                        transcription_queue.put_nowait('Audio stream restarted after input stall')
                    except Exception as restart_err:
                        runtime_diag['last_audio_restart_reason'] = f'Audio restart failed: {restart_err}'
                continue

            # Drop stale mic frames and keep only the most recent chunk for low-latency feedback.
            while True:
                try:
                    indata = audio_io.get_input(block=False)
                except queue.Empty:
                    break

            last_input_seen_ts = time.monotonic()
            runtime_diag['last_input_age_s'] = 0.0
            runtime_diag['input_frames_seen'] = int(runtime_diag.get('input_frames_seen', 0)) + int(indata.shape[0])

            if not input_seen_logged:
                print(f"Audio input detected: shape={getattr(indata, 'shape', None)}")
                input_seen_logged = True

            stt_input = prepare_stt_audio(indata)
            if stt and config_obj.get('stt_enabled'):
                runtime_diag['stt'] = stt.get_diagnostics()
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

            if bypass_all:
                now = time.monotonic()
                if now - last_quality_emit >= 0.16:
                    enqueue_quality(indata, indata, False)
                    last_quality_emit = now
                runtime_diag['output_frames_sent'] = int(runtime_diag.get('output_frames_sent', 0)) + int(indata.shape[0])
                if config_obj.apply():
                    print("Config updated")
                continue

            # Remove low-frequency current/hum on input before processing.
            if is_mono_chunk(indata):
                x = indata.reshape(-1).astype(np.float32, copy=False)
                y = np.empty_like(x)
                r = 0.98
                px = hp_prev_x_l
                py = hp_prev_y_l
                for i, s in enumerate(x):
                    yi = s - px + r * py
                    y[i] = yi
                    px = float(s)
                    py = float(yi)
                hp_prev_x_l = px
                hp_prev_y_l = py
                indata_proc = y.reshape(-1, 1)
            else:
                left_raw = indata[:, 0].astype(np.float32, copy=False)
                right_raw = indata[:, 1].astype(np.float32, copy=False)
                y_l = np.empty_like(left_raw)
                y_r = np.empty_like(right_raw)
                r = 0.98

                px = hp_prev_x_l
                py = hp_prev_y_l
                for i, s in enumerate(left_raw):
                    yi = s - px + r * py
                    y_l[i] = yi
                    px = float(s)
                    py = float(yi)
                hp_prev_x_l = px
                hp_prev_y_l = py

                px = hp_prev_x_r
                py = hp_prev_y_r
                for i, s in enumerate(right_raw):
                    yi = s - px + r * py
                    y_r[i] = yi
                    px = float(s)
                    py = float(yi)
                hp_prev_x_r = px
                hp_prev_y_r = py

                indata_proc = np.column_stack((y_l, y_r))

            nr_active = False

            if is_mono_chunk(indata_proc):
                mono = indata_proc.reshape(-1)
                processed, nr_active = process_channel(mono, config_obj, noise_reducer_l, compressor_l, eq_l)
                # Duplicate mono processed audio to stereo for both ear sides.
                output = np.column_stack((processed, processed))
            else:
                left = indata_proc[:, 0]
                right = indata_proc[:, 1]
                left_out, nr_left = process_channel(left, config_obj, noise_reducer_l, compressor_l, eq_l)
                right_out, nr_right = process_channel(right, config_obj, noise_reducer_r, compressor_r, eq_r)
                nr_active = bool(nr_left or nr_right)
                output = np.column_stack((left_out, right_out))

            volume = float(np.clip(float(config_obj.get('volume') or 1.0), 0.0, 6.0))
            output = output * volume

            # Downward expansion suppresses residual electrical current noise in quiet frames.
            out_rms = float(np.sqrt(np.mean(output * output) + 1e-12))
            if out_rms < 0.008:
                output *= float(np.clip((out_rms / 0.008) ** 1.4, 0.0, 1.0))

            # Apply adaptive output limiter to prevent distortion and clipping.
            if output.ndim == 1 or output.shape[1] == 1:
                # Mono output
                mono_out = output.reshape(-1)
                output = output_limiter.process(mono_out)
            else:
                # Stereo output - apply limiter to combined signal for consistent limiting
                combined = np.mean(output, axis=1)
                limited_combined = output_limiter.process(combined)
                # Scale stereo channels proportionally to maintain balance
                scale_factor = np.where(
                    np.abs(combined) > 1e-6,
                    np.abs(limited_combined) / (np.abs(combined) + 1e-6),
                    1.0
                )
                output = output * scale_factor[:, np.newaxis]
            
            # Final soft clipping to ensure we stay within [-1, 1] range.
            output = np.tanh(output * 1.05) / np.tanh(1.05)
            output = np.clip(output, -0.95, 0.95)

            now = time.monotonic()
            if now - last_quality_emit >= 0.16:
                enqueue_quality(indata, output, nr_active)
                last_quality_emit = now

            audio_io.put_output(output)
            runtime_diag['output_frames_sent'] = int(runtime_diag.get('output_frames_sent', 0)) + int(output.shape[0])
            if config_obj.apply():
                print("Config updated")

        except Exception as e:
            print(f"Audio processing error (continuing): {e}")
            runtime_diag['processing_thread_error'] = str(e)
            import traceback
            traceback.print_exc()
            time.sleep(0.1)


if __name__ == "__main__":
    audio = None
    cpp_bridge = None
    try:
        cfg = Config()
        use_cpp_bridge = str(os.environ.get('USE_CPP_BRIDGE', '0')).strip().lower() in ('1', 'true', 'yes', 'on')
        if use_cpp_bridge:
            cpp_bridge = CppRealtimeBridge()
            cpp_bridge.start()
            runtime_diag['bridge_mode'] = True
            runtime_diag['bridge'] = cpp_bridge.get_diagnostics()
            print("C++ bridge mode enabled; Python audio pipeline is skipped.")
        else:
            from audio_io import AudioIO
            runtime_diag['bridge_mode'] = False
        is_linux_arm = platform.system().lower() == 'linux' and platform.machine().lower() in ('armv7l', 'aarch64', 'arm64')
        input_channels = 1 if is_linux_arm else 2
        output_channels = 1 if is_linux_arm else 2
        blocksize = int(cfg.get('audio_blocksize') or 512)
        if is_linux_arm and blocksize < 1024:
            blocksize = 1024
            cfg.set('audio_blocksize', blocksize)
        if not use_cpp_bridge:
            # Linux ARM defaults are conservative to avoid PortAudio/ALSA callback crashes.
            audio = AudioIO(samplerate=16000, blocksize=blocksize, input_channels=input_channels, output_channels=output_channels)
            audio.start()
            proc_thread = threading.Thread(target=audio_processing_thread, args=(cfg, audio), daemon=True)
            proc_thread.start()
        else:
            proc_thread = None

        def diagnostics_provider():
            payload = {
                'audio': audio.get_stats() if audio is not None else {'running': False},
                'runtime': dict(runtime_diag),
                'processing_thread_alive': bool(proc_thread.is_alive()) if proc_thread is not None else False,
            }
            if cpp_bridge is not None:
                payload['bridge'] = cpp_bridge.get_diagnostics()
            return payload

        start_server(
            config_obj=cfg,
            meter_q=meter_queue,
            trans_q=transcription_queue,
            quality_q=quality_queue,
            diagnostics_provider_fn=diagnostics_provider,
        )
    except KeyboardInterrupt:
        print("Shutdown requested by user")
    except Exception as e:
        print(f"FATAL startup/runtime error: {e}")
        traceback.print_exc()
        raise
    finally:
        if cpp_bridge is not None:
            try:
                cpp_bridge.stop()
            except Exception:
                pass
        if audio is not None:
            try:
                audio.stop()
            except Exception:
                pass