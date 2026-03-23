import queue
import threading
import os
import time
from pathlib import Path

import numpy as np
import sherpa_onnx


class StreamingSTT:
    """Streaming STT using Sherpa-ONNX with model auto-discovery."""

    def __init__(self, model_path=None, tokens_path=None, sample_rate=16000, model_root="models"):
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue(maxsize=32)
        self.text_queue = queue.Queue(maxsize=64)
        self._running = False
        self._thread = None
        self._last_text = ""
        self._current_text = ""
        self._last_emit_time = 0.0
        self._silence_seconds = 0.0
        self._last_final_text = ""
        self._last_final_time = 0.0

        # Tunables for real-time behavior.
        self._silence_threshold = 0.008
        self._utterance_silence_seconds = 0.45
        self._partial_emit_interval = 0.08
        self._duplicate_final_window_seconds = 3.0

        self.recognizer = self._create_recognizer(
            model_path=model_path,
            tokens_path=tokens_path,
            model_root=model_root,
        )
        self.stream = self.recognizer.create_stream()

    def _create_recognizer(self, model_path=None, tokens_path=None, model_root="models"):
        root = Path(model_root)
        if not root.exists():
            raise FileNotFoundError(f"Model directory not found: {root}")

        if model_path and tokens_path:
            model_file = Path(model_path)
            tokens_file = Path(tokens_path)
            if model_file.is_file() and tokens_file.is_file():
                return sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
                    model=str(model_file),
                    tokens=str(tokens_file),
                    num_threads=2,
                    provider="cpu",
                    sample_rate=self.sample_rate,
                    feature_dim=80,
                )

        transducer = self._find_transducer_bundle(root)
        if transducer is not None:
            num_threads = max(2, min(8, (os.cpu_count() or 4) // 2))
            return sherpa_onnx.OnlineRecognizer.from_transducer(
                encoder=str(transducer["encoder"]),
                decoder=str(transducer["decoder"]),
                joiner=str(transducer["joiner"]),
                tokens=str(transducer["tokens"]),
                num_threads=num_threads,
                provider="cpu",
                sample_rate=self.sample_rate,
                feature_dim=80,
                decoding_method="greedy_search",
            )

        zipformer_ctc = self._find_zipformer2_ctc_bundle(root)
        if zipformer_ctc is not None:
            num_threads = max(2, min(8, (os.cpu_count() or 4) // 2))
            return sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
                model=str(zipformer_ctc["model"]),
                tokens=str(zipformer_ctc["tokens"]),
                num_threads=num_threads,
                provider="cpu",
                sample_rate=self.sample_rate,
                feature_dim=80,
                decoding_method="greedy_search",
            )

        raise FileNotFoundError(
            "No compatible STT model files found. Expected one of: "
            "(encoder*.onnx + decoder*.onnx + joiner*.onnx + tokens.txt) or "
            "(ctc*.onnx + tokens.txt) under models/."
        )

    def _find_transducer_bundle(self, root: Path):
        tokens_candidates = list(root.rglob("tokens.txt"))
        for tokens in tokens_candidates:
            parent = tokens.parent
            encoders = sorted(parent.glob("encoder*.onnx"))
            decoders = sorted(parent.glob("decoder*.onnx"))
            joiners = sorted(parent.glob("joiner*.onnx"))
            if encoders and decoders and joiners:
                return {
                    "tokens": tokens,
                    "encoder": encoders[0],
                    "decoder": decoders[0],
                    "joiner": joiners[0],
                }
        return None

    def _find_zipformer2_ctc_bundle(self, root: Path):
        tokens_candidates = list(root.rglob("tokens.txt"))
        for tokens in tokens_candidates:
            parent = tokens.parent
            ctc_models = sorted(parent.glob("ctc*.onnx"))
            if ctc_models:
                return {
                    "tokens": tokens,
                    "model": ctc_models[0],
                }
        return None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def feed_audio(self, audio_data):
        """Feed a chunk of audio (numpy array, float32, range -1..1)."""
        try:
            self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(audio_data)
            except queue.Empty:
                pass

    def get_transcript(self, block=False, timeout=None):
        """Get latest transcribed text chunk."""
        try:
            return self.text_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def _process_loop(self):
        def emit_final_and_reset_stream():
            # Flush buffered states so the last words of an utterance are emitted.
            try:
                self.stream.input_finished()
                while self.recognizer.is_ready(self.stream):
                    self.recognizer.decode_stream(self.stream)
            except Exception:
                pass

            final_text = str(self.recognizer.get_result(self.stream)).strip()
            if final_text:
                normalized = " ".join(final_text.lower().split())
                now = time.monotonic()
                duplicate_recent = (
                    normalized == self._last_final_text
                    and (now - self._last_final_time) < self._duplicate_final_window_seconds
                )
                if not duplicate_recent:
                    try:
                        self.text_queue.put_nowait(final_text)
                    except queue.Full:
                        pass
                    self._last_final_text = normalized
                    self._last_final_time = now

            self.stream = self.recognizer.create_stream()
            self._current_text = ""
            self._last_text = ""
            self._silence_seconds = 0.0

        while self._running:
            try:
                first = self.audio_queue.get(timeout=0.005)
                chunks = [first]
                # Micro-batch queued chunks to reduce queue overhead and decode gaps.
                while len(chunks) < 4:
                    try:
                        chunks.append(self.audio_queue.get_nowait())
                    except queue.Empty:
                        break

                samples = np.asarray(np.concatenate(chunks), dtype=np.float32)
                if samples.ndim > 1:
                    samples = samples[:, 0]

                duration_s = float(samples.shape[0]) / float(self.sample_rate)
                rms = float(np.sqrt(np.mean(samples * samples) + 1e-12))

                self.stream.accept_waveform(self.sample_rate, samples)

                while self.recognizer.is_ready(self.stream):
                    self.recognizer.decode_stream(self.stream)

                text = str(self.recognizer.get_result(self.stream)).strip()

                if rms < self._silence_threshold:
                    self._silence_seconds += duration_s
                else:
                    self._silence_seconds = 0.0

                if text:
                    self._current_text = text

                # Emit partial updates at a controlled rate for low latency UI.
                now = time.monotonic()
                if self._current_text and self._current_text != self._last_text:
                    if now - self._last_emit_time >= self._partial_emit_interval:
                        self._last_text = self._current_text
                        self._last_emit_time = now
                        try:
                            self.text_queue.put_nowait(self._current_text)
                        except queue.Full:
                            pass

                # End-of-utterance: on sustained silence, reset stream automatically.
                if self._silence_seconds >= self._utterance_silence_seconds:
                    emit_final_and_reset_stream()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"STT error: {e}")
