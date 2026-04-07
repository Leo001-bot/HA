import queue
import os
import platform

import numpy as np
import sounddevice as sd


class AudioIO:
    """Handles real-time capture/playback with small queues."""

    def __init__(self, samplerate=16000, blocksize=64, channels=2, input_channels=None, output_channels=None):
        self.samplerate = samplerate
        self.blocksize = blocksize
        # Backward compatible: `channels` still works if explicit channel args are omitted.
        if input_channels is None:
            input_channels = channels
        if output_channels is None:
            output_channels = channels
        self.input_channels = int(max(1, input_channels))
        self.output_channels = int(max(1, output_channels))
        # Keep queues very short to prioritize low latency over continuity.
        self.input_queue = queue.Queue(maxsize=3)
        self.output_queue = queue.Queue(maxsize=3)
        self.stream = None
        self.input_stream = None
        self.output_stream = None
        self.input_only = False
        self._running = False
        self._passthrough_enabled = False
        self._passthrough_gain = 1.0
        self._latest_passthrough = None
        prefer_split_env = str(os.environ.get("AUDIO_PREFER_SPLIT", "auto")).strip().lower()
        is_linux_arm = platform.system().lower() == "linux" and platform.machine().lower() in ("armv7l", "aarch64", "arm64")
        if prefer_split_env == "1" or prefer_split_env == "true" or prefer_split_env == "yes":
            self._prefer_split_streams = True
        elif prefer_split_env == "0" or prefer_split_env == "false" or prefer_split_env == "no":
            self._prefer_split_streams = False
        else:
            self._prefer_split_streams = is_linux_arm
        self._stats = {
            "input_callbacks": 0,
            "output_callbacks": 0,
            "input_frames": 0,
            "output_frames": 0,
            "input_queue_drops": 0,
            "output_queue_drops": 0,
            "last_input_status": "",
            "last_output_status": "",
        }

    def set_passthrough(self, enabled, gain=1.0):
        self._passthrough_enabled = bool(enabled)
        try:
            self._passthrough_gain = float(gain)
        except (TypeError, ValueError):
            self._passthrough_gain = 1.0

    def _fill_passthrough(self, outdata, source):
        if source is None:
            outdata.fill(0)
            return

        latest = np.asarray(source, dtype=np.float32)
        if latest.ndim == 1:
            latest = latest.reshape(-1, 1)

        # Match output channel count.
        if latest.shape[1] == 1 and outdata.shape[1] > 1:
            latest = np.repeat(latest, outdata.shape[1], axis=1)
        elif latest.shape[1] > outdata.shape[1]:
            latest = latest[:, :outdata.shape[1]]
        elif latest.shape[1] < outdata.shape[1]:
            pad_ch = outdata.shape[1] - latest.shape[1]
            latest = np.hstack((latest, np.zeros((latest.shape[0], pad_ch), dtype=np.float32)))

        # Match frame count.
        if latest.shape[0] < outdata.shape[0]:
            pad = np.zeros((outdata.shape[0] - latest.shape[0], outdata.shape[1]), dtype=np.float32)
            latest = np.vstack((latest, pad))
        elif latest.shape[0] > outdata.shape[0]:
            latest = latest[:outdata.shape[0], :]

        gain = float(np.clip(self._passthrough_gain, 0.0, 8.0))
        outdata[:] = np.clip(latest * gain, -0.95, 0.95)

    def _resolve_channels(self):
        """Choose channel counts supported by current default input/output devices."""
        try:
            in_caps = sd.query_devices(kind='input')
            out_caps = sd.query_devices(kind='output')

            in_max = int(max(1, in_caps.get('max_input_channels', 1)))
            out_max = int(max(1, out_caps.get('max_output_channels', 1)))

            self.input_channels = int(max(1, min(self.input_channels, in_max)))
            self.output_channels = int(max(1, min(self.output_channels, out_max)))
            
            print(f"Audio devices: input_max={in_max}, output_max={out_max}, using: in={self.input_channels}, out={self.output_channels}")
        except Exception as e:
            print(f"Warning: Could not query audio devices, defaulting to mono: {e}")
            self.input_channels = 1
            self.output_channels = 1

    def _input_device_candidates(self):
        """Return candidate input device indices ordered with default first."""
        default_in = None
        ranked = []
        try:
            default_in, _ = sd.default.device
            if default_in is not None:
                default_in = int(default_in)
        except Exception:
            default_in = None

        try:
            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                in_ch = int(dev.get('max_input_channels', 0))
                if in_ch <= 0:
                    continue

                name = str(dev.get('name', '')).lower()
                score = 0

                # Prefer physical mic devices and better channel capability.
                if 'realtek' in name:
                    score += 30
                if '(realtek(r) audio)' in name:
                    score += 8
                if 'mic' in name or 'microphone' in name:
                    score += 20
                score += min(in_ch, 4) * 2

                # De-prioritize virtual/loopback/headset-handsfree style devices.
                if 'stereo mix' in name:
                    score -= 20
                if 'steam streaming' in name:
                    score -= 15
                if 'hands-free' in name or 'hands free' in name:
                    score -= 10
                if 'headset' in name and 'realtek' not in name:
                    score -= 4

                # Prefer modern primary device list entries over duplicated legacy aliases.
                if idx >= 20:
                    score -= 15
                if idx >= 30:
                    score -= 10

                if default_in is not None and idx == default_in:
                    score += 6

                ranked.append((score, idx))
        except Exception:
            pass

        ranked.sort(reverse=True)
        return [idx for _, idx in ranked]

    def _output_device_candidates(self):
        """Return candidate output device indices ordered with default first."""
        candidates = []
        try:
            _, default_out = sd.default.device
            if default_out is not None and int(default_out) >= 0:
                candidates.append(int(default_out))
        except Exception:
            pass

        try:
            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                if int(dev.get('max_output_channels', 0)) > 0 and idx not in candidates:
                    candidates.append(idx)
        except Exception:
            pass

        return candidates

    def _start_input_only_stream(self):
        """Try multiple input device/samplerate combinations."""
        candidates = self._input_device_candidates()
        rates_to_try = [self.samplerate]

        for dev_idx in candidates:
            for rate in rates_to_try:
                try:
                    dev = sd.query_devices(dev_idx)
                    dev_in_max = int(max(1, dev.get('max_input_channels', 1)))
                    use_in_ch = int(max(1, min(self.input_channels, dev_in_max)))
                    self.stream = sd.InputStream(
                        samplerate=rate,
                        blocksize=self.blocksize,
                        channels=use_in_ch,
                        device=dev_idx,
                        dtype=np.float32,
                        callback=self._input_callback,
                        latency='low',
                    )
                    self.stream.start()
                    self.input_stream = self.stream
                    self.input_channels = use_in_ch
                    self.input_only = True
                    print(f"Audio input-only stream started on device {dev_idx}: {self.input_channels} in @ {self.samplerate}Hz, blocksize={self.blocksize}")
                    return True
                except Exception:
                    continue

        return False

    def _start_output_only_stream(self):
        """Try to open an output stream so playback still works with split I/O."""
        candidates = self._output_device_candidates()
        rates_to_try = [self.samplerate]

        for dev_idx in candidates:
            for rate in rates_to_try:
                try:
                    dev = sd.query_devices(dev_idx)
                    dev_out_max = int(max(1, dev.get('max_output_channels', 1)))
                    use_out_ch = int(max(1, min(self.output_channels, dev_out_max)))
                    self.output_stream = sd.OutputStream(
                        samplerate=rate,
                        blocksize=self.blocksize,
                        channels=use_out_ch,
                        device=dev_idx,
                        dtype=np.float32,
                        callback=self._output_callback,
                        latency='low',
                    )
                    self.output_stream.start()
                    self.output_channels = use_out_ch
                    print(f"Audio output-only stream started on device {dev_idx}: {self.output_channels} out @ {rate}Hz, blocksize={self.blocksize}")
                    return True
                except Exception:
                    continue

        return False

    def _start_split_streams(self):
        """Start independent input/output streams for hardware that rejects duplex."""
        in_ok = self._start_input_only_stream()
        if not in_ok:
            return False

        out_ok = self._start_output_only_stream()
        if not out_ok:
            print("Warning: input stream is active but output stream could not start")
            return True
        self.input_only = False
        return True

    def start(self):
        self._running = True
        self.input_only = False
        self._resolve_channels()

        if self._prefer_split_streams:
            print("Preferring split I/O mode for stability")
            if self._start_split_streams():
                return

        # Some Windows MME drivers fail or hang on asymmetric duplex (e.g. 1-in/2-out).
        # Prefer split streams so mic input and speaker output can both remain available.
        if self.input_channels != self.output_channels:
            print(
                f"Skipping duplex open for asymmetric channels in={self.input_channels}, out={self.output_channels}; trying split I/O mode"
            )
            if self._start_split_streams():
                return

        try:
            self.stream = sd.Stream(
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                channels=(self.input_channels, self.output_channels),
                dtype=np.float32,
                callback=self._callback,
                latency='low',
            )
            self.stream.start()
            self.input_stream = self.stream
            self.output_stream = self.stream
            print(f"Audio stream started: {self.input_channels} in, {self.output_channels} out @ {self.samplerate}Hz, blocksize={self.blocksize}")
        except Exception as e:
            print(f"Error starting audio stream, trying mono fallback: {e}")
            self.input_channels = 1
            self.output_channels = 1
            try:
                self.stream = sd.Stream(
                    samplerate=self.samplerate,
                    blocksize=self.blocksize,
                    channels=(self.input_channels, self.output_channels),
                    dtype=np.float32,
                    callback=self._callback,
                    latency='low',
                )
                self.stream.start()
                self.input_stream = self.stream
                self.output_stream = self.stream
                print(f"Audio stream started (mono fallback): {self.input_channels} in, {self.output_channels} out @ {self.samplerate}Hz, blocksize={self.blocksize}")
            except Exception as e2:
                print(f"Could not start duplex stream, trying split I/O mode: {e2}")
                if not self._start_split_streams():
                    print("FATAL: Could not start input stream on any detected input device")
                    # Continue anyway - web server should still work even without audio
                    self._running = False

    def stop(self):
        self._running = False
        for s in (self.input_stream, self.output_stream, self.stream):
            if s is None:
                continue
            try:
                s.stop()
            except Exception:
                pass
            try:
                s.close()
            except Exception:
                pass
        self.input_stream = None
        self.output_stream = None
        self.stream = None

    def _push_input(self, indata):
        self._stats["input_frames"] += int(indata.shape[0])
        try:
            self.input_queue.put_nowait(indata.copy())
        except queue.Full:
            # Drop oldest chunk to keep latency bounded.
            self._stats["input_queue_drops"] += 1
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.input_queue.put_nowait(indata.copy())
            except queue.Full:
                pass

    def _fill_output(self, outdata):
        self._stats["output_frames"] += int(outdata.shape[0])
        # Play the newest available output frame and drop stale frames.
        latest = None
        while True:
            try:
                latest = self.output_queue.get_nowait()
            except queue.Empty:
                break

        if latest is None:
            outdata.fill(0)
            return

        latest = np.asarray(latest, dtype=np.float32)
        if latest.ndim == 1:
            latest = latest.reshape(-1, 1)

        # Match frame count.
        if latest.shape[0] < outdata.shape[0]:
            pad = np.zeros((outdata.shape[0] - latest.shape[0], latest.shape[1]), dtype=np.float32)
            latest = np.vstack((latest, pad))
        elif latest.shape[0] > outdata.shape[0]:
            latest = latest[:outdata.shape[0], :]

        # Match output channels.
        if latest.shape[1] == 1 and outdata.shape[1] > 1:
            latest = np.repeat(latest, outdata.shape[1], axis=1)
        elif latest.shape[1] != outdata.shape[1]:
            latest = latest[:, :outdata.shape[1]]

        outdata[:] = latest

    def _callback(self, indata, outdata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")
            self._stats["last_input_status"] = str(status)
            self._stats["last_output_status"] = str(status)

        self._stats["input_callbacks"] += 1
        self._stats["output_callbacks"] += 1

        if self._running:
            self._push_input(indata)

        if self._passthrough_enabled:
            self._fill_passthrough(outdata, indata)
            return

        self._fill_output(outdata)

    def _output_callback(self, outdata, frames, time, status):
        if status:
            print(f"Audio output callback status: {status}")
            self._stats["last_output_status"] = str(status)
        self._stats["output_callbacks"] += 1

        if self._passthrough_enabled:
            self._fill_passthrough(outdata, self._latest_passthrough)
            return

        self._fill_output(outdata)

    def _input_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio input callback status: {status}")
            self._stats["last_input_status"] = str(status)

        self._stats["input_callbacks"] += 1

        if not self._running:
            return

        if self._passthrough_enabled:
            self._latest_passthrough = indata.copy()

        self._push_input(indata)

    def get_input(self, block=True, timeout=None):
        return self.input_queue.get(block=block, timeout=timeout)

    def put_output(self, data):
        try:
            self.output_queue.put_nowait(data)
        except queue.Full:
            self._stats["output_queue_drops"] += 1
            try:
                self.output_queue.get_nowait()
                self.output_queue.put_nowait(data)
            except queue.Empty:
                pass

    def get_stats(self):
        return {
            "running": bool(self._running),
            "input_only": bool(self.input_only),
            "samplerate": int(self.samplerate),
            "blocksize": int(self.blocksize),
            "input_channels": int(self.input_channels),
            "output_channels": int(self.output_channels),
            "input_queue_size": int(self.input_queue.qsize()),
            "output_queue_size": int(self.output_queue.qsize()),
            **self._stats,
        }
