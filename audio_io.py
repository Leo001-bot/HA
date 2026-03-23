import queue

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
        # Keep short queues to avoid long buffering delay that sounds like echo.
        self.input_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=2)
        self.stream = None
        self._running = False

    def _resolve_channels(self):
        """Choose channel counts supported by current default input/output devices."""
        in_caps = sd.query_devices(kind='input')
        out_caps = sd.query_devices(kind='output')

        in_max = int(max(1, in_caps.get('max_input_channels', 1)))
        out_max = int(max(1, out_caps.get('max_output_channels', 1)))

        self.input_channels = int(max(1, min(self.input_channels, in_max)))
        self.output_channels = int(max(1, min(self.output_channels, out_max)))

    def start(self):
        self._running = True
        self._resolve_channels()
        self.stream = sd.Stream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            channels=(self.input_channels, self.output_channels),
            dtype=np.float32,
            callback=self._callback,
            latency='low',
        )
        self.stream.start()

    def stop(self):
        self._running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _callback(self, indata, outdata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")

        if self._running:
            try:
                self.input_queue.put_nowait(indata.copy())
            except queue.Full:
                # Drop oldest chunk to keep latency bounded.
                try:
                    self.input_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.input_queue.put_nowait(indata.copy())
                except queue.Full:
                    pass

        # Play the newest available output frame and drop stale frames.
        latest = None
        while True:
            try:
                latest = self.output_queue.get_nowait()
            except queue.Empty:
                break

        if latest is None:
            outdata.fill(0)
        else:
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

    def get_input(self, block=True, timeout=None):
        return self.input_queue.get(block=block, timeout=timeout)

    def put_output(self, data):
        try:
            self.output_queue.put_nowait(data)
        except queue.Full:
            try:
                self.output_queue.get_nowait()
                self.output_queue.put_nowait(data)
            except queue.Empty:
                pass
