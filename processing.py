import numpy as np


class NoiseReducer:
    """Single-channel spectral noise reducer."""

    def __init__(self, sample_rate=16000, frame_size=256, hop_size=128):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.window = np.hanning(frame_size)
        self.noise_psd = None
        self.ph1_mean = None
        self.prev_gain = None
        self.prev_post_snr = None
        self.frame_count = 0
        self.min_gain = 0.06
        self.gain_smoothing = 0.82

        # openMHA-like SPP/noise-PSD parameters (adapted for low-latency frame-wise use)
        self.alpha_ph1_mean = 0.90
        self.alpha_psd = 0.84
        self.prior_q = 0.50
        self.xi_opt_db = 15.0
        xi_opt = 10.0 ** (self.xi_opt_db / 10.0)
        self.prior_fact = self.prior_q / max(1e-6, (1.0 - self.prior_q))
        self.log_glr_fact = np.log(1.0 / (1.0 + xi_opt))
        self.glr_exp = xi_opt / (1.0 + xi_opt)
        self.dd_alpha = 0.92
        self.wind_ratio_lp = 0.0

    def _cepstral_smooth_gain(self, gain, strength):
        # Real-cepstrum smoothing to reduce musical noise while keeping spectral envelope.
        mag = np.maximum(gain.astype(np.float64, copy=False), 1e-6)
        log_mag = np.log(mag)
        ceps = np.fft.irfft(log_mag)
        q_keep = int(np.clip(6 + 6 * strength, 4, max(4, ceps.size // 2 - 1)))
        if q_keep < ceps.size:
            ceps[q_keep:-q_keep] = 0.0
        smooth_log_mag = np.fft.rfft(ceps, n=(mag.size - 1) * 2).real
        smooth_gain = np.exp(smooth_log_mag)
        return np.clip(smooth_gain.astype(np.float32), 0.0, 1.0)

    def _apply_wind_mode(self, gain, power, wind_mode, wind_sensitivity, strength):
        if not wind_mode:
            return gain

        n_bins = gain.size
        if n_bins < 8:
            return gain

        lf_bins = max(3, int(0.14 * n_bins))
        low_e = float(np.mean(power[:lf_bins])) + 1e-10
        full_e = float(np.mean(power)) + 1e-10
        ratio = low_e / full_e

        # Low-pass ratio detector similar to openMHA wind-noise logic.
        det_alpha = np.clip(0.90 + 0.06 * (1.0 - wind_sensitivity), 0.82, 0.98)
        self.wind_ratio_lp = det_alpha * self.wind_ratio_lp + (1.0 - det_alpha) * ratio
        threshold = np.clip(1.18 - 0.40 * wind_sensitivity, 0.75, 1.25)
        if self.wind_ratio_lp <= threshold:
            return gain

        sev = np.clip((self.wind_ratio_lp - threshold) / max(0.2, threshold), 0.0, 2.0)
        sev *= np.clip(0.7 + 0.4 * strength, 0.7, 2.0)
        max_att_db = np.clip(-6.0 - 10.0 * wind_sensitivity, -18.0, -5.0)
        lf_gain = 10.0 ** ((max_att_db * min(1.0, sev)) / 20.0)

        out = gain.copy()
        out[:lf_bins] *= lf_gain
        if lf_bins < n_bins - 1:
            taper_bins = min(lf_bins, n_bins - lf_bins)
            if taper_bins > 0:
                taper = np.linspace(lf_gain, 1.0, taper_bins, endpoint=True)
                out[lf_bins:lf_bins + taper_bins] *= taper
        return np.clip(out, 0.0, 1.0)

    def process(
        self,
        audio,
        strength=1.0,
        cepstral_smoothing=True,
        attack_release_split=True,
        wind_mode=False,
        wind_sensitivity=0.6,
        band_low_hz=80.0,
        band_high_hz=7000.0,
    ):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return audio

        strength = float(np.clip(strength, 0.3, 3.0))
        wind_sensitivity = float(np.clip(wind_sensitivity, 0.0, 1.0))
        band_low_hz = float(np.clip(band_low_hz, 0.0, self.sample_rate * 0.49))
        band_high_hz = float(np.clip(band_high_hz, band_low_hz + 50.0, self.sample_rate * 0.5))

        fft = np.fft.rfft(audio)
        power = np.abs(fft) ** 2
        eps = 1e-10

        if self.noise_psd is None or self.noise_psd.shape != power.shape:
            self.noise_psd = power.copy()
            self.ph1_mean = np.full_like(power, 0.5)
            self.prev_gain = np.ones_like(power)
            self.prev_post_snr = np.ones_like(power)
            self.frame_count = 0

        # Bootstrap noise PSD using first frames like openMHA's estimator behavior.
        if self.frame_count < 5:
            self.noise_psd = (self.noise_psd * self.frame_count + power) / (self.frame_count + 1)
            self.frame_count += 1

        post_snr = power / (self.noise_psd + eps)
        log_glr = self.log_glr_fact + self.glr_exp * post_snr
        glr = self.prior_fact * np.exp(np.minimum(log_glr, 200.0))
        ph1 = glr / (1.0 + glr)

        self.ph1_mean = self.alpha_ph1_mean * self.ph1_mean + (1.0 - self.alpha_ph1_mean) * ph1
        ph1 = np.where(self.ph1_mean > 0.99, np.minimum(ph1, 0.99), ph1)

        estimate = ph1 * self.noise_psd + (1.0 - ph1) * power
        if attack_release_split:
            speech_mask = ph1 >= 0.5
            alpha_release = np.clip(0.90 + 0.04 * strength, 0.90, 0.98)
            alpha_attack = np.clip(0.70 + 0.07 * (2.0 - strength), 0.65, 0.88)
            alpha_vec = np.where(speech_mask, alpha_release, alpha_attack)
            self.noise_psd = alpha_vec * self.noise_psd + (1.0 - alpha_vec) * estimate
        else:
            alpha_psd = np.clip(self.alpha_psd + 0.05 * (strength - 1.0), 0.70, 0.96)
            self.noise_psd = alpha_psd * self.noise_psd + (1.0 - alpha_psd) * estimate

        # Decision-directed a-priori SNR with Wiener gain floor.
        ml_snr = np.maximum(post_snr - 1.0, 0.0)
        priori_snr = self.dd_alpha * (self.prev_gain ** 2) * self.prev_post_snr + (1.0 - self.dd_alpha) * ml_snr
        priori_snr = np.maximum(priori_snr, 10.0 ** (-27.0 / 10.0))

        min_gain = np.clip(self.min_gain - 0.025 * (strength - 1.0), 0.02, 0.20)
        gain = priori_snr / (1.0 + priori_snr)
        gain = gain ** (1.10 + 0.40 * (strength - 1.0))
        gain = np.clip(gain, min_gain, 1.0)
        if cepstral_smoothing:
            gain = self._cepstral_smooth_gain(gain, strength)
        gain = self._apply_wind_mode(gain, power, wind_mode, wind_sensitivity, strength)

        # Apply NR only inside selected bandwidth; outside the band remains untouched.
        freqs = np.fft.rfftfreq(audio.size, d=1.0 / self.sample_rate)
        band_mask = (freqs >= band_low_hz) & (freqs <= band_high_hz)
        if np.any(~band_mask):
            gain = np.where(band_mask, gain, 1.0)

        # Light smoothing across neighboring bins to reduce musical noise.
        gain = (np.roll(gain, 1) + 2.0 * gain + np.roll(gain, -1)) / 4.0
        smoothing = np.clip(self.gain_smoothing + 0.04 * (strength - 1.0), 0.70, 0.93)
        gain = smoothing * self.prev_gain + (1.0 - smoothing) * gain
        self.prev_gain = gain
        self.prev_post_snr = post_snr

        fft_filtered = fft * gain
        return np.fft.irfft(fft_filtered, n=len(audio)).astype(np.float32, copy=False)


class Compressor:
    """Simple multiband dynamic range compressor."""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.bands = np.array([250, 500, 1000, 2000, 4000, 8000])
        self.thresholds = np.array([30, 30, 30, 30, 30, 30])
        self.ratios = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def set_audiogram(self, frequencies, thresholds_db):
        for i, band in enumerate(self.bands):
            idx = np.argmin(np.abs(frequencies - band))
            thresh = thresholds_db[idx]
            if thresh > 60:
                self.ratios[i] = 3.0
                self.thresholds[i] = 70
            elif thresh > 30:
                self.ratios[i] = 2.0
                self.thresholds[i] = 50
            else:
                self.ratios[i] = 1.0
                self.thresholds[i] = 30

    def process(self, audio):
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), d=1 / self.sample_rate)

        gain_per_band = np.ones(len(self.bands))
        for i, band_freq in enumerate(self.bands):
            idx = np.abs(freqs - band_freq).argmin()
            level_db = 20 * np.log10(np.abs(fft[idx]) + 1e-10)
            if level_db > self.thresholds[i]:
                excess = level_db - self.thresholds[i]
                compressed_excess = excess / self.ratios[i]
                target_level_db = self.thresholds[i] + compressed_excess
                gain_db = target_level_db - level_db
                gain_per_band[i] = 10 ** (gain_db / 20)

        gain_curve = np.interp(freqs, self.bands, gain_per_band, left=1, right=1)
        fft_compressed = fft * gain_curve
        return np.fft.irfft(fft_compressed, n=len(audio))


class FeedbackCanceller:
    """NLMS adaptive filter for feedback cancellation."""

    def __init__(self, filter_length=128, mu=0.01, sample_rate=16000):
        self.filter_length = filter_length
        self.mu = mu
        self.sample_rate = sample_rate
        self.weights = np.zeros(filter_length, dtype=np.float32)
        self.buffer = np.zeros(filter_length, dtype=np.float32)
        self.leak = 0.9997

    def _align_reference(self, mic_block, speaker_block):
        n = min(mic_block.size, speaker_block.size)
        if n <= 8:
            return speaker_block[:n]

        mic = mic_block[:n]
        spk = speaker_block[:n]
        mic_norm = float(np.linalg.norm(mic)) + 1e-8

        best_ref = spk
        best_score = -1.0
        max_delay = min(48, n // 2)

        for d in range(0, max_delay + 1, 2):
            if d == 0:
                cand = spk
            else:
                cand = np.empty_like(spk)
                cand[:d] = 0.0
                cand[d:] = spk[:-d]

            cand_norm = float(np.linalg.norm(cand)) + 1e-8
            score = abs(float(np.dot(mic, cand))) / (mic_norm * cand_norm)
            if score > best_score:
                best_score = score
                best_ref = cand

        return best_ref

    def process(self, input_signal, speaker_signal):
        self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = speaker_signal

        feedback_estimate = np.dot(self.weights, self.buffer)
        error = input_signal - feedback_estimate

        power = np.dot(self.buffer, self.buffer) + 1e-8
        # Freeze adaptation when there is almost no speaker reference.
        if abs(speaker_signal) > 1e-6:
            step = self.mu / (1.0 + 0.2 * abs(error))
            self.weights = self.leak * self.weights + step * error * self.buffer / power
        return error

    def process_block(self, input_block, speaker_block, strength=1.0, near_end_mode=False, near_end_strength=0.6):
        input_block = np.asarray(input_block, dtype=np.float32)
        speaker_block = np.asarray(speaker_block, dtype=np.float32)
        n = min(input_block.size, speaker_block.size)
        if n == 0:
            return input_block

        strength = float(np.clip(strength, 0.2, 3.0))
        near_end_strength = float(np.clip(near_end_strength, 0.0, 1.0))

        aligned_ref = self._align_reference(input_block, speaker_block)

        # First remove the coherent part via block projection to suppress strong leakage quickly.
        denom = float(np.dot(aligned_ref, aligned_ref)) + 1e-8
        proj = float(np.dot(input_block[:n], aligned_ref)) / denom
        proj = float(np.clip(proj * (0.75 + 0.45 * strength), 0.0, 1.8))
        mic_pre = input_block.copy()
        mic_pre[:n] = mic_pre[:n] - proj * aligned_ref

        old_mu = self.mu
        self.mu = float(np.clip(old_mu * (0.7 + 0.5 * strength), 0.002, 0.08))

        out = np.empty_like(input_block)
        for i in range(n):
            out[i] = self.process(float(mic_pre[i]), float(aligned_ref[i]))

        self.mu = old_mu

        # Residual echo suppression driven by short-term coherence.
        mic = out[:n]
        spk = aligned_ref
        mic_norm = float(np.linalg.norm(mic)) + 1e-8
        spk_norm = float(np.linalg.norm(spk)) + 1e-8
        coherence = abs(float(np.dot(mic, spk))) / (mic_norm * spk_norm)
        spk_rms = float(np.sqrt(np.mean(spk * spk)) + 1e-8)

        if coherence > 0.35 and spk_rms > 0.004:
            # Attenuate only when signal resembles speaker leakage.
            att = np.clip((coherence - 0.35) * (1.0 + 0.6 * strength), 0.0, 0.85)
            out[:n] *= (1.0 - att)

        # Optional near-end suppression mode (sidetone reduction for earphone tests).
        if near_end_mode and near_end_strength > 0.0:
            near_rms = float(np.sqrt(np.mean(out[:n] * out[:n])) + 1e-8)
            if near_rms > 0.01:
                spec = np.abs(np.fft.rfft(out[:n]))
                freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
                speech_band = (freqs >= 250.0) & (freqs <= 3500.0)
                if np.any(speech_band):
                    speech_ratio = float(np.sum(spec[speech_band])) / (float(np.sum(spec)) + 1e-8)
                else:
                    speech_ratio = 0.0

                # Trigger when dominant near-end speech is detected and not strongly echo-coherent.
                if speech_ratio > 0.45 and coherence < 0.55:
                    near_att = np.clip((speech_ratio - 0.40) * (0.6 + 0.9 * near_end_strength), 0.0, 0.80)
                    out[:n] *= (1.0 - near_att)

        if n < input_block.size:
            out[n:] = input_block[n:]
        return out
