# Multiple quality checks (duration, silence, RMS, clipping, SNR, spectral flatness) for the audio file
from pydantic import BaseModel
import numpy as np
import librosa
from typing import Dict, List, Optional, Tuple

class AudioQualityParams(BaseModel):
    min_duration: Optional[float] = 3.0
    max_duration: Optional[float] = 10.0
    min_rms_db: Optional[float] = -30
    max_rms_db: Optional[float] = -10
    min_snr_db: Optional[float] = 15
    max_clipping_ratio: Optional[float] = 0.01
    min_spectral_flatness: Optional[float] = 0.05
    max_spectral_flatness: Optional[float] = 0.5
    sample_rate: Optional[int] = 22050

class QualityCheckConfig(AudioQualityParams):
    checks: Optional[List[str]] = None

class AudioQualityChecker:
    def __init__(self, params: Optional[AudioQualityParams] = None):
        if params is None:
            params = AudioQualityParams()
        
        self.min_duration = params.min_duration
        self.max_duration = params.max_duration
        self.min_rms_db = params.min_rms_db
        self.max_rms_db = params.max_rms_db
        self.min_snr_db = params.min_snr_db
        self.max_clipping_ratio = params.max_clipping_ratio
        self.min_spectral_flatness = params.min_spectral_flatness
        self.max_spectral_flatness = params.max_spectral_flatness
        self.sample_rate = params.sample_rate
        self.duration = None  # Will be set during check_quality
        self.current_sample_rate = None  # Will be set during check_quality

    def check_duration(self, audio: np.ndarray) -> Tuple[bool, float]:
        duration = len(audio) / self.sample_rate
        self.duration = duration
        self.current_sample_rate = self.sample_rate
        return self.min_duration <= duration <= self.max_duration, duration

    def check_rms(self, audio: np.ndarray) -> Tuple[bool, float]:
        rms_value = np.sqrt(np.mean(np.square(audio)))
        rms_db = 20 * np.log10(rms_value)
        return self.min_rms_db <= rms_db <= self.max_rms_db, rms_db

    def check_clipping(self, audio: np.ndarray) -> Tuple[bool, float]:
        clipping_mask = np.abs(audio) > 0.99
        clipping_ratio = np.mean(clipping_mask)
        return clipping_ratio <= self.max_clipping_ratio, clipping_ratio

    def estimate_noise_floor(self, audio: np.ndarray) -> float:
        frame_length = 2048
        hop_length = 512
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        frame_rms = np.sqrt(np.mean(frames**2, axis=0))
        noise_threshold = np.percentile(frame_rms, 5)
        return noise_threshold**2

    def check_snr(self, audio: np.ndarray) -> Tuple[bool, float]:
        noise_floor = self.estimate_noise_floor(audio)
        signal_power = np.mean(np.square(audio))
        snr = 10 * np.log10(signal_power / noise_floor) if noise_floor > 0 else float('inf')
        return snr >= self.min_snr_db, snr

    def check_spectral_flatness(self, audio: np.ndarray) -> Tuple[bool, float]:
        spec = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        geometric_mean = np.exp(np.mean(np.log(spec + 1e-10), axis=0))
        arithmetic_mean = np.mean(spec, axis=0)
        flatness = np.mean(geometric_mean / (arithmetic_mean + 1e-10))
        return self.min_spectral_flatness <= flatness <= self.max_spectral_flatness, flatness

    def check_quality(self, audio: np.ndarray, bypass_quality_checks: bool = False) -> Tuple[bool, Dict[str, float]]:
        metrics = {}

        # Check duration
        duration_ok, metrics['duration'] = self.check_duration(audio)
        if not duration_ok:
            return False, metrics

        if bypass_quality_checks:
            return True, metrics

        # Quick silence check
        if np.max(np.abs(audio)) < 0.01:
            return False, metrics

        # Check RMS
        rms_ok, metrics['rms_db'] = self.check_rms(audio)
        if not rms_ok:
            return False, metrics

        # Check clipping
        clipping_ok, metrics['clipping_ratio'] = self.check_clipping(audio)
        if not clipping_ok:
            return False, metrics

        # Check SNR
        snr_ok, metrics['snr_db'] = self.check_snr(audio)
        if not snr_ok:
            return False, metrics

        # Check spectral flatness
        flatness_ok, metrics['spectral_flatness'] = self.check_spectral_flatness(audio)
        if not flatness_ok:
            return False, metrics

        return True, metrics

    def get_metric_analysis(self, metrics: Dict[str, float]) -> List[Dict[str, str]]:
        analysis = []
        
        analysis_methods = {
            'duration': self._analyze_duration,
            'rms_db': self._analyze_rms,
            'snr_db': self._analyze_snr,
            'clipping_ratio': self._analyze_clipping,
            'spectral_flatness': self._analyze_spectral_flatness
        }
        
        for metric, analyze_func in analysis_methods.items():
            if metric in metrics:
                analysis.append(analyze_func(metrics[metric]))
                
        return analysis

    def _analyze_duration(self, value: float) -> Dict[str, str]:
        if value < self.min_duration:
            status = "❌ Too Short"
            details = (
                "- Recording too short to capture voice characteristics\n"
                "- Need longer sample for proper voice modeling\n"
                "- Recommend at least 3 seconds of clear speech"
            )
        elif value > self.max_duration:
            status = "❌ Too Long"
            details = (
                "- Recording length exceeds recommended maximum\n"
                "- May introduce inconsistencies in voice modeling\n"
                "- Consider breaking into shorter segments"
            )
        else:
            status = "✅ Ideal"
            if value < 5.0:
                details = "- Acceptable but shorter length\n- Contains minimum required voice characteristics\n- Consider slightly longer recording if possible"
            elif value > 8.0:
                details = "- Acceptable but longer length\n- Good for capturing voice variations\n- Ensure consistent voice quality throughout"
            else:
                details = "- Perfect length for voice modeling\n- Optimal balance of duration and content\n- Excellent for capturing voice characteristics"

        return {
            "metric": "Duration",
            "status": status,
            "value": f"{value:.2f}s",
            "range": f"{self.min_duration}-{self.max_duration}s",
            "details": details
        }

    def _analyze_rms(self, value: float) -> Dict[str, str]:
        if value < self.min_rms_db:
            status = "❌ Too Quiet"
            details = "- Audio level too low for effective processing\n- May result in poor signal-to-noise ratio\n- Recommend re-recording with higher input gain"
        elif value > self.max_rms_db:
            status = "❌ Too Loud"
            details = "- Audio level too high, risking distortion\n- May contain clipping or compression artifacts\n- Recommend re-recording with lower input gain"
        else:
            status = "✅ Ideal"
            if value < -25:
                details = "- On the quieter side but acceptable\n- Could benefit from slightly higher gain\n- Ensure consistent voice projection"
            elif value > -15:
                details = "- On the louder side but acceptable\n- Good voice presence\n- Monitor for potential peak clipping"
            else:
                details = "- Perfect volume level for processing\n- Excellent dynamic range\n- Optimal for voice clarity"

        return {
            "metric": "RMS Level",
            "status": status,
            "value": f"{value:.1f}dB",
            "range": f"{self.min_rms_db} to {self.max_rms_db}dB",
            "details": details
        }

    def _analyze_snr(self, value: float) -> Dict[str, str]:
        if value < self.min_snr_db:
            status = "❌ Too Noisy"
            details = "- Excessive background noise detected\n- Voice quality compromised by noise\n- Record in quieter environment"
        else:
            status = "✅ Ideal"
            if value > 70:
                details = "- Studio-quality signal-to-noise ratio\n- Exceptional recording environment\n- Perfect for voice processing"
            elif value > 50:
                details = "- Professional-grade quality\n- Very clean recording\n- Excellent for voice processing"
            elif value > 30:
                details = "- Good signal-to-noise ratio\n- Clean recording with minimal noise\n- Well-suited for voice processing"
            else:
                details = "- Acceptable signal-to-noise ratio\n- Some background noise present\n- Consider quieter recording environment"

        return {
            "metric": "Signal-to-Noise Ratio",
            "status": status,
            "value": f"{value:.1f}dB",
            "range": f"Minimum {self.min_snr_db}dB",
            "details": details
        }

    def _analyze_clipping(self, value: float) -> Dict[str, str]:
        if value > self.max_clipping_ratio:
            status = "❌ Excessive Clipping"
            details = "- Significant audio clipping detected\n- Voice quality compromised by distortion\n- Re-record with lower input gain"
        else:
            status = "✅ Ideal"
            if value == 0:
                details = "- No clipping detected\n- Perfect peak control\n- Optimal dynamic range preserved"
            elif value < 0.001:
                details = "- Minimal clipping detected\n- Excellent peak control\n- Very good dynamic range"
            else:
                details = "- Acceptable clipping levels\n- Good peak control\n- Consider slightly lower input gain"

        return {
            "metric": "Clipping",
            "status": status,
            "value": f"{value*100:.2f}%",
            "range": f"Maximum {self.max_clipping_ratio*100}%",
            "details": details
        }

    def _analyze_spectral_flatness(self, value: float) -> Dict[str, str]:
        if value < self.min_spectral_flatness:
            status = "❌ Too Tonal"
            details = "- Audio too tonal/harmonic\n- May indicate singing or musical content\n- Need more natural speech characteristics"
        elif value > self.max_spectral_flatness:
            status = "❌ Too Noisy"
            details = "- Spectrum too noise-like\n- May indicate white noise or fricatives\n- Need more tonal voice content"
        else:
            status = "✅ Ideal"
            if value < 0.15:
                details = "- More tonal characteristics\n- Good voice resonance\n- Clear harmonic structure"
            elif value > 0.35:
                details = "- More balanced spectrum\n- Good mix of harmonics and fricatives\n- Natural speech characteristics"
            else:
                details = "- Perfect spectral balance\n- Optimal mix of voice characteristics\n- Excellent for voice modeling"

        return {
            "metric": "Spectral Flatness",
            "status": status,
            "value": f"{value:.3f}",
            "range": f"{self.min_spectral_flatness}-{self.max_spectral_flatness}",
            "details": details
        }
