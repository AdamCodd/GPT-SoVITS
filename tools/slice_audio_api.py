import numpy as np
import librosa
from scipy.io import wavfile
import os
from tqdm import tqdm

class Slicer:
    """
    A class for slicing audio based on silence detection with integrated audio processing.
    """
    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 5000,
        min_interval: int = 300,
        hop_size: int = 20,
        max_sil_kept: int = 5000,
    ):
        """
        Initialize the Slicer.
        
        Args:
            sr: Sample rate of the audio
            threshold: The threshold (in dB) below which is considered silence
            min_length: Minimum length (in milliseconds) required for each slice
            min_interval: Minimum length (in milliseconds) of silence between slices
            hop_size: Length of hop between frames (in milliseconds)
            max_sil_kept: Maximum length of silence kept around the slices (in milliseconds)
        """
        if not min_length >= min_interval >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: min_length >= min_interval >= hop_size"
            )
        if not max_sil_kept >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: max_sil_kept >= hop_size"
            )
        
        self.sr = sr
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        """Apply the slice to the waveform."""
        if len(waveform.shape) > 1:
            return waveform[
                :, begin * self.hop_size : min(waveform.shape[1], end * self.hop_size)
            ]
        else:
            return waveform[
                begin * self.hop_size : min(waveform.shape[0], end * self.hop_size)
            ]

    def slice(self, waveform):
        """
        Slice the waveform based on silence detection.
        
        Args:
            waveform: The audio waveform to slice (can be mono or stereo)
            
        Returns:
            Generator yielding tuples of (processed_chunk, start_frame, end_frame)
        """
        # Handle mono/stereo input
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
            
        # Return early if audio is too short
        if samples.shape[0] <= self.min_length:
            yield waveform, 0, samples.shape[0]
            return

        # Calculate RMS using librosa
        rms_list = librosa.feature.rms(
            y=samples,
            frame_length=self.win_size,
            hop_length=self.hop_size,
            center=True
        ).squeeze(0)

        # Initialize variables for slicing
        sil_tags = []
        silence_start = None
        clip_start = 0
        
        # Iterate through RMS values to find silence boundaries
        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue
                
            if silence_start is None:
                continue
                
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (
                i - silence_start >= self.min_interval
                and i - clip_start >= self.min_length
            )
            
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue

            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start : i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
                
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[
                    i - self.max_sil_kept : silence_start + self.max_sil_kept + 1
                ].argmin()
                pos += i - self.max_sil_kept
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
                    
            else:
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
                
            silence_start = None

        # Handle trailing silence
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start : silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))

        # Generate and yield chunks
        if len(sil_tags) == 0:
            yield waveform, 0, int(total_frames * self.hop_size)
            return
        
        if sil_tags[0][0] > 0:
            chunk = self._apply_slice(waveform, 0, sil_tags[0][0])
            yield chunk, 0, int(sil_tags[0][0] * self.hop_size)
            
        for i in range(len(sil_tags) - 1):
            chunk = self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0])
            yield (
                chunk,
                int(sil_tags[i][1] * self.hop_size),
                int(sil_tags[i + 1][0] * self.hop_size)
            )
            
        if sil_tags[-1][1] < total_frames:
            chunk = self._apply_slice(waveform, sil_tags[-1][1], total_frames)
            yield (
                chunk,
                int(sil_tags[-1][1] * self.hop_size),
                int(total_frames * self.hop_size)
            )

def process_and_save_chunks(
    input_path,
    output_dir,
    sr=32000,
    threshold=-40,
    min_length=5000,
    min_interval=300,
    hop_size=20,
    max_sil_kept=5000,
    max_amp=1.0,
    alpha=1.0
):
    """
    Process audio file(s) and save chunks with amplitude normalization.
    Includes progress bars for both file processing and chunk saving.
    
    Args:
        input_path: Path to audio file or directory
        output_dir: Directory to save processed chunks
        sr: Sample rate
        threshold: Silence threshold in dB
        min_length: Minimum length in ms
        min_interval: Minimum silence interval in ms
        hop_size: Hop size in ms
        max_sil_kept: Maximum silence kept in ms
        max_amp: Maximum amplitude for normalization
        alpha: Blending factor for normalization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.isfile(input_path):
        input_files = [input_path]
    elif os.path.isdir(input_path):
        input_files = [os.path.join(input_path, name) 
                      for name in sorted(os.listdir(input_path))]
    else:
        raise ValueError("Input path must be a file or directory")

    slicer = Slicer(
        sr=sr,
        threshold=threshold,
        min_length=min_length,
        min_interval=min_interval,
        hop_size=hop_size,
        max_sil_kept=max_sil_kept,
    )

    # Main progress bar for files
    with tqdm(total=len(input_files), desc="Processing files", unit="file") as pbar_files:
        for input_file in input_files:
            try:
                name = os.path.basename(input_file)
                
                # Load audio with progress message
                pbar_files.set_postfix_str(f"Loading {name}")
                audio = librosa.load(input_file, sr=sr)[0]
                
                # Pre-compute slices to get total count for progress bar
                slices = list(slicer.slice(audio))
                
                # Nested progress bar for chunks within current file
                with tqdm(
                    total=len(slices),
                    desc=f"Saving chunks for {name}",
                    unit="chunk",
                    leave=False
                ) as pbar_chunks:
                    for chunk, start, end in slices:
                        # Normalize audio chunk
                        tmp_max = np.abs(chunk).max()
                        if tmp_max > 1:
                            chunk = chunk / tmp_max
                        chunk = (chunk / tmp_max * (max_amp * alpha)) + (1 - alpha) * chunk
                        
                        # Save as WAV
                        output_path = os.path.join(
                            output_dir,
                            f"{name}_{start:010d}_{end:010d}.wav"
                        )
                        wavfile.write(
                            output_path,
                            sr,
                            (chunk * 32767).astype(np.int16)
                        )
                        
                        pbar_chunks.update(1)
                
                pbar_files.update(1)
                
            except Exception as e:
                pbar_files.set_postfix_str(f"Error: {str(e)}")
                print(f"\nFailed to process {input_file}: {str(e)}")
                continue
