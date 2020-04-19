# coding: utf-8
from os import PathLike
from typing import Optional, Dict, Any

import audiofile
import numpy
from numpy.core.records import ndarray


class Clip(object):
    """
    An audio clip.

    Attributes:
        rate: sample rate in samples per second.
        channels: number of audio channels, eg. 2 for stereo.
        data: sample PCM data in float.
        duration: sample length in seconds.
        length: sample length in PCM samples.
        metadata: metadata

    """

    rate: int
    channels: int
    data: ndarray
    length: int
    filename: str
    metadata: Dict[str, Any]

    @property
    def duration(self) -> float:
        """
        Duration of the sample in seconds at default rate.

        Returns:
            duration in seconds.

        """
        return self.length / self.rate

    def load(self, path: PathLike, duration: float = None):
        """
        Load an audio file.

        Args:
            path:
                Audio file local path.
                Can be any supported media containing an audio track.
            duration:
                Maximum duration to load in seconds.
                Any track longer than the provided duration will be truncated.

        """
        self.data, self.rate = audiofile.read(path, duration=duration)
        self.channels = self.data.shape[0]
        self.length = self.data.shape[1]
        self.filename = str(path)

    def new(self, channels: int, length: int):
        """
        Create an empty audio segment with n channels.

        Args:
            channels: number of channels to create.
            length: number of PCM samples.

        """
        self.channels = channels
        self.length = length
        self.data = numpy.zeros((channels, length), dtype=numpy.float64)
        self.rate = 44100

    def save(self, path: PathLike):
        """
        Save the audio file to disk.

        Args:
            path:
                Audio output file. WAV, OGG and FLAC extensions only.

        """
        audiofile.write(path, self.data, self.rate)

    def get_channel(self, index: int = 0) -> ndarray:
        """
        Get one audio channel.

        Args:
            index: index of the audio channel, zero-based.

        Returns:
            A subset of the audio data.

        """
        return self.data[index]
