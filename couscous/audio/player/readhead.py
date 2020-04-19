# coding: utf-8

import numpy
from numpy.core.records import ndarray

from audio.clip import Clip
from audio.io.mixer import Mixer


class ReadHead(object):
    """
    Play head for a channel of an audio clip.

    Instances of this class are used to fill device audio buffers
    with audio clip data.

    Attributes:
        frame: current frame in transport playback
        clip: audio sample to read from.
        channel: sample channel to read from.
        position: current reading position in PCM samples.
        speed: head speed in multiples of the nominal sample frequency.
        active: head is stopped

    """

    frame: int
    clip: Clip
    channel: int
    position: float
    speed: float
    active: bool

    def __init__(
        self,
        clip: Clip,
        channel: int = 0,
        position: float = 0,
        speed: float = 1.0,
        active: bool = True,
    ):
        """
        Initialize the new play head.

        Args:
            clip: sample instance.
            channel: multichannel index in the sample.
            position: sample position in the channel.
            speed: 1.0 means normal speed, 2.0 means double speed.
            active: if `False`, the play head will not move.

        """
        self.frame = 0
        self.clip = clip
        self.channel = channel % clip.channels
        self.position = position % clip.length
        self.speed = speed
        self.active = active

    def seek(self, position: float) -> float:
        """
        Move the play head in the sample.

        Args:
            position: PCM sample index. Can be float.

        Returns:
            The new position. Can be different from the passed position,
            in the case where the passed position is out of bounds.

        """
        self.position = position % self.clip.length
        return self.position

    def read(self, mixer: Mixer) -> ndarray:
        """
        Fill an audio chunk from reading the sample using current properties.

        The play head properties are updated in-place.

        Args:
            mixer: target mixer, which has a buffer size and frequency.

        Returns:
            A one-dimensional array with data for the channel.

        """
        output: ndarray = numpy.zeros(mixer.size, dtype=numpy.float64)
        # The head speed is also determined by the output frequency.
        step: float = self.speed * self.clip.rate / mixer.rate
        # Get a 1D array of the sample data for the channel
        data: ndarray = self.clip.get_channel(self.channel)
        # Fill the output buffer, with no interpolation.
        for i in range(mixer.size):
            if self.active:
                output[i] = data[int(self.position)]
                self.position += step
            self.frame += 1
        return output
