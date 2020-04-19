# coding: utf-8
import random
import time as _time
from functools import cached_property
from math import sin, pi
from typing import Callable, Tuple, Dict, List, Set, Union, Iterable, Optional

import numpy
import sounddevice as sd
from numba import jit

from utils.typing import DeviceType
from sounddevice import check_output_settings as output_settings
from sounddevice import check_input_settings as input_settings

sintable = numpy.array(
    [sin(i / 1024 * pi * 2) for i in range(1024)], dtype=numpy.float32,
)


class Device(object):
    """
    Playback device.

    Attributes:
        identifier: identifier for the device, used in callback to differentiate targets.
        device: identifier and name, as returned by sounddevice/PortAudio.
        rate: sample rate. In 44100, 48000, 88200, 96000.
        channels: number of channels, 1 (mono) or 2 (stereo) usually.
        length: playback/recording buffer length in PCM samples.
        duplex: indicates whether the selected device supports recording.

    """

    SAMPLE_RATES = (44100, 48000, 88200, 96000)
    SAMPLE_FORMAT = "float32"
    CHANNEL_COUNTS = (1, 2)

    identifier: int
    channels: int
    rate: int
    length: int
    duplex: bool

    _device: DeviceType
    _stream: Optional[sd._StreamBase] = None

    @property
    def device(self) -> DeviceType:
        return self._device

    @device.setter
    def device(self, identifier: DeviceType):
        """
        Prepare the object for using a specific PortAudio device.

        Args:
            device:
                Device id or name, as exposed by PortAudio.
                If ``None`` or 0, use the default system device.
                If the system supports it, the default system device
                has full duplex support.

        """
        info: Dict = sd.query_devices(identifier)
        if info["max_output_channels"] > 0:
            self.duplex = info["max_input_channels"] > 0
            self.device = identifier or 0
            return
        raise ValueError("The device does not support playback.")

    def can_start(self) -> Tuple[bool, bool]:
        """
        Get whether I/O can be started with current settings.

        Returns:
            A 2-tuple bool, the first telling if recording can
            be made with selected settings, and the second one
            telling whether playback is available with currently
            selected settings.

        """
        try:
            sd.check_input_settings(
                device=self.device,
                channels=self.channels,
                dtype=self.SAMPLE_FORMAT,
                samplerate=self.rate,
            )
            can_record: bool = True
        except sd.PortAudioError:
            can_record: bool = False

        try:
            sd.check_output_settings(
                device=self.device,
                channels=self.channels,
                dtype=self.SAMPLE_FORMAT,
                samplerate=self.rate,
            )
            can_playback: bool = True
        except sd.PortAudioError:
            can_playback: bool = False

        return (can_record, can_playback)

    def play(self):
        """Start playback."""
        if self._stream:
            self._stream.close()
            self._stream = None
        self._stream = sd.OutputStream(
            device=self.device,
            samplerate=self.rate,
            channels=self.channels,
            blocksize=self.length,
            callback=self.play_callback,
        )
        self._stream.start()

    @staticmethod
    def play_callback(device: "Device") -> Callable:
        """
        Callback method for audio.

        Notes:
            Returns a proper callback function for PortAudio.
            Sets frame information that can be used inside the
            callback function.

        """
        frame: int = 0  # playback frame
        channels: int = device.channels

        def playback(out: numpy.array, frames: int, time, status: sd.CallbackFlags):
            """
            Playback real callback.

            Args:
                out: numpy array to fill with audio data.
                frames: number of frames.
                time: structure with time info, see sounddevice docs.
                status: status information, see sounddevice docs.

            """
            nonlocal frame
            # Advance the total played frames
            frame += frames

        return playback


@jit(cache=True, nopython=True, fastmath=True)
def fill(outdata, frames: int, frame):
    for i in range(frames):
        # 440Hz : Cycle samples × 440 repeats ÷ 44100 samples = stride per sample (10.21678)
        val: float = sintable[int((i + frame) * 10.21678) % 1024]
        outdata[i, 0] = val
        outdata[i, 1] = val


def cback(number: int) -> Callable:
    frame = 0
    allowed = 2048 / 48000

    def callback(outdata, frames, time, status: sd.CallbackFlags):
        nonlocal frame
        start = _time.time()
        fill(outdata, frames, frame)
        frame += frames
        elapsed = _time.time() - start
        percent = elapsed / allowed * 100
        print("Slot time used: ", "{percent:.3f}%".format(percent=percent))
        print("Callback ID", number)

    return callback


print(Device().get_devices()[0])
sd.default.device = 10
s = sd.OutputStream(
    channels=2,
    callback=cback(5),
    device="pulse",
    dtype="float32",
    blocksize=2048,
    samplerate=48000,
)
s2 = sd.OutputStream(
    channels=2,
    callback=cback(9),
    device="hdmi",
    dtype="float32",
    blocksize=2048,
    samplerate=48000,
)

s.start()
# s2.start()
sd.sleep(int(1 * 3000))
s.stop()
s.close()
# s2.stop()
# s2.close()
