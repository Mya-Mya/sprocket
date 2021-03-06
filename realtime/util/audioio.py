import logging
import atexit
import pyaudio
import numpy
from scipy.signal import firwin, lfilter

_audio = pyaudio.PyAudio()


def get_available_input_devices()->dict:
    return {i: _audio.get_device_info_by_index(i)['name'] for i in range(_audio.get_device_count())}

def get_available_output_devices()->dict:
    return {i: _audio.get_device_info_by_index(i)['name'] for i in range(_audio.get_device_count())}


class Microphone:
    def __init__(self,
                 frames_per_buffer: int,
                 frame_rate: int,
                 input_device_index: int,
                 cutoff:int = 70
                 ):
        self._stream = _audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=frame_rate,
            input_device_index=input_device_index,
            input=True
        )
        self._frame_per_buffer = frames_per_buffer
        self._lowcutfilter = firwin(255, cutoff / (frame_rate // 2), pass_zero=False)
        atexit.register(self.close_stream)

    def read_frame(self)->numpy.ndarray:
        '''
        Returns
        -------
        1フレーム内の音声情報
        '''
        wav_bytes = self._stream.read(self._frame_per_buffer)
        wav_array = numpy.frombuffer(wav_bytes,dtype=numpy.core.int16)
        wav_array = lfilter(self._lowcutfilter, 1, wav_array)
        return wav_array

    def close_stream(self):
        self._stream.close()

class Speaker:
    def __init__(self,
                 frames_per_buffer: int,
                 frame_rate: int,
                 output_device_index: int
                 ):
        self._stream = _audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=frame_rate,
            frames_per_buffer=frames_per_buffer,
            output_device_index=output_device_index,
            output=True
        )
        atexit.register(self.close_stream)

    def play_frame(self,wav_data:numpy.ndarray):
        self._stream.write(wav_data.tobytes())

    def close_stream(self):
        self._stream.close()