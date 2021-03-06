from multiprocessing import Queue
from scipy.signal import firwin, lfilter

from realtime.util.audioio import Microphone


class RecorderWorker:
    def __init__(self,
                 microphone: Microphone,
                 cutoff: int,
                 fs: int,
                 recorded_queue: Queue
                 ):
        '''
        マイク設定に従いマイクに接続する。
        Parameters
        ----------
        microphone
        cutoff
            カットオフ周波数
        fs
            サンプリング周波数
        recorded_queue
        '''
        self._microphone: Microphone = microphone
        self._lowcutfilter = firwin(255, cutoff / (fs // 2), pass_zero=False)
        self._queue: Queue = recorded_queue

    def start(self):
        '''
        入力デバイスから受け取った音声に対してローカットフィルターを適用し、
        キューへ入れるループを行う。
        '''
        while True:
            wav_data = self._microphone.read_frame()
            wav_data = lfilter(self._lowcutfilter, 1, wav_data)
            self._queue.put(wav_data)
