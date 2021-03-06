from multiprocessing import Queue

from realtime.util.audioio import Speaker

class PlayerWorker:
    def __init__(self,
                 speaker:Speaker,
                 converted_queue:Queue
                 ):
        self._speaker = speaker
        self._converted_queue = converted_queue
    def start(self):
        while True:
            converted_wav = self._converted_queue.get()
            self._speaker.play_frame(converted_wav)