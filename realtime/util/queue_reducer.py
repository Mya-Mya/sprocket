from multiprocessing import Queue


class QueueReducer:
    def __init__(self, max_queue_size: int, queue: Queue):
        self._max_queue_size = max_queue_size
        self._queue = queue

    def check(self):
        size = self._queue.qsize()
        if size > self._max_queue_size:
            for _ in range(size - self._max_queue_size):
                self._queue.get()
            size = self._max_queue_size
        return size