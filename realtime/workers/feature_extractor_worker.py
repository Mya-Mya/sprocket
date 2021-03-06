from multiprocessing import Queue
from sprocket.speech.feature_extractor import FeatureExtractor
from realtime.workers import configs
from dataclasses import dataclass
import numpy


def create_feature_extractor_result(f0, spc, ap, mcep): return (f0, spc, ap, mcep)


class FeatureExtractorWorker:
    '''
    音声→特徴量
    '''

    def __init__(self,
                 recorded_queue: Queue,
                 feature_queue: Queue,
                 feature_extractor_config: configs.FeatureExtractorConfig
                 ):
        self._recorded_queue = recorded_queue
        self._feature_queue = feature_queue
        self._feat = FeatureExtractor(
            analyzer='world',
            fs=feature_extractor_config.fs,
            fftl=feature_extractor_config.fftl,
            shiftms=feature_extractor_config.shiftms,
            minf0=feature_extractor_config.minf0,
            maxf0=feature_extractor_config.maxf0
        )
        self._feature_extractor_config = feature_extractor_config

    def start(self):
        while True:
            recorded_wav = self._recorded_queue.get()  # 同期処理
            recorded_wav = recorded_wav.astype(numpy.float)
            f0, spc, ap = self._feat.analyze(recorded_wav)
            mcep = self._feat.mcep(
                dim=self._feature_extractor_config.mcep_dim,
                alpha=self._feature_extractor_config.mcep_alpha
            )
            feature = create_feature_extractor_result(f0, spc, ap, mcep)
            self._feature_queue.put(feature)
