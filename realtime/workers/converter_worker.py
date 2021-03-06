from multiprocessing import Queue
from sprocket.speech.feature_extractor import FeatureExtractor
from sprocket.speech.synthesizer import Synthesizer
from sprocket.model.f0statistics import F0statistics
from sprocket.model.GMM import GMMConvertor
from sprocket.model.gv import GV
from realtime.workers import configs
from sprocket.util.delta import static_delta
import numpy


class ConverterWorker:
    '''
    特徴量→変換特徴量→修正特徴量→変換音声
    '''

    def __init__(self,
                 feature_queue: Queue, converted_queue: Queue,
                 mcep_gmm_config: configs.McepGMMConfig,
                 f0_stats_config: configs.F0StatsConfig,
                 gv_config: configs.GVConfig,
                 synthesizer_config: configs.SynthesizerConfig
                 ):

        self._mcep_gmm = GMMConvertor(
            n_mix=mcep_gmm_config.n_mix,
            covtype=mcep_gmm_config.covtype,
            gmmmode=None
        )
        self._mcep_gmm.open_from_param(mcep_gmm_config.param)
        self._mcep_gmm_config = mcep_gmm_config

        self._feature_queue: Queue = feature_queue
        self._converted_queue: Queue = converted_queue

        self._f0_stats = F0statistics()
        self._f0_stats_config = f0_stats_config

        self._mcep_gv = GV()
        self._mcep_gv_config = gv_config

        self._synthesizer = Synthesizer(
            fs=synthesizer_config.fs,
            fftl=synthesizer_config.fftl,
            shiftms=synthesizer_config.shiftms
        )
        self._synthesizer_config = synthesizer_config

    def convert_from_feature(self, f0, spc, ap, mcep) -> numpy.ndarray:
        cv_f0 = self._f0_stats.convert(f0, self._f0_stats_config.source_stats, self._f0_stats_config.target_stats)

        cv_mcep_wopow = self._mcep_gmm.convert(
            static_delta(mcep[:, 1:]),
            cvtype=self._mcep_gmm_config.cvtype
        )
        cv_mcep = numpy.c_[mcep[:, 0], cv_mcep_wopow]

        cv_mcep_wGV = self._mcep_gv.postfilter(
            cv_mcep,
            self._mcep_gv_config.target_stats,
            cvgvstats=self._mcep_gv_config.cvgv_stats,
            alpha=self._mcep_gv_config.morph_coeff,
            startdim=1
        )

        output_wav = self._synthesizer.synthesis(
            cv_f0,
            cv_mcep_wGV,
            ap,
            rmcep=mcep,
            alpha=self._synthesizer_config.mcep_alpha
        )
        return output_wav.clip(-32768, 32767).astype(numpy.core.int16)

    def start(self):
        while True:
            feature = self._feature_queue.get()  # 同期処理
            f0, spc, ap, mcep = feature
            output_wav = self.convert_from_feature(f0, spc, ap, mcep)
            self._converted_queue.put(output_wav)
