import argparse
import logging
import sys
from multiprocessing import Process, Queue
from pathlib import Path

import joblib

from example.src.yml import SpeakerYML, PairYML
from realtime.util import audioio
from realtime.workers import converter_worker, feature_extractor_worker
from realtime.workers import configs
from sprocket.util.hdf5 import HDF5

logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter("[%(asctime)s] [%(process)d] [%(name)s] [%(levelname)s] %(message)s"))
logger.addHandler(handler)

MAX_RECORDED_QUEUE_SIZE = 2

if __name__ == '__main__':
    # 引数を解析する
    logger.info('引数を解析中')
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog='realtime-sprocket',
        description='exampleにて作成した声質変換モデルを用いてリアルタイム音声変換を行う。'
    )
    parser.add_argument('source', help='変換前の話者名', type=str)
    parser.add_argument('target', help='変換後の話者名', type=str)
    parser.add_argument('--i', help='入力デバイスの番号', type=int, default=None)
    parser.add_argument('--o', help='出力デバイスの番号', type=int, default=None)
    parser.add_argument('--alpha', help='オールパスフィルターの係数', type=float, default=None)
    parser.add_argument('--frames', help='フレームあたりのバッファ数', type=int, default=16384)
    args = parser.parse_args(argv)

    # パスに関係する定数
    logger.info('パスを生成中')
    PAIR = '{}-{}'.format(args.source, args.target)
    CURRENT_DIR = Path(__file__).parent
    ROOT_DIR = CURRENT_DIR.parent
    EXAMPLE_DIR = ROOT_DIR / 'example'
    CONF_DIR = EXAMPLE_DIR / 'conf'
    DATA_DIR = EXAMPLE_DIR / 'data'
    PAIR_DIR = DATA_DIR / 'pair' / PAIR
    STATS_DIR = PAIR_DIR / 'stats'
    MODEL_DIR = PAIR_DIR / 'model'

    # 設定を読み込む
    logger.info('設定を読み込み中')
    speaker_config_fp = CONF_DIR / 'speaker' / '{}.yml'.format(args.source)
    speaker_config = SpeakerYML(speaker_config_fp)
    pair_config_fp = CONF_DIR / 'pair' / '{}.yml'.format(PAIR)
    pair_config = PairYML(pair_config_fp)

    # 統計情報を読み込む
    logger.info('統計情報を読み込み中')
    mcep_gmm_param_fp = MODEL_DIR / 'GMM_mcep.pkl'
    mcep_gmm_param = joblib.load(mcep_gmm_param_fp)

    source_stats_h5 = HDF5(STATS_DIR / '{}.h5'.format(args.source), mode='r')
    source_f0stats = source_stats_h5.read(ext='f0stats')
    source_stats_h5.close()

    target_stats_h5 = HDF5(STATS_DIR / '{}.h5'.format(args.target), mode='r')
    target_f0stats = target_stats_h5.read(ext='f0stats')
    target_gv = target_stats_h5.read(ext='gv')
    target_stats_h5.close()

    cvgv_h5 = HDF5(MODEL_DIR / 'cvgv.h5', mode='r')
    cvgv = cvgv_h5.read(ext='cvgv')
    cvgv_h5.close()

    # プログラム実行に必要なパラメータを作る
    mcep_alpha = args.alpha or speaker_config.mcep_alpha

    logger.info('パラメータを生成中')
    feature_extractor_config = configs.FeatureExtractorConfig(
        fs=speaker_config.wav_fs,
        fftl=speaker_config.wav_fftl,
        shiftms=speaker_config.wav_shiftms,
        minf0=speaker_config.f0_minf0,
        maxf0=speaker_config.f0_maxf0,
        mcep_dim=speaker_config.mcep_dim,
        mcep_alpha=mcep_alpha
    )
    logger.debug(feature_extractor_config)

    mcep_gmm_config = configs.McepGMMConfig(
        param=mcep_gmm_param,
        n_mix=pair_config.GMM_mcep_n_mix,
        covtype=pair_config.GMM_mcep_covtype,
        cvtype=pair_config.GMM_mcep_cvtype
        # gmmode = None
    )
    logger.debug(mcep_gmm_config)

    f0stats_config = configs.F0StatsConfig(
        source_stats=source_f0stats,
        target_stats=target_f0stats
    )
    logger.debug(f0stats_config)

    gv_config = configs.GVConfig(
        target_stats=target_gv,
        cvgv_stats=cvgv,
        morph_coeff=pair_config.GV_morph_coeff
    )
    logger.debug(gv_config)

    synthesizer_config = configs.SynthesizerConfig(
        fs=speaker_config.wav_fs,
        fftl=speaker_config.wav_fftl,
        shiftms=speaker_config.wav_shiftms,
        mcep_alpha=mcep_alpha
    )
    logger.debug(synthesizer_config)

    # 入出力デバイスを決定する
    input_device_index = args.i
    while input_device_index is None:
        print('以下より入力デバイスの番号を指定')
        available_input_devices = audioio.get_available_input_devices()
        for k, v in available_input_devices.items():
            print('{:03d}'.format(k), ':', v)
        x = int(input())
        if 0 <= x < len(available_input_devices):
            input_device_index = x

    output_device_index = args.o
    while output_device_index is None:
        print('以下より出力デバイスの番号を指定')
        available_output_devices = audioio.get_available_output_devices()
        for k, v in available_output_devices.items():
            print('{:03d}'.format(k), ':', v)
        x = int(input())
        if 0 <= x < len(available_output_devices):
            output_device_index = x

    # マイクとスピーカーを起動
    logger.info('マイクを起動中')
    microphone = audioio.Microphone(
        frames_per_buffer=args.frames,
        frame_rate=speaker_config.wav_fs,
        input_device_index=input_device_index
    )
    logger.info('スピーカーを起動中')
    speaker = audioio.Speaker(
        frames_per_buffer=args.frames,
        frame_rate=speaker_config.wav_fs,
        output_device_index=output_device_index
    )

    # モデルを起動
    logger.info('モデルを起動中')
    recorded_queue = Queue()
    feature_queue = Queue()
    converted_queue = Queue()

    feature_extractor = feature_extractor_worker.FeatureExtractorWorker(
        recorded_queue=recorded_queue,
        feature_queue=feature_queue,
        feature_extractor_config=feature_extractor_config
    )

    converter = converter_worker.ConverterWorker(
        feature_queue=feature_queue,
        converted_queue=converted_queue,
        mcep_gmm_config=mcep_gmm_config,
        f0_stats_config=f0stats_config,
        gv_config=gv_config,
        synthesizer_config=synthesizer_config
    )

    # 声質変換器を起動
    logger.info('声質変換器を起動中')
    feature_extractor_process = Process(target=feature_extractor.start)
    converter_process = Process(target=converter.start)

    feature_extractor_process.start()
    converter_process.start()

    while True:
        recorded_queue_size = recorded_queue.qsize()
        logger.debug('録音キュー:{} 特徴量キュー:{} 変換済みキュー:{}'.format(
            recorded_queue_size,
            feature_queue.qsize(),
            converted_queue.qsize()
        ))

        if recorded_queue_size > MAX_RECORDED_QUEUE_SIZE:
            num_remove = recorded_queue_size - MAX_RECORDED_QUEUE_SIZE
            logger.debug('録音キューから強制的に{}個取り除く'.format(num_remove))
            for _ in range(num_remove):
                recorded_queue.get()

        recorded_wav = microphone.read_frame()
        recorded_queue.put(recorded_wav)
        if not converted_queue.empty():
            converted_wav = converted_queue.get()
            speaker.play_frame(converted_wav)
        else:
            logger.debug('待機中')
