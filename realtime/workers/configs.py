from dataclasses import dataclass

@dataclass
class FeatureExtractorConfig:
    fs: int
    fftl: int = 1024
    shiftms: int = 5
    minf0: float = 50.
    maxf0: float = 500.
    mcep_dim: int = 24
    mcep_alpha: float = 0.42


@dataclass
class McepGMMConfig:
    param: object
    n_mix: int = 32
    covtype: str = 'full'
    cvtype: str = 'mlpg'


@dataclass
class F0StatsConfig:
    source_stats: object
    target_stats: object


@dataclass
class GVConfig:
    target_stats: object
    cvgv_stats: object
    morph_coeff: int = 1.0


@dataclass
class SynthesizerConfig:
    fs: int
    fftl: int = 1024
    shiftms: int = 5
    mcep_alpha: float = 0.42