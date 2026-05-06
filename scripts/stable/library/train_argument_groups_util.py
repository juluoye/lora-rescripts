from __future__ import annotations

import argparse

from library.argument_help_util import build_add


_DYNAMO_BACKENDS = [
    "eager",
    "aot_eager",
    "inductor",
    "aot_ts_nvfuser",
    "nvprims_nvfuser",
    "cudagraphs",
    "ofi",
    "fx2trt",
    "onnxrt",
    "tensort",
    "ipex",
    "tvm",
]

_SAMPLE_SAMPLERS = [
    "ddim",
    "pndm",
    "lms",
    "euler",
    "euler_a",
    "heun",
    "dpm_2",
    "dpm_2_a",
    "dpmsolver",
    "dpmsolver++",
    "dpmsingle",
    "k_lms",
    "k_euler",
    "k_euler_a",
    "k_dpm_2",
    "k_dpm_2_a",
]

def add_output_and_huggingface_arguments(parser: argparse.ArgumentParser) -> None:
    add = build_add(parser)
    add("--output_dir", type=str, default=None, help="directory to output trained model / 学習後のモデル出力先ディレクトリ")
    add("--output_name", type=str, default=None, help="base name of trained model file / 学習後のモデルの拡張子を除くファイル名")
    add("--huggingface_repo_id", type=str, default=None, help="huggingface repo name to upload / huggingfaceにアップロードするリポジトリ名")
    add("--huggingface_repo_type", type=str, default=None, help="huggingface repo type to upload / huggingfaceにアップロードするリポジトリの種類")
    add("--huggingface_path_in_repo", type=str, default=None, help="huggingface model path to upload files / huggingfaceにアップロードするファイルのパス")
    add("--huggingface_token", type=str, default=None, help="huggingface token / huggingfaceのトークン")
    add("--huggingface_repo_visibility", type=str, default=None, help="huggingface repository visibility ('public' for public, 'private' or None for private) / huggingfaceにアップロードするリポジトリの公開設定（'public'で公開、'private'またはNoneで非公開）")
    add("--save_state_to_huggingface", action="store_true", help="save state to huggingface / huggingfaceにstateを保存する")
    add("--resume_from_huggingface", action="store_true", help="resume from huggingface (ex: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type}) / huggingfaceから学習を再開する(例: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type})")
    add("--async_upload", action="store_true", help="upload to huggingface asynchronously / huggingfaceに非同期でアップロードする")
    add("--save_precision", type=str, default=None, choices=[None, "float", "fp16", "bf16"], help="precision in saving / 保存時に精度を変更して保存する")


def add_checkpoint_and_resume_arguments(parser: argparse.ArgumentParser) -> None:
    add = build_add(parser)
    add("--save_every_n_epochs", type=int, default=None, help="save checkpoint every N epochs / 学習中のモデルを指定エポックごとに保存する")
    add("--save_every_n_steps", type=int, default=None, help="save checkpoint every N steps / 学習中のモデルを指定ステップごとに保存する")
    add("--save_n_epoch_ratio", type=int, default=None, help="save checkpoint N epoch ratio (for example 5 means save at least 5 files total) / 学習中のモデルを指定のエポック割合で保存する（たとえば5を指定すると最低5個のファイルが保存される）")
    add("--save_last_n_epochs", type=int, default=None, help="save last N checkpoints when saving every N epochs (remove older checkpoints) / 指定エポックごとにモデルを保存するとき最大Nエポック保存する（古いチェックポイントは削除する）")
    add("--save_last_n_epochs_state", type=int, default=None, help="save last N checkpoints of state (overrides the value of --save_last_n_epochs)/ 最大Nエポックstateを保存する（--save_last_n_epochsの指定を上書きする）")
    add("--save_last_n_steps", type=int, default=None, help="save checkpoints until N steps elapsed (remove older checkpoints if N steps elapsed) / 指定ステップごとにモデルを保存するとき、このステップ数経過するまで保存する（このステップ数経過したら削除する）")
    add("--save_last_n_steps_state", type=int, default=None, help="save states until N steps elapsed (remove older states if N steps elapsed, overrides --save_last_n_steps) / 指定ステップごとにstateを保存するとき、このステップ数経過するまで保存する（このステップ数経過したら削除する。--save_last_n_stepsを上書きする）")
    add("--save_state", action="store_true", help="save training state additionally (including optimizer states etc.) when saving model / optimizerなど学習状態も含めたstateをモデル保存時に追加で保存する")
    add("--save_state_on_train_end", action="store_true", help="save training state (including optimizer states etc.) on train end / optimizerなど学習状態も含めたstateを学習完了時に保存する")
    add("--cooldown_every_n_epochs", type=int, default=None, help="cool down training every N epochs after epoch-end save/preview / 各epoch末尾の保存・プレビュー後にNエポックごとにクールダウンする / 每 N 个 epoch 在轮次结束保存与预览完成后执行一次冷却暂停")
    add("--cooldown_minutes", type=float, default=None, help="minimum cooldown time in minutes for each cooldown pause / クールダウン時に最低限待機する分数 / 每次冷却至少暂停多少分钟")
    add("--cooldown_until_temp_c", type=int, default=None, help="wait until local training GPU temperature drops to this Celsius target / ローカル学習GPU温度がこの摂氏温度以下になるまで待機する / 等待本机训练显卡温度降到该摄氏度以下再继续")
    add("--cooldown_poll_seconds", type=int, default=15, help="polling interval in seconds when waiting for GPU temperature / GPU温度待機時のポーリング間隔（秒） / 按温度等待时的轮询间隔（秒）")
    add("--gpu_power_limit_w", type=int, default=None, help="set whole-GPU power limit in watts before training starts (not per-process) / 学習開始前にGPU全体の電力上限をワットで設定する（プロセス単位ではない） / 训练开始前尝试设置整张训练显卡的功率墙，单位瓦，不是单个进程限制")
    add("--resume", type=str, default=None, help="saved state to resume training / 学習再開するモデルのstate")


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    add = build_add(parser)
    add("--train_batch_size", type=int, default=1, help="batch size for training / 学習時のバッチサイズ")
    add("--max_token_length", type=int, default=None, choices=[None, 150, 225], help="max token length of text encoder (default for 75, 150 or 225) / text encoderのトークンの最大長（未指定で75、150または225が指定可）")
    add("--mem_eff_attn", action="store_true", help="use memory efficient attention for CrossAttention / CrossAttentionに省メモリ版attentionを使う")
    add(
        "--torch_compile",
        action="store_true",
        help="use torch.compile (requires PyTorch 2.0) / 使用 torch.compile（需要 PyTorch 2.0） / torch.compile を使う",
    )
    add("--dynamo_backend", type=str, default="inductor", choices=_DYNAMO_BACKENDS, help="dynamo backend type (default is inductor) / dynamoのbackendの種類（デフォルトは inductor）")
    add("--opt_channels_last", action="store_true", help="set channels last memory format for convolution-heavy training models / 畳み込み系モデルでchannels lastメモリ形式を使う / 为卷积型训练模型启用 channels_last 内存格式")
    add("--xformers", action="store_true", help="use xformers for CrossAttention / CrossAttentionにxformersを使う")
    add("--sdpa", action="store_true", help="use sdpa for CrossAttention (requires PyTorch 2.0) / CrossAttentionにsdpaを使う（PyTorch 2.0が必要）")
    add("--sageattn", action="store_true", help="use SageAttention for CrossAttention (experimental; requires SageAttention runtime) / CrossAttentionにSageAttentionを使う（実験的・SageAttention実行環境が必要） / 为 CrossAttention 启用 SageAttention（实验性，需要 SageAttention 环境）")
    add("--flashattn", action="store_true", help="use FlashAttention 2 for CrossAttention (experimental; requires flash-attn runtime) / CrossAttentionにFlashAttention 2を使う（実験的・flash-attn実行環境が必要） / 为 CrossAttention 启用 FlashAttention 2（实验性，需要 flash-attn 环境）")
    add("--cross_attn_fused_kv", action="store_true", help="enable experimental fused K/V projection for SDXL cross attention / SDXL の cross attention に fused K/V projection 実験機能を使う / 为 SDXL 的 cross attention 启用 fused K/V projection 实验功能")
    add("--vae", type=str, default=None, help="path to checkpoint of vae to replace / VAEを入れ替える場合、VAEのcheckpointファイルまたはディレクトリ")
    add("--max_train_steps", type=int, default=1600, help="training steps / 学習ステップ数")
    add("--max_train_epochs", type=int, default=None, help="training epochs (overrides max_train_steps) / 学習エポック数（max_train_stepsを上書きします）")
    add("--max_data_loader_n_workers", type=int, default=8, help="max num workers for DataLoader (lower is less main RAM usage, faster epoch start and slower data loading) / DataLoaderの最大プロセス数（小さい値ではメインメモリの使用量が減りエポック間の待ち時間が減りますが、データ読み込みは遅くなります）")
    add("--persistent_data_loader_workers", action="store_true", help="persistent DataLoader workers (useful for reduce time gap between epoch, but may use more memory) / DataLoader のワーカーを持続させる (エポック間の時間差を少なくするのに有効だが、より多くのメモリを消費する可能性がある)")
    add("--seed", type=int, default=None, help="random seed for training / 学習時の乱数のseed")
    add("--ema_enabled", action="store_true", help="enable EMA (Exponential Moving Average) over trainable model weights during training / 学習中に学習対象重みのEMAを有効化する")
    add("--ema_decay", type=float, default=0.999, help="EMA decay ratio / EMA の減衰率")
    add("--ema_update_every", type=int, default=1, help="update EMA every N optimizer steps / N ステップごとに EMA を更新する")
    add("--ema_update_after_step", type=int, default=0, help="start EMA updates after this optimizer step / この step 以降で EMA 更新を開始する")
    add("--ema_use_warmup", action="store_true", help="use warmup schedule for EMA decay / EMA の減衰率にウォームアップを使う")
    add("--ema_inv_gamma", type=float, default=1.0, help="EMA warmup inverse gamma / EMA ウォームアップの inverse gamma")
    add("--ema_power", type=float, default=0.75, help="EMA warmup power / EMA ウォームアップの power")
    add("--safeguard_enabled", action="store_true", help="enable lightweight training safeguard / 軽量版トレーニング SafeGuard を有効化する")
    add("--safeguard_nan_check_interval", type=int, default=1, help="check non-finite loss every N optimizer steps / N ステップごとに NaN・Inf loss をチェックする")
    add("--safeguard_max_nan_count", type=int, default=3, help="stop training after this many non-finite losses / この回数だけ NaN・Inf loss が続いたら停止する")
    add("--safeguard_loss_spike_threshold", type=float, default=5.0, help="skip a step if current loss is this multiple of the rolling average / 現在 loss が移動平均のこの倍数を超えたらその step をスキップする")
    add("--safeguard_loss_window_size", type=int, default=20, help="rolling window size used for loss spike detection / loss スパイク判定に使う移動窓サイズ")
    add("--safeguard_auto_reduce_lr", action="store_true", help="automatically reduce LR when SafeGuard triggers / SafeGuard 発動時に自動で学習率を下げる")
    add("--safeguard_lr_reduction_factor", type=float, default=0.5, help="learning-rate multiplier used when SafeGuard reduces LR / SafeGuard が学習率を下げるときの倍率")
    add("--gradient_checkpointing", action="store_true", help="enable gradient checkpointing / gradient checkpointingを有効にする")
    add("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass / 学習時に逆伝播をする前に勾配を合計するステップ数")
    add("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="use mixed precision / 混合精度を使う場合、その精度")
    add("--full_fp16", action="store_true", help="fp16 training including gradients, some models are not supported / 勾配も含めてfp16で学習する、一部のモデルではサポートされていません")
    add(
        "--full_bf16",
        action="store_true",
        help="bf16 training including gradients, some models are not supported / 包括梯度在内都使用 bf16 训练，部分模型暂不支持 / 勾配も含めてbf16で学習する、一部のモデルではサポートされていません",
    )
    add("--fp8_base", action="store_true", help="use fp8 for base model, some models are not supported / base modelにfp8を使う、一部のモデルではサポートされていません")
    add("--ddp_timeout", type=int, default=None, help="DDP timeout (min, None for default of accelerate) / DDPのタイムアウト（分、Noneでaccelerateのデフォルト）")
    add("--ddp_gradient_as_bucket_view", action="store_true", help="enable gradient_as_bucket_view for DDP / DDPでgradient_as_bucket_viewを有効にする")
    add("--ddp_static_graph", action="store_true", help="enable static_graph for DDP / DDPでstatic_graphを有効にする")
    add("--clip_skip", type=int, default=None, help="use output of nth layer from back of text encoder (n>=1) / text encoderの後ろからn番目の層の出力を用いる（nは1以上）")


def add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    add = build_add(parser)
    add("--logging_dir", type=str, default=None, help="enable logging and output TensorBoard log to this directory / ログ出力を有効にしてこのディレクトリにTensorBoard用のログを出力する")
    add("--logging_run_dir", type=str, default=None, help=argparse.SUPPRESS)
    add("--log_with", type=str, default=None, choices=["tensorboard", "wandb", "all"], help="what logging tool(s) to use (if 'all', TensorBoard and WandB are both used) / ログ出力に使用するツール (allを指定するとTensorBoardとWandBの両方が使用される)")
    add("--log_prefix", type=str, default=None, help="add prefix for each log directory / ログディレクトリ名の先頭に追加する文字列")
    add("--log_tracker_name", type=str, default=None, help="name of tracker to use for logging, default is script-specific default name / ログ出力に使用するtrackerの名前、省略時はスクリプトごとのデフォルト名")
    add("--wandb_run_name", type=str, default=None, help="The name of the specific wandb session / wandb ログに表示される特定の実行の名前")
    add("--log_tracker_config", type=str, default=None, help="path to tracker config file to use for logging / ログ出力に使用するtrackerの設定ファイルのパス")
    add("--wandb_api_key", type=str, default=None, help="specify WandB API key to log in before starting training (optional). / WandB APIキーを指定して学習開始前にログインする（オプション）")
    add("--log_config", action="store_true", help="log training configuration / 学習設定をログに出力する")


def add_noise_loss_sampling_and_config_arguments(parser: argparse.ArgumentParser, support_dreambooth: bool) -> None:
    add = build_add(parser)
    add("--noise_offset", type=float, default=None, help="enable noise offset with this value (if enabled, around 0.1 is recommended) / Noise offsetを有効にしてこの値を設定する（有効にする場合は0.1程度を推奨）")
    add("--noise_offset_random_strength", action="store_true", help="use random strength between 0~noise_offset for noise offset. / noise offsetにおいて、0からnoise_offsetの間でランダムな強度を使用します。")
    add("--multires_noise_iterations", type=int, default=None, help="enable multires noise with this number of iterations (if enabled, around 6-10 is recommended) / Multires noiseを有効にしてこのイテレーション数を設定する（有効にする場合は6-10程度を推奨）")
    add("--ip_noise_gamma", type=float, default=None, help="enable input perturbation noise. used for regularization. recommended value: around 0.1 (from arxiv.org/abs/2301.11706) /  input perturbation noiseを有効にする。正則化に使用される。推奨値: 0.1程度 (arxiv.org/abs/2301.11706 より)")
    add("--ip_noise_gamma_random_strength", action="store_true", help="Use random strength between 0~ip_noise_gamma for input perturbation noise./ input perturbation noiseにおいて、0からip_noise_gammaの間でランダムな強度を使用します。")
    add("--multires_noise_discount", type=float, default=0.3, help="set discount value for multires noise (has no effect without --multires_noise_iterations) / Multires noiseのdiscount値を設定する（--multires_noise_iterations指定時のみ有効）")
    add("--adaptive_noise_scale", type=float, default=None, help="add `latent mean absolute value * this value` to noise_offset (disabled if None, default) / latentの平均値の絶対値 * この値をnoise_offsetに加算する（Noneの場合は無効、デフォルト）")
    add("--zero_terminal_snr", action="store_true", help="fix noise scheduler betas to enforce zero terminal SNR / noise schedulerのbetasを修正して、zero terminal SNRを強制する")
    add("--min_timestep", type=int, default=None, help="set minimum time step for U-Net training (0~999, default is 0) / U-Net学習時のtime stepの最小値を設定する（0~999で指定、省略時はデフォルト値(0)） ")
    add("--max_timestep", type=int, default=None, help="set maximum time step for U-Net training (1~1000, default is 1000) / U-Net学習時のtime stepの最大値を設定する（1~1000で指定、省略時はデフォルト値(1000)）")
    add("--loss_type", type=str, default="l2", choices=["l1", "l2", "huber", "smooth_l1"], help="The type of loss function to use (L1, L2, Huber, or smooth L1), default is L2 / 使用する損失関数の種類（L1、L2、Huber、またはsmooth L1）、デフォルトはL2")
    add("--huber_schedule", type=str, default="snr", choices=["constant", "exponential", "snr"], help="The scheduling method for Huber loss (constant, exponential, or SNR-based). Only used when loss_type is 'huber' or 'smooth_l1'. default is snr / Huber損失のスケジューリング方法（constant、exponential、またはSNRベース）。loss_typeが'huber'または'smooth_l1'の場合に有効、デフォルトは snr")
    add("--huber_c", type=float, default=0.1, help="The Huber loss decay parameter. Only used if one of the huber loss modes (huber or smooth l1) is selected with loss_type. default is 0.1 / Huber損失の減衰パラメータ。loss_typeがhuberまたはsmooth l1の場合に有効。デフォルトは0.1")
    add("--huber_scale", type=float, default=1.0, help="The Huber loss scale parameter. Only used if one of the huber loss modes (huber or smooth l1) is selected with loss_type. default is 1.0 / Huber損失のスケールパラメータ。loss_typeがhuberまたはsmooth l1の場合に有効。デフォルトは1.0")
    add(
        "--wavelet_loss_enabled",
        action="store_true",
        help="enable experimental additive wavelet-domain loss / 启用实验性的额外 wavelet 域损失 / 実験的な wavelet 補助損失を有効化する",
    )
    add(
        "--wavelet_loss_weight",
        type=float,
        default=0.0,
        help="weight for additive wavelet-domain loss / wavelet 域额外损失的权重 / wavelet 補助損失的权重",
    )
    add(
        "--wavelet_loss_levels",
        type=int,
        default=1,
        help="number of wavelet pyramid levels to use / 使用的 wavelet 金字塔层数 / wavelet 金字塔层数",
    )
    add(
        "--wavelet_loss_approx_weight",
        type=float,
        default=0.0,
        help="optional low-frequency (LL) wavelet loss weight on the last level / 最后一层低频（LL）wavelet 损失的可选权重 / 最后一层低频 LL 分量的可选权重",
    )
    add("--lowram", action="store_true", help="enable low RAM optimization. e.g. load models to VRAM instead of RAM (for machines which have bigger VRAM than RAM such as Colab and Kaggle) / メインメモリが少ない環境向け最適化を有効にする。たとえばVRAMにモデルを読み込む等（ColabやKaggleなどRAMに比べてVRAMが多い環境向け）")
    add("--highvram", action="store_true", help="disable low VRAM optimization. e.g. do not clear CUDA cache after each latent caching (for machines which have bigger VRAM) / VRAMが少ない環境向け最適化を無効にする。たとえば各latentのキャッシュ後のCUDAキャッシュクリアを行わない等（VRAMが多い環境向け）")
    add("--sample_every_n_steps", type=int, default=None, help="generate sample images every N steps / 学習中のモデルで指定ステップごとにサンプル出力する")
    add("--sample_at_first", action="store_true", help="generate sample images before training / 学習前にサンプル出力する")
    add("--sample_every_n_epochs", type=int, default=None, help="generate sample images every N epochs (overwrites n_steps) / 学習中のモデルで指定エポックごとにサンプル出力する（ステップ数指定を上書きします）")
    add("--sample_prompts", type=str, default=None, help="file for prompts to generate sample images / 学習中モデルのサンプル出力用プロンプトのファイル")
    add("--sample_sampler", type=str, default="ddim", choices=_SAMPLE_SAMPLERS, help="sampler (scheduler) type for sample images / サンプル出力時のサンプラー（スケジューラ）の種類")
    add("--config_file", type=str, default=None, help="using .toml instead of args to pass hyperparameter / ハイパーパラメータを引数ではなく.tomlファイルで渡す")
    add("--output_config", action="store_true", help="output command line args to given .toml file / 引数を.tomlファイルに出力する")
    if support_dreambooth:
        add("--prior_loss_weight", type=float, default=1.0, help="loss weight for regularization images / 正則化画像のlossの重み")


def add_training_arguments(parser: argparse.ArgumentParser, support_dreambooth: bool) -> None:
    add_output_and_huggingface_arguments(parser)
    add_checkpoint_and_resume_arguments(parser)
    add_runtime_arguments(parser)
    add_logging_arguments(parser)
    add_noise_loss_sampling_and_config_arguments(parser, support_dreambooth)
