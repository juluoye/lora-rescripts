from __future__ import annotations

import argparse

from library.utils import add_logging_arguments


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)

    parser.add_argument(
        "--sdxl", action="store_true", help="load Stable Diffusion XL model / Stable Diffusion XLのモデルを読み込む"
    )
    parser.add_argument(
        "--v1", action="store_true", help="load Stable Diffusion v1.x model / Stable Diffusion 1.xのモデルを読み込む"
    )
    parser.add_argument(
        "--v2", action="store_true", help="load Stable Diffusion v2.0 model / Stable Diffusion 2.0のモデルを読み込む"
    )
    parser.add_argument(
        "--v_parameterization", action="store_true", help="enable v-parameterization training / v-parameterization学習を有効にする"
    )
    parser.add_argument(
        "--zero_terminal_snr",
        action="store_true",
        help="fix noise scheduler betas to enforce zero terminal SNR / noise schedulerのbetasを修正して、zero terminal SNRを強制する",
    )
    parser.add_argument(
        "--pyramid_noise_prob", type=float, default=None, help="probability for pyramid noise / ピラミッドノイズの確率"
    )
    parser.add_argument(
        "--pyramid_noise_discount_range",
        type=float,
        nargs=2,
        default=None,
        help="discount range for pyramid noise / ピラミッドノイズの割引範囲",
    )
    parser.add_argument(
        "--noise_offset_prob", type=float, default=None, help="probability for noise offset / ノイズオフセットの確率"
    )
    parser.add_argument(
        "--noise_offset_range", type=float, nargs=2, default=None, help="range for noise offset / ノイズオフセットの範囲"
    )

    parser.add_argument("--prompt", type=str, default=None, help="prompt / プロンプト")
    parser.add_argument(
        "--from_file",
        type=str,
        default=None,
        help="if specified, load prompts from this file / 指定時はプロンプトをファイルから読み込む",
    )
    parser.add_argument(
        "--from_module",
        type=str,
        default=None,
        help="if specified, load prompts from this module / 指定時はプロンプトをモジュールから読み込む",
    )
    parser.add_argument(
        "--prompter_module_args", type=str, default=None, help="args for prompter module / prompterモジュールの引数"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="interactive mode (generates one image) / 対話モード（生成される画像は1枚になります）",
    )
    parser.add_argument(
        "--no_preview", action="store_true", help="do not show generated image in interactive mode / 対話モードで画像を表示しない"
    )
    parser.add_argument(
        "--image_path", type=str, default=None, help="image to inpaint or to generate from / img2imgまたはinpaintを行う元画像"
    )
    parser.add_argument("--mask_path", type=str, default=None, help="mask in inpainting / inpaint時のマスク")
    parser.add_argument("--strength", type=float, default=None, help="img2img strength / img2img時のstrength")
    parser.add_argument("--images_per_prompt", type=int, default=1, help="number of images per prompt / プロンプトあたりの出力枚数")
    parser.add_argument("--outdir", type=str, default="outputs", help="dir to write results to / 生成画像の出力先")
    parser.add_argument(
        "--sequential_file_name", action="store_true", help="sequential output file name / 生成画像のファイル名を連番にする"
    )
    parser.add_argument(
        "--use_original_file_name",
        action="store_true",
        help="prepend original file name in img2img / img2imgで元画像のファイル名を生成画像のファイル名の先頭に付ける",
    )
    parser.add_argument("--n_iter", type=int, default=1, help="sample this often / 繰り返し回数")
    parser.add_argument("--H", type=int, default=None, help="image height, in pixel space / 生成画像高さ")
    parser.add_argument("--W", type=int, default=None, help="image width, in pixel space / 生成画像幅")
    parser.add_argument(
        "--original_height",
        type=int,
        default=None,
        help="original height for SDXL conditioning / SDXLの条件付けに用いるoriginal heightの値",
    )
    parser.add_argument(
        "--original_width",
        type=int,
        default=None,
        help="original width for SDXL conditioning / SDXLの条件付けに用いるoriginal widthの値",
    )
    parser.add_argument(
        "--original_height_negative",
        type=int,
        default=None,
        help="original height for SDXL unconditioning / SDXLのネガティブ条件付けに用いるoriginal heightの値",
    )
    parser.add_argument(
        "--original_width_negative",
        type=int,
        default=None,
        help="original width for SDXL unconditioning / SDXLのネガティブ条件付けに用いるoriginal widthの値",
    )
    parser.add_argument(
        "--crop_top", type=int, default=None, help="crop top for SDXL conditioning / SDXLの条件付けに用いるcrop topの値"
    )
    parser.add_argument(
        "--crop_left", type=int, default=None, help="crop left for SDXL conditioning / SDXLの条件付けに用いるcrop leftの値"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size / バッチサイズ")
    parser.add_argument(
        "--vae_batch_size",
        type=float,
        default=None,
        help="batch size for VAE, < 1.0 for ratio / VAE処理時のバッチサイズ、1未満の値の場合は通常バッチサイズの比率",
    )
    parser.add_argument(
        "--vae_slices",
        type=int,
        default=None,
        help="number of slices to split image into for VAE to reduce VRAM usage, None for no splitting (default), slower if specified. 16 or 32 recommended / VAE処理時にVRAM使用量削減のため画像を分割するスライス数、Noneの場合は分割しない（デフォルト）、指定すると遅くなる。16か32程度を推奨",
    )
    parser.add_argument(
        "--no_half_vae", action="store_true", help="do not use fp16/bf16 precision for VAE / VAE処理時にfp16/bf16を使わない"
    )
    parser.add_argument("--steps", type=int, default=50, help="number of ddim sampling steps / サンプリングステップ数")
    parser.add_argument(
        "--sampler",
        type=str,
        default="ddim",
        choices=[
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
        ],
        help="sampler (scheduler) type / サンプラー（スケジューラ）の種類",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty)) / guidance scale",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="path to checkpoint of model / モデルのcheckpointファイルまたはディレクトリ"
    )
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="path to checkpoint of vae to replace / VAEを入れ替える場合、VAEのcheckpointファイルまたはディレクトリ",
    )
    parser.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default=None,
        help="directory for caching Tokenizer (for offline training) / Tokenizerをキャッシュするディレクトリ（ネット接続なしでの学習のため）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed, or seed of seeds in multiple generation / 1枚生成時のseed、または複数枚生成時の乱数seedを決めるためのseed",
    )
    parser.add_argument(
        "--iter_same_seed",
        action="store_true",
        help="use same seed for all prompts in iteration if no seed specified / 乱数seedの指定がないとき繰り返し内はすべて同じseedを使う（プロンプト間の差異の比較用）",
    )
    parser.add_argument(
        "--shuffle_prompts",
        action="store_true",
        help="shuffle prompts in iteration / 繰り返し内のプロンプトをシャッフルする",
    )
    parser.add_argument("--fp16", action="store_true", help="use fp16 / fp16を指定し省メモリ化する")
    parser.add_argument("--bf16", action="store_true", help="use bfloat16 / bfloat16を指定し省メモリ化する")
    parser.add_argument("--xformers", action="store_true", help="use xformers / xformersを使用し高速化する")
    parser.add_argument("--sdpa", action="store_true", help="use sdpa in PyTorch 2 / sdpa")
    parser.add_argument(
        "--diffusers_xformers",
        action="store_true",
        help="use xformers by diffusers (Hypernetworks doesn't work) / Diffusersでxformersを使用する（Hypernetwork利用不可）",
    )
    parser.add_argument(
        "--opt_channels_last",
        action="store_true",
        help="set channels last option to model / モデルにchannels lastを指定し最適化する",
    )
    parser.add_argument(
        "--network_module",
        type=str,
        default=None,
        nargs="*",
        help="additional network module to use / 追加ネットワークを使う時そのモジュール名",
    )
    parser.add_argument(
        "--network_weights", type=str, default=None, nargs="*", help="additional network weights to load / 追加ネットワークの重み"
    )
    parser.add_argument(
        "--network_mul", type=float, default=None, nargs="*", help="additional network multiplier / 追加ネットワークの効果の倍率"
    )
    parser.add_argument(
        "--network_args",
        type=str,
        default=None,
        nargs="*",
        help="additional arguments for network (key=value) / ネットワークへの追加の引数",
    )
    parser.add_argument(
        "--network_show_meta", action="store_true", help="show metadata of network model / ネットワークモデルのメタデータを表示する"
    )
    parser.add_argument(
        "--network_merge_n_models",
        type=int,
        default=None,
        help="merge this number of networks / この数だけネットワークをマージする",
    )
    parser.add_argument(
        "--network_merge", action="store_true", help="merge network weights to original model / ネットワークの重みをマージする"
    )
    parser.add_argument(
        "--network_pre_calc",
        action="store_true",
        help="pre-calculate network for generation / ネットワークのあらかじめ計算して生成する",
    )
    parser.add_argument(
        "--network_regional_mask_max_color_codes",
        type=int,
        default=None,
        help="max color codes for regional mask (default is None, mask by channel) / regional maskの最大色数（デフォルトはNoneでチャンネルごとのマスク）",
    )
    parser.add_argument(
        "--textual_inversion_embeddings",
        type=str,
        default=None,
        nargs="*",
        help="Embeddings files of Textual Inversion / Textual Inversionのembeddings",
    )
    parser.add_argument(
        "--clip_skip",
        type=int,
        default=None,
        help="layer number from bottom to use in CLIP, default is 1 for SD1/2, 2 for SDXL / CLIPの後ろからn層目の出力を使う（デフォルトはSD1/2の場合1、SDXLの場合2）",
    )
    parser.add_argument(
        "--max_embeddings_multiples",
        type=int,
        default=None,
        help="max embedding multiples, max token length is 75 * multiples / トークン長をデフォルトの何倍とするか 75*この値 がトークン長となる",
    )
    parser.add_argument(
        "--emb_normalize_mode",
        type=str,
        default="original",
        choices=["original", "none", "abs"],
        help="embedding normalization mode / embeddingの正規化モード",
    )
    parser.add_argument(
        "--force_scheduler_zero_steps_offset",
        action="store_true",
        help="force scheduler steps offset to zero / スケジューラのステップオフセットをスケジューラ設定の `steps_offset` の値に関わらず強制的にゼロにする",
    )
    parser.add_argument(
        "--guide_image_path", type=str, default=None, nargs="*", help="image to ControlNet / ControlNetでガイドに使う画像"
    )
    parser.add_argument(
        "--highres_fix_scale",
        type=float,
        default=None,
        help="enable highres fix, reso scale for 1st stage / highres fixを有効にして最初の解像度をこのscaleにする",
    )
    parser.add_argument(
        "--highres_fix_steps",
        type=int,
        default=28,
        help="1st stage steps for highres fix / highres fixの最初のステージのステップ数",
    )
    parser.add_argument(
        "--highres_fix_strength",
        type=float,
        default=None,
        help="1st stage img2img strength for highres fix / highres fixの最初のステージのimg2img時のstrength、省略時はstrengthと同じ",
    )
    parser.add_argument(
        "--highres_fix_save_1st",
        action="store_true",
        help="save 1st stage images for highres fix / highres fixの最初のステージの画像を保存する",
    )
    parser.add_argument(
        "--highres_fix_latents_upscaling",
        action="store_true",
        help="use latents upscaling for highres fix / highres fixでlatentで拡大する",
    )
    parser.add_argument(
        "--highres_fix_upscaler",
        type=str,
        default=None,
        help="upscaler module for highres fix / highres fixで使うupscalerのモジュール名",
    )
    parser.add_argument(
        "--highres_fix_upscaler_args",
        type=str,
        default=None,
        help="additional arguments for upscaler (key=value) / upscalerへの追加の引数",
    )
    parser.add_argument(
        "--highres_fix_disable_control_net",
        action="store_true",
        help="disable ControlNet for highres fix / highres fixでControlNetを使わない",
    )

    parser.add_argument(
        "--negative_scale",
        type=float,
        default=None,
        help="set another guidance scale for negative prompt / ネガティブプロンプトのscaleを指定する",
    )

    parser.add_argument(
        "--control_net_lllite_models",
        type=str,
        default=None,
        nargs="*",
        help="ControlNet models to use / 使用するControlNetのモデル名",
    )
    parser.add_argument(
        "--control_net_models", type=str, default=None, nargs="*", help="ControlNet models to use / 使用するControlNetのモデル名"
    )
    parser.add_argument(
        "--control_net_preps",
        type=str,
        default=None,
        nargs="*",
        help="ControlNet preprocess to use / 使用するControlNetのプリプロセス名",
    )
    parser.add_argument(
        "--control_net_multipliers", type=float, default=None, nargs="*", help="ControlNet multiplier / ControlNetの適用率"
    )
    parser.add_argument(
        "--control_net_ratios",
        type=float,
        default=None,
        nargs="*",
        help="ControlNet guidance ratio for steps / ControlNetでガイドするステップ比率",
    )
    parser.add_argument(
        "--clip_vision_strength",
        type=float,
        default=None,
        help="enable CLIP Vision Conditioning for img2img with this strength / img2imgでCLIP Vision Conditioningを有効にしてこのstrengthで処理する",
    )

    parser.add_argument(
        "--ds_depth_1",
        type=int,
        default=None,
        help="Enable Deep Shrink with this depth 1, valid values are 0 to 8 / Deep Shrinkをこのdepthで有効にする",
    )
    parser.add_argument(
        "--ds_timesteps_1",
        type=int,
        default=650,
        help="Apply Deep Shrink depth 1 until this timesteps / Deep Shrink depth 1を適用するtimesteps",
    )
    parser.add_argument("--ds_depth_2", type=int, default=None, help="Deep Shrink depth 2 / Deep Shrinkのdepth 2")
    parser.add_argument(
        "--ds_timesteps_2",
        type=int,
        default=650,
        help="Apply Deep Shrink depth 2 until this timesteps / Deep Shrink depth 2を適用するtimesteps",
    )
    parser.add_argument(
        "--ds_ratio", type=float, default=0.5, help="Deep Shrink ratio for downsampling / Deep Shrinkのdownsampling比率"
    )

    parser.add_argument(
        "--gradual_latent_timesteps",
        type=int,
        default=None,
        help="enable Gradual Latent hires fix and apply upscaling from this timesteps / Gradual Latent hires fixをこのtimestepsで有効にし、このtimestepsからアップスケーリングを適用する",
    )
    parser.add_argument(
        "--gradual_latent_ratio",
        type=float,
        default=0.5,
        help=" this size ratio, 0.5 means 1/2 / Gradual Latent hires fixをこのサイズ比率で有効にする、0.5は1/2を意味する",
    )
    parser.add_argument(
        "--gradual_latent_ratio_step",
        type=float,
        default=0.125,
        help="step to increase ratio for Gradual Latent / Gradual Latentのratioをどのくらいずつ上げるか",
    )
    parser.add_argument(
        "--gradual_latent_every_n_steps",
        type=int,
        default=3,
        help="steps to increase size of latents every this steps for Gradual Latent / Gradual Latentでlatentsのサイズをこのステップごとに上げる",
    )
    parser.add_argument(
        "--gradual_latent_s_noise",
        type=float,
        default=1.0,
        help="s_noise for Gradual Latent / Gradual Latentのs_noise",
    )
    parser.add_argument(
        "--gradual_latent_unsharp_params",
        type=str,
        default=None,
        help="unsharp mask parameters for Gradual Latent: ksize, sigma, strength, target-x (1 means True). `3,0.5,0.5,1` or `3,1.0,1.0,0` is recommended / Gradual Latentのunsharp maskのパラメータ: ksize, sigma, strength, target-x. `3,0.5,0.5,1` または `3,1.0,1.0,0` が推奨",
    )

    return parser


__all__ = ["setup_parser"]
