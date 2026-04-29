from __future__ import annotations

import argparse

from library.argument_help_util import build_add


def add_dataset_path_and_caption_arguments(parser: argparse.ArgumentParser) -> None:
    add = build_add(parser)
    add("--train_data_dir", type=str, default=None, help="directory for train images / 学習画像データのディレクトリ")
    add(
        "--cache_info",
        action="store_true",
        help="cache meta information (caption and image size) for faster dataset loading. only available for DreamBooth"
        + " / メタ情報（キャプションとサイズ）をキャッシュしてデータセット読み込みを高速化する。DreamBooth方式のみ有効",
    )
    add("--shuffle_caption", action="store_true", help="shuffle separated caption / 区切られたcaptionの各要素をshuffleする")
    add("--caption_separator", type=str, default=",", help="separator for caption / captionの区切り文字")
    add("--caption_extension", type=str, default=".caption", help="extension of caption files / 読み込むcaptionファイルの拡張子")
    add(
        "--caption_extention",
        type=str,
        default=None,
        help="extension of caption files (backward compatibility) / 読み込むcaptionファイルの拡張子（スペルミスを残してあります）",
    )
    add(
        "--keep_tokens",
        type=int,
        default=0,
        help="keep heading N tokens when shuffling caption tokens (token means comma separated strings) / captionのシャッフル時に、先頭からこの個数のトークンをシャッフルしないで残す（トークンはカンマ区切りの各部分を意味する）",
    )
    add(
        "--keep_tokens_separator",
        type=str,
        default="",
        help="A custom separator to divide the caption into fixed and flexible parts. Tokens before this separator will not be shuffled. If not specified, '--keep_tokens' will be used to determine the fixed number of tokens."
        + " / captionを固定部分と可変部分に分けるためのカスタム区切り文字。この区切り文字より前のトークンはシャッフルされない。指定しない場合、'--keep_tokens'が固定部分のトークン数として使用される。",
    )
    add(
        "--secondary_separator",
        type=str,
        default=None,
        help="a secondary separator for caption. This separator is replaced to caption_separator after dropping/shuffling caption"
        + " / captionのセカンダリ区切り文字。この区切り文字はcaptionのドロップやシャッフル後にcaption_separatorに置き換えられる",
    )
    add(
        "--enable_wildcard",
        action="store_true",
        help="enable wildcard for caption (e.g. '{image|picture|rendition}') / captionのワイルドカードを有効にする（例：'{image|picture|rendition}'）",
    )
    add("--caption_prefix", type=str, default=None, help="prefix for caption text / captionのテキストの先頭に付ける文字列")
    add("--caption_suffix", type=str, default=None, help="suffix for caption text / captionのテキストの末尾に付ける文字列")


def add_dataset_augmentation_arguments(parser: argparse.ArgumentParser) -> None:
    add = build_add(parser)
    add("--color_aug", action="store_true", help="enable weak color augmentation / 学習時に色合いのaugmentationを有効にする")
    add("--flip_aug", action="store_true", help="enable horizontal flip augmentation / 学習時に左右反転のaugmentationを有効にする")
    add(
        "--face_crop_aug_range",
        type=str,
        default=None,
        help="enable face-centered crop augmentation and its range (e.g. 2.0,4.0) / 学習時に顔を中心とした切り出しaugmentationを有効にするときは倍率を指定する（例：2.0,4.0）",
    )
    add(
        "--random_crop",
        action="store_true",
        help="enable random crop (for style training in face-centered crop augmentation) / ランダムな切り出しを有効にする（顔を中心としたaugmentationを行うときに画風の学習用に指定する）",
    )
    add(
        "--debug_dataset",
        action="store_true",
        help="show images for debugging (do not train) / デバッグ用に学習データを画面表示する（学習は行わない）",
    )


def add_dataset_resolution_and_cache_arguments(parser: argparse.ArgumentParser) -> None:
    add = build_add(parser)
    add(
        "--resolution",
        type=str,
        default=None,
        help="resolution in training ('size' or 'width,height') / 学習時の画像解像度（'サイズ'指定、または'幅,高さ'指定）",
    )
    add(
        "--cache_latents",
        action="store_true",
        help="cache latents to main memory to reduce VRAM usage (augmentations must be disabled) / VRAM削減のためにlatentをメインメモリにcacheする（augmentationは使用不可） ",
    )
    add("--vae_batch_size", type=int, default=1, help="batch size for caching latents / latentのcache時のバッチサイズ")
    add(
        "--cache_latents_to_disk",
        action="store_true",
        help="cache latents to disk to reduce VRAM usage (augmentations must be disabled) / VRAM削減のためにlatentをディスクにcacheする（augmentationは使用不可）",
    )
    add(
        "--skip_cache_check",
        action="store_true",
        help="skip the content validation of cache (latent and text encoder output). Cache file existence check is always performed, and cache processing is performed if the file does not exist"
        " / cacheの内容の検証をスキップする（latentとテキストエンコーダの出力）。キャッシュファイルの存在確認は常に行われ、ファイルがなければキャッシュ処理が行われる",
    )


def add_bucket_arguments(parser: argparse.ArgumentParser) -> None:
    add = build_add(parser)
    add("--enable_bucket", action="store_true", help="enable buckets for multi aspect ratio training / 複数解像度学習のためのbucketを有効にする")
    add(
        "--min_bucket_reso",
        type=int,
        default=256,
        help="minimum resolution for buckets, must be divisible by bucket_reso_steps / bucketの最小解像度、bucket_reso_stepsで割り切れる必要があります",
    )
    add(
        "--max_bucket_reso",
        type=int,
        default=1024,
        help="maximum resolution for buckets, must be divisible by bucket_reso_steps / bucketの最大解像度、bucket_reso_stepsで割り切れる必要があります",
    )
    add(
        "--bucket_reso_steps",
        type=int,
        default=64,
        help="steps of resolution for buckets, divisible by 8 is recommended / bucketの解像度の単位、8で割り切れる値を推奨します",
    )
    add("--bucket_no_upscale", action="store_true", help="make bucket for each image without upscaling / 画像を拡大せずbucketを作成します")
    add(
        "--resize_interpolation",
        type=str,
        default=None,
        choices=["lanczos", "nearest", "bilinear", "linear", "bicubic", "cubic", "area"],
        help="Resize interpolation when required. Default: area Options: lanczos, nearest, bilinear, bicubic, area / 必要に応じてサイズ補間を変更します。デフォルト: area オプション: lanczos, nearest, bilinear, bicubic, area",
    )
    add(
        "--token_warmup_min",
        type=int,
        default=1,
        help="start learning at N tags (token means comma separated strinfloatgs) / タグ数をN個から増やしながら学習する",
    )
    add(
        "--token_warmup_step",
        type=float,
        default=0,
        help="tag length reaches maximum on N steps (or N*max_train_steps if N<1) / N（N<1ならN*max_train_steps）ステップでタグ長が最大になる。デフォルトは0（最初から最大）",
    )
    add("--alpha_mask", action="store_true", help="use alpha channel as mask for training / 画像のアルファチャンネルをlossのマスクに使用する")


def add_dataset_class_arguments(parser: argparse.ArgumentParser) -> None:
    add = build_add(parser)
    add(
        "--dataset_class",
        type=str,
        default=None,
        help="dataset class for arbitrary dataset (package.module.Class) / 任意のデータセットを用いるときのクラス名 (package.module.Class)",
    )


def add_caption_dropout_arguments(parser: argparse.ArgumentParser) -> None:
    add = build_add(parser)
    add("--caption_dropout_rate", type=float, default=0.0, help="Rate out dropout caption(0.0~1.0) / captionをdropoutする割合")
    add(
        "--caption_dropout_every_n_epochs",
        type=int,
        default=0,
        help="Dropout all captions every N epochs / captionを指定エポックごとにdropoutする",
    )
    add(
        "--caption_tag_dropout_rate",
        type=float,
        default=0.0,
        help="Rate out dropout comma separated tokens(0.0~1.0) / カンマ区切りのタグをdropoutする割合",
    )


def add_dreambooth_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    add = build_add(parser)
    add("--reg_data_dir", type=str, default=None, help="directory for regularization images / 正則化画像データのディレクトリ")


def add_caption_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    add = build_add(parser)
    add("--in_json", type=str, default=None, help="json metadata for dataset / データセットのmetadataのjsonファイル")
    add("--dataset_repeats", type=int, default=1, help="repeat dataset when training with captions / キャプションでの学習時にデータセットを繰り返す回数")


def add_dataset_arguments(
    parser: argparse.ArgumentParser,
    support_dreambooth: bool,
    support_caption: bool,
    support_caption_dropout: bool,
) -> None:
    add_dataset_path_and_caption_arguments(parser)
    add_dataset_augmentation_arguments(parser)
    add_dataset_resolution_and_cache_arguments(parser)
    add_bucket_arguments(parser)
    add_dataset_class_arguments(parser)

    if support_caption_dropout:
        add_caption_dropout_arguments(parser)

    if support_dreambooth:
        add_dreambooth_dataset_arguments(parser)

    if support_caption:
        add_caption_dataset_arguments(parser)
