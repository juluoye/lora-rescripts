import argparse
import typing
from typing import Any, List, Union, Optional
import random
import time

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.types import Number
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed
from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from library import deepspeed_utils, model_util, sai_model_spec, strategy_base, strategy_sd

import library.train_util as train_util
import library.train_loop_setup_util as train_loop_setup_util
import library.train_metadata_util as train_metadata_util
import library.train_network_batch_util as train_network_batch_util
import library.train_network_checkpoint_util as train_network_checkpoint_util
import library.train_network_prepare_util as train_network_prepare_util
import library.train_network_runtime_util as train_network_runtime_util
import library.train_network_sync_step_util as train_network_sync_step_util
import library.train_network_train_step_util as train_network_train_step_util
import library.train_network_validation_util as train_network_validation_util
import library.train_resume_util as train_resume_util
import library.train_runtime_support_util as train_runtime_support_util
import library.train_network_setup_util as train_network_setup_util
import library.config_util as config_util
import library.huggingface_util as huggingface_util
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    get_weighted_text_embeddings,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_masked_loss,
)
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)


class PreparedDatasetSetup(typing.NamedTuple):
    train_dataset_group: Any
    val_dataset_group: Any
    current_epoch: Any
    current_step: Any
    collator: Any


class PreparedModelInputs(typing.NamedTuple):
    text_encoding_strategy: Any
    text_encoders: List[torch.nn.Module]
    unet: torch.nn.Module


class PreparedRuntimeModels(typing.NamedTuple):
    accelerator: Accelerator
    is_main_process: bool
    weight_dtype: Any
    save_dtype: Any
    vae_dtype: Any
    model_version: Any
    text_encoder: Any
    vae: Any
    unet: Any
    text_encoders: List[torch.nn.Module]


class PreparedNetworkSetup(typing.NamedTuple):
    network: Any
    net_kwargs: dict[str, Any]
    train_unet: bool
    train_text_encoder: bool


class PreparedExecutionRuntime(typing.NamedTuple):
    network: Any
    optimizer: Any
    train_dataloader: Any
    val_dataloader: Any
    lr_scheduler: Any
    text_encoder: Any
    text_encoders: List[torch.nn.Module]
    unet: torch.nn.Module
    training_model: Any
    unet_weight_dtype: Any


class PreparedTrainingComponents(typing.NamedTuple):
    text_encoder_lr: Any
    optimizer_name: str
    optimizer_args: str
    optimizer: Any
    optimizer_train_fn: Any
    optimizer_eval_fn: Any
    lr_descriptions: Optional[List[str]]
    train_dataloader: Any
    val_dataloader: Any
    lr_scheduler: Any


class NetworkTrainer:
    def __init__(self):
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False

    # TODO 他のスクリプトと共通化する
    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
        optimizer=None,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        mean_grad_norm=None,
        mean_combined_norm=None,
    ):
        logs = {"loss/current": current_loss, "loss/average": avr_loss}

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/max_key_norm"] = maximum_norm
        if mean_norm is not None:
            logs["norm/avg_key_norm"] = mean_norm
        if mean_grad_norm is not None:
            logs["norm/avg_grad_norm"] = mean_grad_norm
        if mean_combined_norm is not None:
            logs["norm/avg_combined_norm"] = mean_combined_norm

        lrs = lr_scheduler.get_last_lr()
        if lr_descriptions is not None:
            for i, lr in enumerate(lrs):
                lr_desc = lr_descriptions[i]
                logs[f"lr/{lr_desc}"] = lr

                if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower():
                    # tracking d*lr value
                    logs[f"lr/d*lr/{lr_desc}"] = (
                        lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                    )
                if (
                    args.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower()) and optimizer is not None
                ):  # tracking d*lr value of unet.
                    logs["lr/d*lr"] = optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]
        else:
            if len(lrs) == 0:
                return logs

            idx = 0
            if not args.network_train_unet_only:
                logs["lr/textencoder"] = float(lrs[0])
                idx = 1

            for i in range(idx, len(lrs)):
                logs[f"lr/group{i}"] = float(lrs[i])
                if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower():
                    logs[f"lr/d*lr/group{i}"] = (
                        lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                    )
                if args.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower()) and optimizer is not None:
                    logs[f"lr/d*lr/group{i}"] = optimizer.param_groups[i]["d"] * optimizer.param_groups[i]["lr"]

        return logs

    def step_logging(self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int):
        self.accelerator_logging(accelerator, logs, global_step, global_step, epoch)

    def epoch_logging(self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int):
        self.accelerator_logging(accelerator, logs, epoch, global_step, epoch)

    def val_logging(self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int, val_step: int):
        self.accelerator_logging(accelerator, logs, global_step + val_step, global_step, epoch, val_step)

    def accelerator_logging(
        self, accelerator: Accelerator, logs: dict, step_value: int, global_step: int, epoch: int, val_step: Optional[int] = None
    ):
        """
        step_value is for tensorboard, other values are for wandb
        """
        tensorboard_tracker = None
        wandb_tracker = None
        other_trackers = []
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                tensorboard_tracker = accelerator.get_tracker("tensorboard")
            elif tracker.name == "wandb":
                wandb_tracker = accelerator.get_tracker("wandb")
            else:
                other_trackers.append(accelerator.get_tracker(tracker.name))

        if tensorboard_tracker is not None:
            tensorboard_tracker.log(logs, step=step_value)

        if wandb_tracker is not None:
            logs["global_step"] = global_step
            logs["epoch"] = epoch
            if val_step is not None:
                logs["val_step"] = val_step
            wandb_tracker.log(logs)

        for tracker in other_trackers:
            tracker.log(logs, step=step_value)

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        train_dataset_group.verify_bucket_reso_steps(64)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(64)

    def load_target_model(self, args, weight_dtype, accelerator) -> tuple[str, nn.Module, nn.Module, Optional[nn.Module]]:
        text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)

        # モデルに xformers とか memory efficient attention を組み込む
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

        return model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization), text_encoder, vae, unet

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders) -> tuple[nn.Module, List[nn.Module]]:
        raise NotImplementedError()

    def get_tokenize_strategy(self, args):
        return strategy_sd.SdTokenizeStrategy(args.v2, args.max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_sd.SdTokenizeStrategy) -> List[Any]:
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, args):
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            True, args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        return latents_caching_strategy

    def get_text_encoding_strategy(self, args):
        return strategy_sd.SdTextEncodingStrategy(args.clip_skip)

    def get_text_encoder_outputs_caching_strategy(self, args):
        return None

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        """
        Returns a list of models that will be used for text encoding. SDXL uses wrapped and unwrapped models.
        FLUX.1 and SD3 may cache some outputs of the text encoder, so return the models that will be used for encoding (not cached).
        """
        return text_encoders

    # returns a list of bool values indicating whether each text encoder should be trained
    def get_text_encoders_train_flags(self, args, text_encoders):
        return [True] * len(text_encoders) if self.is_train_text_encoder(args) else [False] * len(text_encoders)

    def is_train_text_encoder(self, args):
        return not args.network_train_unet_only

    def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, text_encoders, dataset, weight_dtype):
        for t_enc in text_encoders:
            t_enc.to(accelerator.device, dtype=weight_dtype)

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype, **kwargs):
        noise_pred = unet(noisy_latents, timesteps, text_conds[0]).sample
        return noise_pred

    def all_reduce_network(self, accelerator, network):
        for param in network.parameters():
            if param.grad is not None:
                param.grad = accelerator.reduce(param.grad, reduction="mean")

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizers, text_encoder, unet):
        train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizers[0], text_encoder, unet)

    # region SD/SDXL

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        pass

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
        )
        prepare_scheduler_for_custom_training(noise_scheduler, device)
        if args.zero_terminal_snr:
            custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)
        return noise_scheduler

    def encode_images_to_latents(self, args, vae: AutoencoderKL, images: torch.FloatTensor) -> torch.FloatTensor:
        return vae.encode(images).latent_dist.sample()

    def shift_scale_latents(self, args, latents: torch.FloatTensor) -> torch.FloatTensor:
        return latents * self.vae_scale_factor

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        return train_network_batch_util.get_noise_pred_and_target(
            self,
            args,
            accelerator,
            noise_scheduler,
            latents,
            batch,
            text_encoder_conds,
            unet,
            network,
            weight_dtype,
            train_unet,
            is_train=is_train,
        )

    def post_process_loss(self, loss, args, timesteps: torch.IntTensor, noise_scheduler) -> torch.FloatTensor:
        return train_network_batch_util.post_process_loss(loss, args, timesteps, noise_scheduler)

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec(None, args, self.is_sdxl, True, False)

    def update_metadata(self, metadata, args):
        pass

    def is_text_encoder_not_needed_for_training(self, args):
        return False  # use for sample images

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        # set top parameter requires_grad = True for gradient checkpointing works
        text_encoder.text_model.embeddings.requires_grad_(True)

    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        text_encoder.text_model.embeddings.to(dtype=weight_dtype)

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        return accelerator.prepare(unet)

    def resolve_text_encoder_lr(self, args, network):
        return train_network_prepare_util.resolve_text_encoder_lr(args, network)

    def prepare_network_trainable_params(self, args, network):
        return train_network_prepare_util.prepare_network_trainable_params(args, network)

    def prepare_training_components(
        self,
        args,
        accelerator,
        network,
        train_dataset_group,
        val_dataset_group,
        collator,
    ) -> PreparedTrainingComponents:
        return train_network_prepare_util.prepare_training_components(
            args,
            accelerator,
            network,
            train_dataset_group,
            val_dataset_group,
            collator,
            PreparedTrainingComponents,
        )

    def prepare_dataset_setup(
        self,
        args,
        *,
        use_user_config: bool,
        use_dreambooth_method: bool,
        cache_latents: bool,
    ) -> Optional[PreparedDatasetSetup]:
        return train_network_prepare_util.prepare_dataset_setup(
            self,
            args,
            use_user_config=use_user_config,
            use_dreambooth_method=use_dreambooth_method,
            cache_latents=cache_latents,
            prepared_cls=PreparedDatasetSetup,
            logger=logger,
        )

    def prepare_cached_model_inputs(
        self,
        args,
        accelerator,
        unet,
        vae,
        text_encoders,
        train_dataset_group,
        val_dataset_group,
        weight_dtype,
        vae_dtype,
        cache_latents: bool,
    ) -> PreparedModelInputs:
        return train_network_prepare_util.prepare_cached_model_inputs(
            self,
            args,
            accelerator,
            unet,
            vae,
            text_encoders,
            train_dataset_group,
            val_dataset_group,
            weight_dtype,
            vae_dtype,
            cache_latents=cache_latents,
            prepared_cls=PreparedModelInputs,
        )

    def prepare_runtime_models(self, args) -> PreparedRuntimeModels:
        return train_network_prepare_util.prepare_runtime_models(self, args, PreparedRuntimeModels, logger)

    def import_network_module(self, args, accelerator):
        return train_network_setup_util.import_network_module(args, accelerator)

    def merge_base_network_weights(self, args, accelerator, network_module, vae, text_encoder, unet, weight_dtype) -> None:
        train_network_setup_util.merge_base_network_weights(
            args, accelerator, network_module, vae, text_encoder, unet, weight_dtype
        )

    def parse_network_kwargs(self, args) -> dict[str, Any]:
        return train_network_setup_util.parse_network_kwargs(args)

    def configure_network_gradient_checkpointing(self, args, accelerator, network, text_encoders, unet) -> None:
        train_network_setup_util.configure_network_gradient_checkpointing(
            self, args, accelerator, network, text_encoders, unet
        )

    def prepare_network_setup(
        self,
        args,
        accelerator,
        vae,
        text_encoder,
        unet,
        text_encoders,
        weight_dtype,
    ) -> PreparedNetworkSetup:
        return train_network_setup_util.prepare_network_setup(
            self,
            args,
            accelerator,
            vae,
            text_encoder,
            unet,
            text_encoders,
            weight_dtype,
            PreparedNetworkSetup,
            logger,
        )

    def prepare_execution_runtime(
        self,
        args,
        accelerator,
        network,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
        text_encoder,
        text_encoders,
        unet,
        weight_dtype,
        vae,
        vae_dtype,
        cache_latents: bool,
        train_unet: bool,
        train_text_encoder: bool,
    ) -> PreparedExecutionRuntime:
        return train_network_setup_util.prepare_execution_runtime(
            self,
            args,
            accelerator,
            network,
            optimizer,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
            text_encoder,
            text_encoders,
            unet,
            weight_dtype,
            vae,
            vae_dtype,
            cache_latents=cache_latents,
            train_unet=train_unet,
            train_text_encoder=train_text_encoder,
            prepared_cls=PreparedExecutionRuntime,
            logger=logger,
        )

    def on_step_start(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train: bool = True):
        pass

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        pass

    # endregion

    def process_batch(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy: strategy_base.TextEncodingStrategy,
        tokenize_strategy: strategy_base.TokenizeStrategy,
        is_train=True,
        train_text_encoder=True,
        train_unet=True,
    ) -> torch.Tensor:
        return train_network_batch_util.process_batch(
            self,
            batch,
            text_encoders,
            unet,
            network,
            vae,
            noise_scheduler,
            vae_dtype,
            weight_dtype,
            accelerator,
            args,
            text_encoding_strategy,
            tokenize_strategy,
            is_train=is_train,
            train_text_encoder=train_text_encoder,
            train_unet=train_unet,
        )

    def cast_text_encoder(self, args):
        return True  # default for other than HunyuanImage

    def cast_vae(self, args):
        return True  # default for other than HunyuanImage

    def cast_unet(self, args):
        return True  # default for other than HunyuanImage

    def train(self, args):
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        train_util.verify_training_args(args)
        train_util.prepare_dataset_args(args, True)
        deepspeed_utils.prepare_deepspeed_args(args)
        setup_logging(args, reset=True)

        cache_latents = args.cache_latents
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        tokenize_strategy = self.get_tokenize_strategy(args)
        strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
        tokenizers = self.get_tokenizers(tokenize_strategy)  # will be removed after sample_image is refactored

        # prepare caching strategy: this must be set before preparing dataset. because dataset may use this strategy for initialization.
        latents_caching_strategy = self.get_latents_caching_strategy(args)
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

        prepared_dataset = self.prepare_dataset_setup(
            args,
            use_user_config=use_user_config,
            use_dreambooth_method=use_dreambooth_method,
            cache_latents=cache_latents,
        )
        if prepared_dataset is None:
            return

        train_dataset_group = prepared_dataset.train_dataset_group
        val_dataset_group = prepared_dataset.val_dataset_group
        current_epoch = prepared_dataset.current_epoch
        current_step = prepared_dataset.current_step
        collator = prepared_dataset.collator

        prepared_runtime_models = self.prepare_runtime_models(args)
        accelerator = prepared_runtime_models.accelerator
        is_main_process = prepared_runtime_models.is_main_process
        weight_dtype = prepared_runtime_models.weight_dtype
        save_dtype = prepared_runtime_models.save_dtype
        vae_dtype = prepared_runtime_models.vae_dtype
        model_version = prepared_runtime_models.model_version
        text_encoder = prepared_runtime_models.text_encoder
        vae = prepared_runtime_models.vae
        unet = prepared_runtime_models.unet
        text_encoders = prepared_runtime_models.text_encoders

        prepared_model_inputs = self.prepare_cached_model_inputs(
            args,
            accelerator,
            unet,
            vae,
            text_encoders,
            train_dataset_group,
            val_dataset_group,
            weight_dtype,
            vae_dtype,
            cache_latents,
        )
        text_encoding_strategy = prepared_model_inputs.text_encoding_strategy
        text_encoders = prepared_model_inputs.text_encoders
        unet = prepared_model_inputs.unet

        prepared_network_setup = self.prepare_network_setup(
            args,
            accelerator,
            vae,
            text_encoder,
            unet,
            text_encoders,
            weight_dtype,
        )
        network = prepared_network_setup.network
        net_kwargs = prepared_network_setup.net_kwargs
        train_unet = prepared_network_setup.train_unet
        train_text_encoder = prepared_network_setup.train_text_encoder

        # 学習に必要なクラスを準備する
        accelerator.print("prepare optimizer, data loader etc.")
        prepared_training = self.prepare_training_components(
            args,
            accelerator,
            network,
            train_dataset_group,
            val_dataset_group,
            collator,
        )
        text_encoder_lr = prepared_training.text_encoder_lr
        optimizer_name = prepared_training.optimizer_name
        optimizer_args = prepared_training.optimizer_args
        optimizer = prepared_training.optimizer
        optimizer_train_fn = prepared_training.optimizer_train_fn
        optimizer_eval_fn = prepared_training.optimizer_eval_fn
        lr_descriptions = prepared_training.lr_descriptions
        train_dataloader = prepared_training.train_dataloader
        val_dataloader = prepared_training.val_dataloader
        lr_scheduler = prepared_training.lr_scheduler

        prepared_execution_runtime = self.prepare_execution_runtime(
            args,
            accelerator,
            network,
            optimizer,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
            text_encoder,
            text_encoders,
            unet,
            weight_dtype,
            vae,
            vae_dtype,
            cache_latents,
            train_unet,
            train_text_encoder,
        )
        network = prepared_execution_runtime.network
        optimizer = prepared_execution_runtime.optimizer
        train_dataloader = prepared_execution_runtime.train_dataloader
        val_dataloader = prepared_execution_runtime.val_dataloader
        lr_scheduler = prepared_execution_runtime.lr_scheduler
        text_encoder = prepared_execution_runtime.text_encoder
        text_encoders = prepared_execution_runtime.text_encoders
        unet = prepared_execution_runtime.unet
        training_model = prepared_execution_runtime.training_model
        unet_weight_dtype = prepared_execution_runtime.unet_weight_dtype

        loaded_training_state = train_resume_util.register_network_state_hooks(
            accelerator,
            args,
            network,
            current_epoch,
            current_step,
            logger,
        )

        # resumeする
        train_util.resume_from_local_or_hf_if_specified(accelerator, args)

        training_schedule = train_loop_setup_util.prepare_training_schedule(
            args,
            train_dataloader_len=len(train_dataloader),
            accelerator_num_processes=accelerator.num_processes,
        )
        num_train_epochs = training_schedule.num_train_epochs
        total_batch_size = training_schedule.total_batch_size
        train_loop_setup_util.log_training_start_summary(
            accelerator,
            args,
            train_dataset_group,
            val_dataset_group,
            train_dataloader_len=len(train_dataloader),
            displayed_num_train_epochs=num_train_epochs,
        )

        metadata_bundle = train_metadata_util.build_metadata_bundle(
            args,
            session_id=session_id,
            training_started_at=training_started_at,
            text_encoder_lr=text_encoder_lr,
            train_dataset_group=train_dataset_group,
            val_dataset_group=val_dataset_group,
            train_dataloader=train_dataloader,
            num_train_epochs=num_train_epochs,
            model_version=model_version,
            optimizer_name=optimizer_name,
            optimizer_args=optimizer_args,
            use_user_config=use_user_config,
            use_dreambooth_method=use_dreambooth_method,
            total_batch_size=total_batch_size,
            net_kwargs=net_kwargs,
            update_metadata=self.update_metadata,
        )
        metadata = metadata_bundle.metadata
        minimum_metadata = metadata_bundle.minimum_metadata

        initial_training_plan = train_resume_util.resolve_initial_training_plan(
            args,
            logger,
            train_dataloader_len=len(train_dataloader),
            accelerator_num_processes=accelerator.num_processes,
            steps_from_state=loaded_training_state.steps_from_state,
            track_initial_progress=False,
        )
        initial_step = initial_training_plan.initial_step
        epoch_to_start = initial_training_plan.epoch_to_start

        global_step = 0

        noise_scheduler = self.get_noise_scheduler(args, accelerator.device)

        train_util.init_trackers(accelerator, args, "network_train")

        loss_recorder = train_util.LossRecorder()
        val_step_loss_recorder = train_util.LossRecorder()
        val_epoch_loss_recorder = train_util.LossRecorder()

        del train_dataset_group
        if val_dataset_group is not None:
            del val_dataset_group

        on_step_start_for_network = train_network_runtime_util.resolve_on_step_start_callback(accelerator, network)
        save_model, remove_model = train_network_runtime_util.make_checkpoint_handlers(
            args,
            accelerator,
            metadata,
            minimum_metadata,
            save_dtype,
            self.get_sai_model_spec,
            huggingface_util.upload,
        )

        # if text_encoder is not needed for training, delete it to save memory.
        # TODO this can be automated after SDXL sample prompt cache is implemented
        text_encoders, text_encoder = train_network_runtime_util.drop_unused_text_encoders_if_needed(
            self, args, accelerator, logger, text_encoders, text_encoder
        )

        is_tracking = train_runtime_support_util.run_initial_sampling(
            accelerator,
            optimizer_eval_fn,
            optimizer_train_fn,
            lambda: self.sample_images(accelerator, args, 0, global_step, accelerator.device, vae, tokenizers, text_encoder, unet),
        )

        loop_progress = train_loop_setup_util.prepare_loop_progress_state(
            logger,
            args,
            initial_step=initial_step,
            epoch_to_start=epoch_to_start,
            train_dataloader_len=len(train_dataloader),
        )
        initial_step = loop_progress.initial_step
        global_step = loop_progress.global_step

        train_network_runtime_util.log_runtime_model_state(logger, unet_weight_dtype, unet, text_encoders)

        clean_memory_on_device(accelerator.device)

        progress_bar = tqdm(
            range(loop_progress.progress_total), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps"
        )

        validation_runtime = train_network_runtime_util.prepare_validation_runtime(args, noise_scheduler, val_dataloader)
        validation_steps = validation_runtime.validation_steps
        validation_timesteps = validation_runtime.validation_timesteps
        validation_total_steps = validation_runtime.validation_total_steps
        original_args_min_timestep = validation_runtime.original_args_min_timestep
        original_args_max_timestep = validation_runtime.original_args_max_timestep

        for epoch in range(epoch_to_start, num_train_epochs):
            accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}\n")
            current_epoch.value = epoch + 1

            metadata["ss_epoch"] = str(epoch + 1)

            accelerator.unwrap_model(network).on_epoch_start(text_encoder, unet)  # network.train() is called here

            # TRAINING
            skipped_dataloader = None
            if initial_step > 0:
                skipped_dataloader = accelerator.skip_first_batches(train_dataloader, initial_step - 1)
                initial_step = 1

            for step, batch in enumerate(skipped_dataloader or train_dataloader):
                current_step.value = global_step
                if initial_step > 0:
                    initial_step -= 1
                    continue

                current_loss = train_network_train_step_util.execute_train_step(
                    self,
                    args=args,
                    batch=batch,
                    accelerator=accelerator,
                    training_model=training_model,
                    on_step_start_for_network=on_step_start_for_network,
                    text_encoder=text_encoder,
                    unet=unet,
                    text_encoders=text_encoders,
                    network=network,
                    vae=vae,
                    noise_scheduler=noise_scheduler,
                    vae_dtype=vae_dtype,
                    weight_dtype=weight_dtype,
                    text_encoding_strategy=text_encoding_strategy,
                    tokenize_strategy=tokenize_strategy,
                    train_text_encoder=train_text_encoder,
                    train_unet=train_unet,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                )

                norm_metrics = train_network_sync_step_util.collect_network_norm_metrics(args, accelerator, network)
                keys_scaled = norm_metrics.keys_scaled
                mean_norm = norm_metrics.mean_norm
                maximum_norm = norm_metrics.maximum_norm
                mean_grad_norm = norm_metrics.mean_grad_norm
                mean_combined_norm = norm_metrics.mean_combined_norm
                max_mean_logs = norm_metrics.max_mean_logs

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    global_step = train_network_sync_step_util.handle_sync_step_completion(
                        self,
                        args=args,
                        accelerator=accelerator,
                        network=network,
                        global_step=global_step,
                        epoch=epoch,
                        progress_bar=progress_bar,
                        optimizer_eval_fn=optimizer_eval_fn,
                        optimizer_train_fn=optimizer_train_fn,
                        vae=vae,
                        tokenizers=tokenizers,
                        text_encoder=text_encoder,
                        unet=unet,
                        save_model=save_model,
                        remove_model=remove_model,
                    )

                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                avr_loss: float = loss_recorder.moving_average
                logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**{**max_mean_logs, **logs})

                if is_tracking:
                    logs = self.generate_step_logs(
                        args,
                        current_loss,
                        avr_loss,
                        lr_scheduler,
                        lr_descriptions,
                        optimizer,
                        keys_scaled,
                        mean_norm,
                        maximum_norm,
                        mean_grad_norm,
                        mean_combined_norm,
                    )
                    self.step_logging(accelerator, logs, global_step, epoch + 1)

                # VALIDATION PER STEP: global_step is already incremented
                # for example, if validate_every_n_steps=100, validate at step 100, 200, 300, ...
                should_validate_step = args.validate_every_n_steps is not None and global_step % args.validate_every_n_steps == 0
                if accelerator.sync_gradients and validation_steps > 0 and should_validate_step:
                    train_network_validation_util.run_validation_pass(
                        self,
                        args=args,
                        accelerator=accelerator,
                        network=network,
                        text_encoders=text_encoders,
                        unet=unet,
                        vae=vae,
                        noise_scheduler=noise_scheduler,
                        vae_dtype=vae_dtype,
                        weight_dtype=weight_dtype,
                        text_encoding_strategy=text_encoding_strategy,
                        tokenize_strategy=tokenize_strategy,
                        train_text_encoder=train_text_encoder,
                        train_unet=train_unet,
                        optimizer_eval_fn=optimizer_eval_fn,
                        optimizer_train_fn=optimizer_train_fn,
                        val_dataloader=val_dataloader,
                        validation_steps=validation_steps,
                        validation_timesteps=validation_timesteps,
                        validation_total_steps=validation_total_steps,
                        validation_loss_recorder=val_step_loss_recorder,
                        training_loss_recorder=loss_recorder,
                        is_tracking=is_tracking,
                        global_step=global_step,
                        recorder_epoch=epoch,
                        epoch_display=epoch + 1,
                        progress_bar=progress_bar,
                        progress_desc="validation steps",
                        progress_postfix_key="val_avg_loss",
                        log_callback=lambda logs: self.step_logging(accelerator, logs, global_step, epoch=epoch + 1),
                        log_prefix="step",
                        original_args_min_timestep=original_args_min_timestep,
                        original_args_max_timestep=original_args_max_timestep,
                    )

                if global_step >= args.max_train_steps:
                    break

            # EPOCH VALIDATION
            should_validate_epoch = (
                (epoch + 1) % args.validate_every_n_epochs == 0 if args.validate_every_n_epochs is not None else True
            )

            if should_validate_epoch and len(val_dataloader) > 0:
                train_network_validation_util.run_validation_pass(
                    self,
                    args=args,
                    accelerator=accelerator,
                    network=network,
                    text_encoders=text_encoders,
                    unet=unet,
                    vae=vae,
                    noise_scheduler=noise_scheduler,
                    vae_dtype=vae_dtype,
                    weight_dtype=weight_dtype,
                    text_encoding_strategy=text_encoding_strategy,
                    tokenize_strategy=tokenize_strategy,
                    train_text_encoder=train_text_encoder,
                    train_unet=train_unet,
                    optimizer_eval_fn=optimizer_eval_fn,
                    optimizer_train_fn=optimizer_train_fn,
                    val_dataloader=val_dataloader,
                    validation_steps=validation_steps,
                    validation_timesteps=validation_timesteps,
                    validation_total_steps=validation_total_steps,
                    validation_loss_recorder=val_epoch_loss_recorder,
                    training_loss_recorder=loss_recorder,
                    is_tracking=is_tracking,
                    global_step=global_step,
                    recorder_epoch=epoch,
                    epoch_display=epoch + 1,
                    progress_bar=progress_bar,
                    progress_desc="epoch validation steps",
                    progress_postfix_key="val_epoch_avg_loss",
                    log_callback=lambda logs: self.epoch_logging(accelerator, logs, global_step, epoch + 1),
                    log_prefix="epoch",
                    original_args_min_timestep=original_args_min_timestep,
                    original_args_max_timestep=original_args_max_timestep,
                )

            # END OF EPOCH
            epoch_logs = {"loss/epoch_average": loss_recorder.moving_average}
            train_network_checkpoint_util.handle_epoch_end(
                args=args,
                accelerator=accelerator,
                is_tracking=is_tracking,
                epoch_logs=epoch_logs,
                epoch_log_callback=lambda logs: self.epoch_logging(accelerator, logs, global_step, epoch + 1),
                optimizer_eval_fn=optimizer_eval_fn,
                optimizer_train_fn=optimizer_train_fn,
                is_main_process=is_main_process,
                network=network,
                global_step=global_step,
                epoch_no=epoch + 1,
                total_epochs=num_train_epochs,
                save_model=save_model,
                remove_model=remove_model,
                progress_bar=progress_bar,
                sample_images_fn=lambda epoch_no, step: self.sample_images(
                    accelerator, args, epoch_no, step, accelerator.device, vae, tokenizers, text_encoder, unet
                ),
            )

            # end of epoch

        # metadata["ss_epoch"] = str(num_train_epochs)
        metadata["ss_training_finished_at"] = str(time.time())

        network = train_network_checkpoint_util.finalize_training(
            args,
            accelerator,
            is_main_process,
            network,
            global_step,
            num_train_epochs,
            save_model,
            optimizer_eval_fn,
            logger,
        )


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    sai_model_spec.add_model_spec_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument(
        "--cpu_offload_checkpointing",
        action="store_true",
        help="[EXPERIMENTAL] enable offloading of tensors to CPU during checkpointing for U-Net or DiT, if supported"
        " / 勾配チェックポイント時にテンソルをCPUにオフロードする（U-NetまたはDiTのみ、サポートされている場合）",
    )
    parser.add_argument(
        "--no_metadata", action="store_true", help="do not save metadata in output model / メタデータを出力先モデルに保存しない"
    )
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）",
    )

    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=None,
        nargs="*",
        help="learning rate for Text Encoder, can be multiple / Text Encoderの学習率、複数指定可能",
    )
    parser.add_argument(
        "--fp8_base_unet",
        action="store_true",
        help="use fp8 for U-Net (or DiT), Text Encoder is fp16 or bf16"
        " / U-Net（またはDiT）にfp8を使用する。Text Encoderはfp16またはbf16",
    )

    parser.add_argument(
        "--network_weights", type=str, default=None, help="pretrained weights for network / 学習するネットワークの初期重み"
    )
    parser.add_argument(
        "--network_module", type=str, default=None, help="network module to train / 学習対象のネットワークのモジュール"
    )
    parser.add_argument(
        "--network_dim",
        type=int,
        default=None,
        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）",
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=1,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）",
    )
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons) / 訓練時に毎ステップでニューロンをdropする（0またはNoneはdropoutなし、1は全ニューロンをdropout）",
    )
    parser.add_argument(
        "--network_args",
        type=str,
        default=None,
        nargs="*",
        help="additional arguments for network (key=value) / ネットワークへの追加の引数",
    )
    parser.add_argument(
        "--network_train_unet_only", action="store_true", help="only training U-Net part / U-Net関連部分のみ学習する"
    )
    parser.add_argument(
        "--network_train_text_encoder_only",
        action="store_true",
        help="only training Text Encoder part / Text Encoder関連部分のみ学習する",
    )
    parser.add_argument(
        "--training_comment",
        type=str,
        default=None,
        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列",
    )
    parser.add_argument(
        "--dim_from_weights",
        action="store_true",
        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point) / 重みの値をスケーリングして勾配爆発を防ぐ（1が初期値としては適当）",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        nargs="*",
        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル",
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--skip_until_initial_step",
        action="store_true",
        help="skip training until initial_step is reached / initial_stepに到達するまで学習をスキップする",
    )
    parser.add_argument(
        "--initial_epoch",
        type=int,
        default=None,
        help="initial epoch number, 1 means first epoch (same as not specifying). NOTE: initial_epoch/step doesn't affect to lr scheduler. Which means lr scheduler will start from 0 without `--resume`."
        + " / 初期エポック数、1で最初のエポック（未指定時と同じ）。注意：initial_epoch/stepはlr schedulerに影響しないため、`--resume`しない場合はlr schedulerは0から始まる",
    )
    parser.add_argument(
        "--initial_step",
        type=int,
        default=None,
        help="initial step number including all epochs, 0 means first step (same as not specifying). overwrites initial_epoch."
        + " / 初期ステップ数、全エポックを含むステップ数、0で最初のステップ（未指定時と同じ）。initial_epochを上書きする",
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=None,
        help="Validation seed for shuffling validation dataset, training `--seed` used otherwise / 検証データセットをシャッフルするための検証シード、それ以外の場合はトレーニング `--seed` を使用する",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.0,
        help="Split for validation images out of the training dataset / 学習画像から検証画像に分割する割合",
    )
    parser.add_argument(
        "--validate_every_n_steps",
        type=int,
        default=None,
        help="Run validation on validation dataset every N steps. By default, validation will only occur every epoch if a validation dataset is available / 検証データセットの検証をNステップごとに実行します。デフォルトでは、検証データセットが利用可能な場合にのみ、検証はエポックごとに実行されます",
    )
    parser.add_argument(
        "--validate_every_n_epochs",
        type=int,
        default=None,
        help="Run validation dataset every N epochs. By default, validation will run every epoch if a validation dataset is available / 検証データセットをNエポックごとに実行します。デフォルトでは、検証データセットが利用可能な場合、検証はエポックごとに実行されます",
    )
    parser.add_argument(
        "--max_validation_steps",
        type=int,
        default=None,
        help="Max number of validation dataset items processed. By default, validation will run the entire validation dataset / 処理される検証データセット項目の最大数。デフォルトでは、検証は検証データセット全体を実行します",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = NetworkTrainer()
    trainer.train(args)
