Schema.intersect([
    Schema.intersect([
        Schema.object({
            model_train_type: Schema.union(["sd-lora", "sdxl-lora"]).default("sd-lora").description("训练种类"),
            pretrained_model_name_or_path: Schema.string().role('filepicker', { type: "model-file" }).default("./sd-models/model.safetensors").description("底模文件路径"),
            resume: Schema.string().role('filepicker', { type: "folder" }).description("从某个 `save_state` 保存的中断状态继续训练，填写文件路径"),
            vae: Schema.string().role('filepicker', { type: "model-file" }).description("(可选) VAE 模型文件路径，使用外置 VAE 文件覆盖模型内本身的"),
        }).description("训练用模型"),

        Schema.union([
            Schema.object({
                model_train_type: Schema.const("sd-lora"),
                v2: Schema.boolean().default(false).description("底模为 sd2.0 以后的版本需要启用"),
            }),
            Schema.object({}),
        ]),

        Schema.union([
            Schema.object({
                model_train_type: Schema.const("sd-lora"),
                v2: Schema.const(true).required(),
                v_parameterization: Schema.boolean().default(false).description("v-parameterization 学习"),
            }),
            Schema.object({}),
        ]),

        // SDXL v预测模型训练选项
        Schema.union([
            Schema.object({
                model_train_type: Schema.const("sdxl-lora"),
                v_parameterization: Schema.boolean().default(false).description("v-parameterization 学习（训练Illustrious等v-pred模型时需要开启）"),
            }),
            Schema.object({}),
        ]),

        Schema.union([
            Schema.object({
                model_train_type: Schema.const("sdxl-lora"),
                v_parameterization: Schema.const(true).required(),
                zero_terminal_snr: Schema.boolean().default(true).description("Zero Terminal SNR（v-pred模型训练推荐开启）"),
            }),
            Schema.object({}),
        ]),

        Schema.union([
            Schema.object({
                v_parameterization: Schema.const(true).required(),
                scale_v_pred_loss_like_noise_pred: Schema.boolean().default(true).description("缩放 v-prediction 损失（v-pred模型训练推荐开启）"),
            }),
            Schema.object({}),
        ]),
    ]),

    // 数据集设置
    Schema.object(SHARED_SCHEMAS.RAW.DATASET_SETTINGS).description("数据集设置"),

    // 保存设置
    SHARED_SCHEMAS.SAVE_SETTINGS,

    Schema.object({
        max_train_epochs: Schema.number().min(1).default(10).description("最大训练 epoch（轮数）"),
        train_batch_size: Schema.number().min(1).default(1).description("批量大小。单卡/单进程时就是实际 batch；多卡/分布式时按全局 batch 解释，启动时会自动换算成每卡。数值越高显存占用越高。"),
        gradient_checkpointing: Schema.boolean().default(false).description("梯度检查点"),
        gradient_accumulation_steps: Schema.number().min(1).description("梯度累加步数"),
        network_train_unet_only: Schema.boolean().default(false).description("仅训练 U-Net 训练SDXL Lora时推荐开启"),
        network_train_text_encoder_only: Schema.boolean().default(false).description("仅训练文本编码器"),
    }).description("训练相关参数"),

    Schema.union([
        Schema.intersect([
            Schema.object({
                model_train_type: Schema.const("sdxl-lora").required(),
                enable_mixed_resolution_training: Schema.boolean().default(false).description("启用阶段分辨率训练（实验性，仅支持 SDXL）。1024 基准使用 512/768/1024；2048 基准使用 1024/1536/2048"),
            }).description("阶段分辨率训练"),
            Schema.union([
                Schema.object({
                    model_train_type: Schema.const("sdxl-lora").required(),
                    enable_mixed_resolution_training: Schema.const(true).required(),
                    staged_resolution_ratio_512: Schema.number().min(0).max(100).step(1).default(20).description("512 阶段占比（百分比）。当最终分辨率最大边小于 512 时会忽略"),
                    staged_resolution_ratio_768: Schema.number().min(0).max(100).step(1).default(30).description("768 阶段占比（百分比）。当最终分辨率最大边小于 768 时会忽略"),
                    staged_resolution_ratio_1024: Schema.number().min(0).max(100).step(1).default(50).description("1024 阶段占比（百分比）。1024 基准和 2048 基准都会用到"),
                    staged_resolution_ratio_1536: Schema.number().min(0).max(100).step(1).default(30).description("1536 阶段占比（百分比）。仅 2048 基准会用到"),
                    staged_resolution_ratio_2048: Schema.number().min(0).max(100).step(1).default(50).description("2048 阶段占比（百分比）。仅 2048 基准会用到"),
                }),
                Schema.object({}),
            ]),
        ]),
        Schema.object({}),
    ]),

    // 学习率&优化器设置
    SHARED_SCHEMAS.LR_OPTIMIZER,

    Schema.intersect([
        Schema.object({
            network_module: Schema.union(["networks.lora", "networks.dylora", "networks.oft", "lycoris.kohya"]).default("networks.lora").description("训练网络模块"),
            network_weights: Schema.string().role('filepicker').description("从已有的 LoRA 模型上继续训练，填写路径"),
            network_dim: Schema.number().min(1).default(32).description("网络维度，常用 4~128，不是越大越好, 低dim可以降低显存占用"),
            network_alpha: Schema.number().min(1).default(32).description("常用值：等于 network_dim 或 network_dim*1/2 或 1。使用较小的 alpha 需要提升学习率"),
            network_dropout: Schema.number().min(0).max(1).step(0.01).default(0).description('dropout 概率 （与 lycoris 不兼容，需要用 lycoris 自带的）'),
            pissa_init: Schema.boolean().default(false).description("启用 PiSSA 初始化（实验性，仅在 network_module=networks.lora 时生效）"),
            dim_from_weights: Schema.boolean().default(false).description("从已有 network_weights 自动推断 rank / dim"),
            scale_weight_norms: Schema.number().step(0.01).min(0).description("最大范数正则化。如果使用，推荐为 1"),
            dora_wd: Schema.boolean().default(false).description('启用 DoRA 训练'),
            vram_swap_to_ram: Schema.boolean().default(false).description("实验性显存兜底项：让原生 LoRA / LoRA-FA / T-LoRA / VeRA 适配器权重常驻 CPU RAM，并在前向时按需拉回训练设备。通常会更慢；暂不支持 DeepSpeed、多进程、full_fp16/full_bf16 与部分 8bit/paged 优化器"),
            network_args_custom: Schema.array(String).role('table').description('自定义 network_args，一行一个'),
            enable_block_weights: Schema.boolean().default(false).description('启用分层学习率训练（只支持网络模块 networks.lora）'),
            enable_base_weight: Schema.boolean().default(false).description('启用基础权重（差异炼丹）'),
        }).description("网络设置"),

        Schema.union([
            Schema.object({
                network_module: Schema.const("networks.lora").required(),
                pissa_init: Schema.const(true).required(),
                pissa_method: Schema.union(["rsvd", "svd"]).default("rsvd").description("PiSSA 分解方式。推荐保持 rSVD 默认值"),
                pissa_niter: Schema.number().min(0).step(1).default(2).description("PiSSA rSVD 幂迭代次数（高级参数，通常保持默认）"),
                pissa_oversample: Schema.number().min(0).step(1).default(8).description("PiSSA rSVD 过采样维度（高级参数，通常保持默认）"),
                pissa_apply_conv2d: Schema.boolean().default(false).description("PiSSA 额外作用于 1x1 Conv（实验性，默认只初始化 Linear）"),
                pissa_export_mode: Schema.union(["LoRA无损兼容导出", "LoRA快速近似导出"]).default("LoRA无损兼容导出").description("PiSSA 模型保存为标准 LoRA 时的导出方式"),
            }),
            Schema.object({}),
        ]),

        // lycoris 参数
        SHARED_SCHEMAS.LYCORIS_MAIN,
        SHARED_SCHEMAS.LYCORIS_LOKR,

        // dylora 参数
        SHARED_SCHEMAS.NETWORK_OPTION_DYLORA,

        // 分层学习率参数
        SHARED_SCHEMAS.NETWORK_OPTION_BLOCK_WEIGHTS,

        SHARED_SCHEMAS.NETWORK_OPTION_BASEWEIGHT,
    ]),

    // 预览图设置
    SHARED_SCHEMAS.PREVIEW_IMAGE,

    // 日志设置
    SHARED_SCHEMAS.LOG_SETTINGS,

    // 验证设置
    SHARED_SCHEMAS.VALIDATION_SETTINGS,

    // caption 选项
    Schema.object(SHARED_SCHEMAS.RAW.CAPTION_SETTINGS).description("caption（Tag）选项"),

    // 噪声设置
    SHARED_SCHEMAS.NOISE_SETTINGS,

    // 数据增强
    SHARED_SCHEMAS.DATA_ENCHANCEMENT,

    // 其他选项
    SHARED_SCHEMAS.OTHER,

    SHARED_SCHEMAS.THERMAL_MANAGEMENT,

    // 速度优化选项
    Schema.intersect([
        Schema.object(SHARED_SCHEMAS.RAW.PRECISION_CACHE_BATCH),
        Schema.union([
            Schema.object({
                model_train_type: Schema.const("sdxl-lora").required(),
                sageattn: Schema.boolean().default(false).description("启用 SageAttention（实验性，需要 SageAttention 专用环境）"),
                flashattn: Schema.boolean().default(false).description("启用 FlashAttention 2（实验性，需要 FlashAttention 运行时）"),
            }),
            Schema.object({}),
        ]),
    ]).description("速度优化选项"),

    // 分布式训练
    SHARED_SCHEMAS.DISTRIBUTED_TRAINING
]);
