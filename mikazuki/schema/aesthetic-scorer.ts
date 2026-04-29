Schema.intersect([
    Schema.object({
        model_train_type: Schema.string().default("aesthetic-scorer").disabled().description("训练种类"),
        output_name: Schema.string().default("aesthetic-scorer-best").description("模型保存名称"),
        output_dir: Schema.string().role("filepicker", { type: "folder" }).default("./output/aesthetic-scorer").description("模型输出目录"),
        save_model_as: Schema.union(["safetensors", "pt", "pth", "ckpt"]).default("safetensors").description("模型保存格式"),
    }).description("输出设置"),

    Schema.object({
        annotations: Schema.string().role("filepicker", { type: "model-file" }).default("./datasets/aesthetic/annotations.jsonl").description("标注文件路径，支持 `.jsonl`、`.csv`、`.db`"),
        image_root: Schema.string().role("filepicker", { type: "folder" }).description("图片根目录。留空时按标注文件中的路径直接解析"),
        train_split: Schema.string().description("训练 split 名称。标注内有 split 字段时可填写，例如 `train`"),
        val_split: Schema.string().description("验证 split 名称。标注内有 split 字段时可填写，例如 `val`"),
        val_ratio: Schema.number().min(0.01).max(0.99).step(0.01).default(0.1).description("未使用 split 字段时，按比例随机切分验证集"),
        target_dims: Schema.string().role("textarea").default("aesthetic\ncomposition\ncolor\nsexual").description("参与训练的评分维度，一行一个，可选 `aesthetic`、`composition`、`color`、`sexual`"),
    }).description("数据集设置"),

    Schema.object({
        batch_size: Schema.number().min(1).default(8).description("训练 batch size"),
        num_workers: Schema.number().min(0).default(4).description("DataLoader worker 数"),
        epochs: Schema.number().min(1).default(10).description("训练轮数"),
        learning_rate: Schema.string().default("3e-4").description("学习率"),
        weight_decay: Schema.string().default("1e-4").description("权重衰减"),
        loss: Schema.union(["mse", "smooth_l1"]).default("mse").description("回归损失函数"),
        cls_loss_weight: Schema.number().min(0).step(0.1).default(1.0).description("in_domain 二分类损失权重"),
        cls_pos_weight: Schema.string().description("分类正样本权重。留空表示不额外加权"),
        seed: Schema.number().default(42).description("随机种子"),
        device: Schema.string().default("cuda").description("运行设备，例如 `cuda`、`cuda:0`、`cpu`"),
    }).description("训练参数"),

    Schema.object({
        hidden_dims: Schema.string().default("1024,256").description("Fusion head 隐层维度，逗号分隔"),
        dropout: Schema.number().min(0).max(1).step(0.01).default(0.2).description("Fusion head dropout"),
        freeze_extractors: Schema.boolean().default(true).description("冻结 JTP-3 与 Waifu CLIP 特征提取器。当前第一版仅支持开启"),
        include_waifu_score: Schema.boolean().default(true).description("启用 Waifu Scorer v3 额外分支特征"),
    }).description("融合头设置"),

    Schema.object({
        jtp3_model_id: Schema.string().default("RedRocket/JTP-3").description("JTP-3 模型 ID 或本地目录"),
        jtp3_fallback_model_id: Schema.string().description("JTP-3 加载失败时的回退模型 ID"),
        hf_token_env: Schema.string().default("HF_TOKEN").description("读取 HuggingFace Token 的环境变量名"),
        waifu_clip_model_name: Schema.string().default("ViT-L-14").description("Waifu CLIP 模型名称"),
        waifu_clip_pretrained: Schema.string().default("openai").description("Waifu CLIP 预训练权重名称"),
        waifu_v3_head_path: Schema.string().role("filepicker", { type: "model-file" }).description("Waifu Scorer v3 头部权重路径。留空时自动尝试内置路径"),
    }).description("特征提取器设置"),
])
