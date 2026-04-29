Schema.intersect([
    Schema.object({
        model_train_type: Schema.string().default("yolo").disabled().description("训练种类"),
        pretrained_model_name_or_path: Schema.string().default("yolo11n.pt").description("YOLO 模型权重或模型 yaml。可填本地路径，或官方模型名如 `yolo11n.pt`"),
        resume: Schema.string().role("filepicker", { type: "model-file" }).description("从已有 YOLO 训练检查点继续训练。填写 `last.pt` 一类的检查点文件路径"),
    }).description("训练用模型"),

    Schema.object({
        yolo_data_config_path: Schema.string().role("filepicker", { type: "model-file" }).description("可选。自定义 YOLO 数据集 yaml。填写后，下方训练/验证目录与类别列表只作为参考，不再参与生成"),
        train_data_dir: Schema.string().role("filepicker", { type: "folder" }).default("./datasets/images/train").description("训练图像目录。留空仅在填写自定义数据集 yaml 时允许"),
        val_data_dir: Schema.string().role("filepicker", { type: "folder" }).default("./datasets/images/val").description("验证图像目录。留空时会回退为训练目录"),
        class_names: Schema.string().role("textarea").default("class0").description("类别名称，一行一个。仅在未填写自定义数据集 yaml 时用于自动生成数据集配置"),
    }).description("数据集设置"),

    Schema.object({
        output_name: Schema.string().default("exp").description("本次训练输出名称"),
        output_dir: Schema.string().role("filepicker", { type: "folder" }).default("./output/yolo").description("训练输出目录"),
        save_every_n_epochs: Schema.number().min(1).default(10).description("每 N 个 epoch 保存一次检查点"),
    }).description("保存设置"),

    Schema.object({
        epochs: Schema.number().min(1).default(100).description("训练 epoch 数"),
        batch: Schema.number().min(1).default(16).description("训练批量大小"),
        imgsz: Schema.number().min(32).default(640).description("训练输入分辨率"),
        workers: Schema.number().min(0).default(8).description("数据加载 worker 数量"),
        device: Schema.string().description("可选。手动指定 Ultralytics device，例如 `0`、`0,1`、`cpu`。留空时按当前可见 GPU 自动决定"),
        seed: Schema.number().default(1337).description("随机种子"),
    }).description("训练参数"),
])
