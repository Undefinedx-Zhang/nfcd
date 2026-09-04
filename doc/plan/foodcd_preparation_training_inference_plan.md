# FoodCD 数据整理、半监督训练与推理计划

## 1. 目标与已确认事实

目标是从 `/mnt/sdb/26_zdj/DATA/Annotations/images` 和同名的
`/mnt/sdb/26_zdj/DATA/Annotations/semantic_mask` 构建 FoodCD 二时相变化检测数据集，
训练“食物由未熟变为已熟”的二分类变化模型，并对单个图像对或整段视频输出变化像素数、
变化比例和阈值判断结果。

已核对的输入事实：

- `images` 与 `semantic_mask` 各有 7,068 个文件，并按相同 stem 一一对应。
- 文件名按最后一个下划线分割：前缀是视频 ID，末尾数字是抽帧序号；必须按该数字的数值排序，
  不能按字符串排序。例如 `2021_03_25_164751_3.jpg` 是该视频的首帧。
- 当前数据共有 61 个视频，每个视频有 30 至 200 帧，无重复的帧序号。
- `semantic_mask/class_map.json` 中 `background` 的值是 `0`，掩码是单通道 8 位 PNG。
- WHU 的标签也是单通道 PNG，实际像素值为 `0` 或 `255`；FoodCD 标签遵循该约定。
- 既定成熟度规则为：类别名后缀 `_0` 表示未熟，后缀不是 `_0` 表示已熟。

## 2. 输出目录与命名

创建以下目录，不修改原始 `images`、`semantic_mask`、`Annotations`：

```text
/mnt/sdb/26_zdj/DATA/Annotations/FoodCD/
├── A/                 # 每个样本的首帧参考图
├── B/                 # 每个样本对应的视频当前帧
├── label/             # 二值变化标签，单通道 PNG，值只能是 0 或 255
├── list/              # train/val/test 与 5/10/20/40 半监督列表
└── manifest.csv        # 每个样本的来源与生成结果，便于追溯

/mnt/sdb/26_zdj/DATA/Annotations/classify/
├── <video_id>/         # 一个视频一个目录，含该视频的所有原始图像
└── manifest.csv        # 视频 ID、首帧、帧数、目录、排序信息
```

样本名使用 B 时相的原始 stem，例如：

```text
A/2021_03_25_164751_10060.jpg
B/2021_03_25_164751_10060.jpg
label/2021_03_25_164751_10060.png
```

每个视频的第一帧同时作为所有样本的 A 时相；B 时相从第一帧开始，故第一对 `A == B`。
这样共有 7,068 个图像对。`manifest.csv` 至少记录 `sample_name`、`video_id`、`frame_index`、
`a_source`、`b_source`、`semantic_mask_source`、`label_path`、`changed_pixels`、`change_ratio`。

## 3. 标签生成规则

标签只依据 B 时相的语义掩码和 `class_map.json` 动态生成，不能将类别 ID 写死到脚本中。

1. 读取 `class_to_id`，将类别名后缀不是 `_0` 的 ID 收集为 `cooked_ids`；`background=0` 与所有
   后缀 `_0` 的 ID 均不属于该集合。
2. 对 B 的语义掩码逐像素计算 `mask in cooked_ids`。
3. 真值像素写为 `255`，其余像素写为 `0`，以无损单通道 PNG 保存。
4. 因此，B 中没有已熟目标时，标签必须是一张全 `0` 图；未熟目标虽然在原语义掩码中为非零值，
   在 FoodCD 标签中仍为 `0`。

此阶段的定义是“B 时相存在已熟区域”，前提是每个视频的首帧均不含已熟区域。生成脚本应先统计
所有首帧的 `cooked_pixels`。若任一首帧非零，停止正式生成并输出异常清单；届时需确认标签是否改为
`cooked(B) AND NOT cooked(A)`，避免把首帧已存在的熟食误标为变化。

## 4. 数据构建脚本

新增仓库脚本 `scripts/prepare_foodcd.py`，默认只在输出根目录写文件，包含以下步骤。

1. 扫描 JPG 和 PNG，以 stem 建立一对一索引；检查缺图、缺掩码、重复 stem、损坏文件、尺寸不一致。
2. 以 `rsplit('_', 1)` 提取 `video_id` 与整数 `frame_index`，按 `(video_id, frame_index)` 稳定排序；
   每组最小 `frame_index` 是 A 参考帧。
3. 为每个 B 帧建立一个样本：在 `A/` 放入对应视频首帧，在 `B/` 放入该 B 帧，并按第 3 节生成
   `label/<stem>.png`。A/B 保留 JPG，避免重复编码和无谓增大数据量。
4. 写入可复现的 `manifest.csv` 及 `summary.json`，其中包含总样本数、视频数、全零标签数、
   已熟像素总数、每视频帧数与首帧异常报告。
5. 支持 `--dry-run`、`--overwrite`、`--video-id`、`--workers` 与 `--copy-mode {hardlink,copy}`；
   默认先 dry-run。硬链接可节省空间，若源文件与目标位于不同文件系统则自动回退为复制。

## 5. 训练/验证划分与半监督列表

划分必须以 `video_id` 为最小单位，禁止将同一视频的相邻帧拆到训练和验证/测试集合，否则会产生严重
时序泄漏。

1. 以固定随机种子对 61 个视频做视频级划分，60 个视频用于训练、1 个视频用于验证，不设置测试集；
   记录种子和每个 split 的视频 ID 到 `list/split_manifest.json`。
2. `train.txt`、`val.txt`、`test.txt` 每行存一个 B 图像文件名（`.jpg`），名称同时索引 A、B 与标签。
3. 对训练集以同一固定排列构建四组嵌套标注比例：
   `5_train_supervised.txt`、`10_train_supervised.txt`、`20_train_supervised.txt`、
   `40_train_supervised.txt`。
4. 各比例对应的 `*_train_unsupervised.txt` 写入训练集的补集；同一比例的 supervised 与
   unsupervised 文件应当不重叠且并集等于训练集。
5. 生成后验证：每个列表没有重复名称；三种 split 没有共同视频；每个名称在 A、B、label 都能解析；
   所有标签只有 `0` 和 `255`。

## 6. 数据加载兼容改动

当前 `dataloaders/CDDataset.py` 用同一列表文件名同时拼接 A、B、label 路径。FoodCD 的 A/B 使用
`.jpg`、label 使用 `.png`，因此不能直接复用该行为。

在 `base/base_dataset.py` 与 `dataloaders/CDDataset.py` 增加可选配置项 `label_extension`：

```json
"label_extension": ".png"
```

加载器应保留列表中的 `.jpg` 名称作为 A/B 文件名，仅在读取 label 时替换扩展名为 `.png`。未配置该项时
保持现有行为，确保 WHU、CDD、LEVIR 不受影响。`BaseDataSet` 已将所有 `label >= 1` 归一为类别 `1`，
所以 FoodCD 的 `0/255` 标签会正确进入二分类交叉熵。

## 7. FoodCD 配置与训练前提

新增 `configs/config_FoodCD.json`，以 `config_WHU.json` 为基础，设置：

- 所有 `data_dir` 为 `/mnt/sdb/26_zdj/DATA/Annotations/FoodCD`。
- 所有 loader 都使用 `label_extension: ".png"`；训练 split 使用生成的 `5/10/20/40` 列表。
- `model.backbone` 初始使用 `ResNet50`，`n_gpu: 1`，`num_workers` 按机器负载设置。
- `trainer.process: [4]`，`sub_epochs[2]` 作为第 4 阶段训练轮数，输出路径使用 FoodCD 专属目录。
- `percent` 分别可取 `5`、`10`、`20`、`40`，并写明每次实验名与输出目录，避免互相覆盖。

当前项目第 4 阶段会训练有标注 CE、弱到强伪标签 CE、特征对齐三项损失；此前已移除两项 NF-concat
损失，并且第 4 阶段不再读取 NF 伪标签文件。它仍要求一个可用的 FoodCD 主模型初始化权重：

```bash
python train.py --config configs/config_FoodCD.json --gpu <GPU_ID> --aug_type all \
  --resume <FoodCD_stage1_checkpoint>
```

`process: [4]` 不会自行加载阶段 1 checkpoint，故不能省略 `--resume`。在仅运行第 4 阶段的前提下，
应先准备一个用 FoodCD 训练得到的阶段 1 权重；若使用跨数据集权重，需在实验记录中明确其来源与风险。

## 8. 推理脚本

新增 `food_inference.py`，不依赖 FoodCD 的标签文件，可加载训练 checkpoint 对新图像对推理。

命令接口：

```bash
python food_inference.py \
  --config configs/config_FoodCD.json \
  --model <stage3_checkpoint.pth> \
  --image-a <reference.jpg> \
  --image-b <current.jpg> \
  --pixel-prob-threshold 0.5 \
  --change-ratio-threshold 0.02 \
  --output-dir <result_dir>
```

脚本职责：

1. 用配置中的骨干构造模型，加载 checkpoint 的 `state_dict`，并采用与验证一致的归一化、缩放和
   padding 逻辑。
2. 计算每像素变化概率 `softmax(logits)[change]`，按 `--pixel-prob-threshold` 生成预测二值掩码。
3. 输出 `changed_pixels`、`total_pixels`、`change_ratio` 和
   `has_change = change_ratio >= change_ratio_threshold`，两个阈值分别可配置，避免混淆像素置信度与
   整体变化比例。
4. 保存 `prediction_mask.png`（0/255）、概率图、A/B/预测掩码叠加图和 `result.json`。
5. 支持 `--video-dir /mnt/sdb/26_zdj/DATA/Annotations/classify/<video_id>`：自动选取数值序最小的首帧为
   A，依次与每张图配对，并输出逐帧 `results.csv` 和整段视频的汇总结果。

## 9. classify 视频目录整理

数据构建脚本增加 `--build-classify`。对每个 `video_id` 创建
`/mnt/sdb/26_zdj/DATA/Annotations/classify/<video_id>/`，并按帧序号将原始 JPG 放入该目录。

默认使用相对符号链接，避免复制 7,068 张高分辨率图；`--classify-mode {symlink,hardlink,copy}` 可按
后续部署位置选择。每个视频目录额外写入 `frames.csv`，列出顺序、原始文件名、帧序号以及首帧标记，
使推理脚本无需依赖文件系统的字典序。

## 10. 验证、试运行与交付顺序

1. 先执行 `prepare_foodcd.py --dry-run`，人工确认 61 个视频分组、首帧、样本数、熟/未熟像素统计。
2. 抽查每种食物和 `_0`/非 `_0` 各至少一个样本，确认标签分别为全 0 与包含 255 的区域；对原图、
   语义掩码和生成标签做叠加可视化。
3. 生成正式 FoodCD 与 classify 目录，运行列表、路径、像素值、尺寸和视频级无泄漏检查。
4. 运行单个 `CDDataset` batch 冒烟测试，确认 JPG A/B、PNG label、弱/强配对增强及标签值均正确。
5. 以最小配置（例如 5%、少量 epoch）运行第 4 阶段，核对日志仅含 `loss_l`、`loss_ul_cls`、
   `loss_ul_alg` 三项组成，并在 FoodCD 输出目录保存 `notes.md`。
6. 用保留视频运行 `food_inference.py`，核对 JSON/CSV 中的变化像素数与比例，再根据业务误报/漏报
   调整像素概率阈值和变化比例阈值。

## 11. 需要在正式执行前确认的决策

- 首帧若已出现熟食，是否按当前规则仍标为变化，还是切换为 `cooked(B) AND NOT cooked(A)`？
- 是否保留当前按视频级划分的训练/验证集合，还是指定固定视频列表？
- `classify` 目录是否使用默认符号链接，还是必须物理复制图像？
- 第 4 阶段的 FoodCD 初始化 checkpoint 的具体路径与来源是什么？
