# 合作者文件放置指引

## 第一步：clone 代码仓库

```bash
git clone https://github.com/Fu-fangteng/AIAA_4051_NLP_Final_Proj.git project
cd project
```

克隆完成后，`project/` 目录中包含所有代码、配置文件和结果图片，但**缺少大文件**（模型权重、数据集缓存、pkl 缓存）。

---

## 第二步：解压三个压缩包

将 `part1_checkpoints.zip`、`part2_gpt2_and_task3_modelA.zip`、`part3_task3_modelB_data_pkl.zip` 放到 `project/` 目录内，然后逐一解压：

```bash
# 在 project/ 目录下执行：
unzip part1_checkpoints.zip
unzip part2_gpt2_and_task3_modelA.zip
unzip part3_task3_modelB_data_pkl.zip
```

> **注意**：三个压缩包都保留了从 `project/` 根目录出发的相对路径，解压后文件会自动落到正确位置，不需要手动移动。

---

## 第三步：解压后删除 zip 文件（可选）

```bash
rm part1_checkpoints.zip part2_gpt2_and_task3_modelA.zip part3_task3_modelB_data_pkl.zip
```

---

## 解压后的目录结构确认

解压完成后，以下路径应均存在文件：

```
project/
├── gpt2/model.safetensors                                      ← part2
├── aiaa4051/checkpoints/sft_final/model.safetensors            ← part1
├── aiaa4051/checkpoints/step1_squad/final/model.safetensors    ← part1
├── aiaa4051/checkpoints/step1_squad/checkpoint-625/model.safetensors  ← part1
├── aiaa4051/checkpoints/step1_squad/checkpoint-1250/model.safetensors ← part1
├── aiaa4051/checkpoints/step2_sciq/checkpoint-375/model.safetensors   ← part1
├── aiaa4051/checkpoints/step2_sciq/checkpoint-750/model.safetensors   ← part1
├── aiaa4051/task3/modelA/final/model.safetensors               ← part2
├── aiaa4051/task3/modelB/final/model.safetensors               ← part3
├── aiaa4051/data/sciq_train3000/data-00000-of-00001.arrow      ← part3
├── aiaa4051/data/squad_v2_train5000/data-00000-of-00001.arrow  ← part3
├── aiaa4051/task1/relevance/relevances.pkl                     ← part3
├── aiaa4051/task1/faithfulness/faithfulness_data.pkl           ← part3
└── aiaa4051/task2/comparison/comparison.pkl                    ← part3
```

---

## 安装依赖

```bash
pip install -r requirements.txt
cd LRP-eXplains-Transformers && pip install -e . && cd ..
```

---

## 完成

解压并安装依赖后，本地文件结构与原始开发环境完全一致。
