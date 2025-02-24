# 🚀 PyTorch Implementation of GEM 🌟

Welcome to the official PyTorch implementation of **GEM**! 🎉

GEM was introduced in our [ICLR 2025 paper](https://openreview.net/forum?id=dulz3WVhMR) "Preserving Diversity in Supervised Fine-tuning of Large Language Models". This work was previously titled "Entropic Distribution Matching in Supervised Fine-tuning of LLMs: Less Overfitting and Better Diversity" and received the Best Paper Runner-up Award at the NeurIPS 2024 FITML Workshop.


<img src='./img/gem_vs_ce.png' width='700'>

GEM can replace the CE loss during SFT to preserve diversity and mitigate overfitting. 🌍✨

For more insights on GEM's potential to enhance RL training through improved cold-start strategies, check out our blog post: ["Can Better Cold-Start Strategies Improve RL Training for LLMs?"](https://tangible-polo-203.notion.site/Can-Better-Cold-Start-Strategies-Improve-RL-Training-for-LLMs-17aa0742a51680828616c867ed53bc6b)

## Quickstart Guide 💻

### Setup 🔧

First, create a new environment and install the required packages:

```bash
conda create -n gem python=3.10
conda activate gem
pip install -r requirements.txt
```

Note that the version of packages in `requirements.txt` is used in the paper. If you use a higher version of transformers (>= 4.46.0), you may need to follow the code in `sft_trainer_v2.py` to adapt to the new version.

### Training 🏋️‍♂️

Kickstart your training process using the `UltraFeedback` dataset from HuggingFace. Here's how:

**Tokenize Data**

```bash
bash scripts/tokenize_data.sh
```

**Training**

```bash
bash scripts/train_gem_ultrafeedback.sh
```

### Evaluation 🧪

Run evaluations for different tasks:

**GSM8K**

```bash 
bash scripts/eval/gsm8k_eval.sh
```

**GSM8K (Voting)**

```bash
bash scripts/eval/gsm8k_voting_eval.sh
```

**Creative Writing**

```bash
bash scripts/eval/creative_writing.sh
```

## 📜 Citation

If you find this repository helpful in your research or projects, please consider citing the GEM paper in your academic work. Your support is much appreciated! 🙌

```bibtex
@article{li2024entropic,
  title={Entropic Distribution Matching in Supervised Fine-tuning of LLMs: Less Overfitting and Better Diversity},
  author={Li, Ziniu and Chen, Congliang and Xu, Tian and Qin, Zeyu and Xiao, Jiancong and Sun, Ruoyu and Luo, Zhi-Quan},
  journal={arXiv preprint arXiv:2408.16673},
  year={2024}
}
```

Ziniu Li would like to acknowledge Zhengyang Tang for his minimalistic and clean implementation of SFT.
