## Pinyin Name Gender Prediction
This is the source code for paper
## Abstract
Achieving gender equality is a pivotal factor in realizing the UN's Global Goals for Sustainable Development. Gender bias studies work towards this and rely on name-based gender inference tools to assign individual gender labels when gender information is unavailable. However, these tools often inaccurately predict gender for Chinese Pinyin names, leading to potential bias in such studies. With the growing participation of Chinese in international activities, this situation is becoming more severe. Specifically, current tools focus on pronunciation (Pinyin) information, neglecting the fact that the latent connections between Pinyin and Chinese characters behind convey critical information. As a first effort, we design a Multi-Task Learning Network assisted by Knowledge Distillation that enables the Pinyin representations in the model to possess semantic features of Chinese characters and to learn gender information from Chinese character names. Our open-sourced method surpasses commercial name-gender guessing tools by 9.70% to 20.08% relatively, and also outperforms the state-of-the-art algorithms.
## Data
We provide the 9,800 Names dataset and the 20,000 Names dataset used in our experiments in the `./data` folder. The 58M Names dataset will be made available on request.
## Prerequisites
The code has been successfully tested in the following environment.
 - Python 3.8.2
 - PyTorch 1.7.0+cu110
 - numpy 1.19.3
 - Sklearn 1.0.2
 - Pandas 1.2.2
 - Transformers 4.3.2
 - pypinyin 0.49.0
 - pinyinsplit 0.1.4
## Getting Started
### Prepare your data
For model training, both Pinyin and Chinese characters are needed. For model inference, only Pinyin is necessary.
### Training Model
Please run following commands for training.
```python
python pinyin_mtl_kd.py
```
## Cite
Please cite our paper if you find this code useful for your research:
