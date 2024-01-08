# GraphEmotionNet
A Pytorch implementation of our paper "Adaptive Spatial-Temporal Aware Graph Learning for1
EEG-based Emotion Recognition".
You can find the corresponding article here: [GraphEmotionNet](https://spj.science.org/doi/pdf/10.34133/cbsystems.0088)
# Dataset
Prepare dataset: [SEED](https://bcmi.sjtu.edu.cn/home/seed/seed.html) and [SEED-IV](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html)
# Training
The model definition is in file: models.py

The code for the Domain Adaptation part is in the file adv_layer.py

You can start training the model by running the main.py file
# Citation
If you find our work helps your research, please kindly consider citing our paper in your publications.

@article{yeadaptive,
  title={Adaptive Spatial-Temporal Aware Graph Learning for EEG-based Emotion Recognition},
  author={Ye, Weishan and Wang, Jiyuan and Chen, Lin and Dai, Lifei and Sun, Zhe and Liang, Zhen},
  journal={Cyborg and Bionic Systems},
  publisher={AAAS}
}
