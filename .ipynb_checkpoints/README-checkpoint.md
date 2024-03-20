# Project: End to End Deep Learning Architectures for Automatic Modulation Recognition

This project explores the application of Convolutional Neural Networks and their applications for Automatic Modulation Recognition on the RadioML and HisarMod Datasets.

Authors: [Timothy Do](https://timothydo.me)

## Dependencies
**Hardware:** Desktop with NVIDIA Geforce RTX GPU (e.g. NVIDIA RTX 3080) running Windows 10/11. RAM capacity should have at least 64 GB (for training). <br>
**Software:** Python 3.11

## Setup
1. Create a folder titled <code>datasets</code>.
2. Download the RadioML 2018.01A dataset from [https://www.deepsig.ai/datasets/](https://www.deepsig.ai/datasets/) and put it inside of <code>datasets</code>
3. Untar the file <code>2018.01.OSC.0001_1024x2M.h5.tar.gz</code>
4. Download the HisarMod 2019.1 dataset from [https://ieee-dataport.org/open-access/hisarmod-new-challenging-modulated-signals-dataset] and put it inside of <code>datasets</code>.
5. Unzip the file <code>HisarMod2019.1.zip</code>.
6. Install Python dependencies by running <code>pip install -r requirements.txt</code>.
7. Run <code>jupyter notebook</code> and open <code>Project.ipynb</code> to execute the project!