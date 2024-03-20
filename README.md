# Project: End to End Deep Learning Architectures for Automatic Modulation Recognition

This project explores the application of Convolutional Neural Networks and their applications for Automatic Modulation Recognition on the RadioML and HisarMod Datasets.

Authors: [Timothy Do](https://timothydo.me)

## Dependencies
**Hardware:** Desktop with an multi-core processor (e.g. Intel Core i7), NVIDIA Geforce RTX GPU with adequate VRAM (e.g. NVIDIA RTX 3080) running Windows 10/11. RAM capacity should have at least 64 GB (for training). <br>
**Software:** Python 3.11, MATLAB R2023b. Other versions of MATLAB/Python may work. <br><br>
For perspective, I developed this project using a custom desktop following components: <br>
<li><b>CPU:</b> Intel Core i7-11700K</li>
<li><b>RAM:</b> 128GB DDR4 @ 3200 MHz</li>
<li><b>GPU:</b> NVIDIA RTX 3090</li> <br>

## Setup
1. Create a folder titled <code>datasets</code>.
2. Download the RadioML 2016.10A dataset from [Deepsig](https://www.deepsig.ai/datasets/) and put it inside of <code>datasets</code>.
3. Untar the file <code>RML2016.10a.tar.bz2</code>.
4. Download the RadioML 2018.01A dataset from [Deepsig](https://www.deepsig.ai/datasets/) and put it inside of <code>datasets</code>
5. Untar the file <code>2018.01.OSC.0001_1024x2M.h5.tar.gz</code>.
6. Download the HisarMod 2019.1 dataset from [IEEEDataport](https://ieee-dataport.org/open-access/hisarmod-new-challenging-modulated-signals-dataset) and put it inside of <code>datasets</code>.
7. Unzip the file <code>HisarMod2019.1.zip</code>.
8. Run the MATLAB script <code>convertHisarToPython.m</code> to convert the data .csv files to Python readable .mat files.
9. Install Python dependencies by running <code>pip install -r requirements.txt</code>.
10. Run <code>jupyter notebook</code> and open <code>Project.ipynb</code> to execute the project!
