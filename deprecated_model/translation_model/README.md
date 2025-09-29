## How to run
**因為儲存參數的檔案過大，不一上傳，所以建議 clone 下來自己跑一次訓練**
### 1. create a env under python = 3.7, and install demand packages
```bash
# create virtual env
conda create -n <name of the env> python=3.7

# activate the env
conda activate <name of the env>

# install packages
pip install -r requirements.txt
```
### 2. download ckiptagger data files
```bash
python download_tool.py
# the downloaded files will be put in data/
```
### 3. train the model
```bash
python train.py
```
### 4. test the model
```bash
python infer.py
# you can enter chinese sentences to test the model now
# if you don't know what sentence to input
# you can start with the sentences in the data/translation_dataset.txt
# with it you can see if the model translate correctly
```
