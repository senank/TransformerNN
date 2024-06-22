<h2>Bigram Transformer Model</h2>
This repository contains a character-level pre-trained transformer-based text generation model, designed to mimic the style and content of its training data (Shakespeare) built from scratch. This model leverages bigrams as its fundamental building blocks, and is structured with multi-headed attention blocks, layer normalization, dropout, and feedforward layers to enhance learning and generalization. This model offers a starting framework for experimenting with various textual styles and formats.<br/>
This application also runs on both a GPU and a CPU (note the training parameters are set for GPU training, please scale down the Model Constants in model.py for faster training<br/><br/>

To run this program, download the required modules with: <br />
```pip install -r requirements.txt```

<br />
If you would like to train this model on your own dataset, please add a .txt file with the contents you would like to mimic. <br/>
<em>*Add the input file to the current working directory</em> </br>
<em>*Ensure that input.txt (from git pull) remains unchanged</em>
<br />
<br />

To run the program run the following and follow the prompts (after cloning this repo): <br />
```python train.py```
