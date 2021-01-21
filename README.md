# subculture-palette
Palette Generation and Colorization for Youth Subculture in China

This project includes 1 dataset (image-palette-text dataset for youth subculture in China) and 2 jupyter notebook file (one for subcultural palette generation and another one for colorization).


## Image-palette-text dataset for youth subculture in China



![data prepare](pic/dataset_str.jpg)

Find the dataset in the folder './dataset'


## Framework: palette generation and colorization

![palette generation and colorization](pic/structure.jpg)

There are two main model in this framework: palette generation model and colorization model. The palettes, words/phrase, images from the dataset as the input are entered into the first model. Through the palette generation cGAN(illustrated in blue boxes), fake palettes of youth subculture are generated. Then the fake palettes and the original images that are changed color are as the input of the colorization model. Through the encoder, decoder and discriminator(illustrated in pink boxes), recolored images are generated.

### train:

palette_gen_training.ipynb

colorization.ipynb



## Demo

![demo](pic/demo.jpg)

### run demo:
 ```python
streamlit run demo.py
 ```