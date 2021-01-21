import streamlit as st

import numpy as np
import pandas as pd

from PIL import Image, ImageDraw

import colorsys

from models.sentence2vec import Sentence2Vec
import torch
import torchvision

from matplotlib import pyplot as plt
import matplotlib
from skimage.color import lab2rgb, rgb2lab
import pickle
import os


from initModel import CA_NET, EncoderRNN, Attn, AttnDecoderRNN, Discriminator, DecoderRNN, Text2ColorDataset
    
#标题：
st.title('Automatic Coloring Tool for Chinese Youth Subculture')
st.sidebar.header('面向中国亚文化的自动上色工具')

st.sidebar.subheader('Choose Process:')

genre = st.sidebar.radio(
    "选择步骤:",
    ('1: Generate Palette', '2: Adjust Palette (not necessary)', '3: Colorization'))

if genre == '1: Generate Palette':

    batch_size = 16
    lr = 1e-4
    weight_decay = 5e-5
    beta1 = 0.5
    beta2 = 0.99
    hidden_dim = 768

    max_iter_cnt = 1e5

    print_every_iter = 1
    save_every_epoch = 3


    device = torch.device('cpu')



    # utils
    def lab2rgb_1d(in_lab, clip=True):
        tmp_rgb = lab2rgb(in_lab[np.newaxis, np.newaxis, :], illuminant='D50').flatten()
        if clip:
            tmp_rgb = np.clip(tmp_rgb, 0, 1)
        return tmp_rgb


    # # define data_loader
    # dataset = Text2ColorDataset(text_path, palette_path)
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)


    @st.cache(allow_output_mutation=True)
    def init():  
        # define models
        encoder = EncoderRNN(hidden_size=150, n_layers=1, dropout_p=0).to(device)
        decoder = AttnDecoderRNN(hidden_size=150, n_layers=1, dropout_p=0).to(device)
        discriminator = Discriminator(color_size=15, hidden_dim=150).to(device)
        embed_model = Sentence2Vec()
        # init model
        encoder.load_state_dict(torch.load('./palette_gen_ckpt/ckpt_666.pt', map_location=lambda storage, loc: storage)['encoder'])
        decoder.load_state_dict(torch.load('./palette_gen_ckpt/ckpt_666.pt', map_location=lambda storage, loc: storage)['decoder_state_dict'])
        decoder.eval()
        print("1") 

        return encoder, decoder, discriminator, embed_model

    encoder, decoder, discriminator, embed_model = init()

    #输入文字
    words = st.text_input('Input any Chinese words or sentence here:',value='新年快乐 万事如意', max_chars=None, key=None, type='default')
    st.write('you want palette that is: ', words) 


    # 使用缓存
    def palette_result(word): 

        text = embed_model.embed(word)['pooler_output'].unsqueeze(0).to(device)
        batch_size = text.size(0)
        nonzero_indices = list(torch.nonzero(text)[:, 0])
        each_input_size = [nonzero_indices.count(j) for j in range(batch_size)]
        print ('gen_text run')
        

        palette = torch.FloatTensor(batch_size, 3).zero_().to(device)
        fake_palettes = torch.FloatTensor(batch_size, 15).zero_().to(device)
        encoder_hidden = encoder.init_hidden(batch_size).to(device)

        encoder_outputs, decoder_hidden, mu, logvar = encoder(text, encoder_hidden)
        
        decoder_hidden = decoder_hidden.squeeze(0)
        
        for i in range(5):
            palette, decoder_context, decoder_hidden = decoder(palette,
                                                                decoder_hidden,
                                                                encoder_outputs,
                                                                each_input_size,
                                                                i)

            fake_palettes[:, 3 * i:3 * (i + 1)] = palette
        

        fake_palettes = fake_palettes.squeeze(0)
        print ('gen_palette run')



        rgb_5 = []
        for k in range(0,5):
            lab = np.array([fake_palettes.data[3*k],
                            fake_palettes.data[3*k+1],
                            fake_palettes.data[3*k+2]], dtype='float64')
            rgb = lab2rgb_1d(lab)
            # axs1[k].imshow([[rgb]])
            # axs1[k].axis('off')
            rgb = rgb*255
            rgb_5.append(rgb)
        print('gen_rgb run')
        

        print (word)
        hsl = [] 
        HEX = []
        # 建个画板开始画
        for j in range(0,5):
            
            r = round(rgb_5[j][0])
            g = round(rgb_5[j][1])
            b = round(rgb_5[j][2])
            HEX.append('#%02X%02X%02X' % (r, g, b))
            hsli = list(colorsys.rgb_to_hls(r/255, g/255, b/255))
            hsl.append(hsli)
            # print (HEX)
            
        palette0 = Image.new('RGB',(100,100), color = HEX[0])
        palette1 = Image.new('RGB',(100,100), color = HEX[1])
        palette2 = Image.new('RGB',(100,100), color = HEX[2])
        palette3 = Image.new('RGB',(100,100), color = HEX[3])
        palette4 = Image.new('RGB',(100,100), color = HEX[4])

        
        print("3") 

        return hsl, palette0, palette1, palette2, palette3, palette4, HEX


    hsl, palette0, palette1, palette2, palette3, palette4, HEX = palette_result(words)
    st.image([palette0,palette1,palette2,palette3,palette4])  
    st.subheader('Please copy the palette HEX below: ')
    st.write(HEX[0], HEX[1], HEX[2], HEX[3], HEX[4])




if genre == '2: Adjust Palette (not necessary)':
    #正文
    st.write('Paste your HEX here:') 
    words = st.text_input('在此处粘贴您的色板代码：',value='#E9E3D1 #513624 #55442C #947550 #7B623F', max_chars=None, key=None, type='default')
    st.write('You want to change palette: ', words) 


    HEX = str.split(words)
    print (HEX[4])

    def Hex_to_RGB(hex):
        r = int(hex[1:3],16)
        g = int(hex[3:5],16)
        b = int(hex[5:7], 16)
        rgb = (r,g,b)
        return rgb

    rgb_list = []
    for k in range(0,5):
        rgb_list.append(Hex_to_RGB(HEX[k]))

    # st.write("rgb is :", rgb_list) 

    palette = rgb_list

    # print (palette[0][0])
    # exit(0)

    hsl = []
    # 建个画板开始画
    for i in range(0,5):
        
        r = palette[i][0]
        g = palette[i][1]
        b = palette[i][2]
        HEX = '#%02X%02X%02X' % (r, g, b)
        hsl.append(colorsys.rgb_to_hls(r/255, g/255, b/255))
        # print (HEX)
        locals()['palette'+str(i)] = Image.new('RGB',(100,100), color = HEX)
        #imnew.save('interactivedemo/%d.png' %(i))
        
        # im1 = ImageDraw.Draw(imnew1)
        # imnew1.show()

    st.image([palette0,palette1,palette2,palette3,palette4])

    st.write('The adjustment shows below: ') 

        
    h = []
    l = []
    s = []
    #更改hsl
    # st.sidebar.write(hsl[color_num][0]*360,hsl[color_num][1],hsl[color_num][2])
    for i in range(0,5):
        st.sidebar.write("——————Change color: ", i+1,'——————')

        h.append(st.sidebar.slider('Hue:', 0.0, 360.0, hsl[i][0]*360))
        st.sidebar.write("Hue is ", '%.2f' %h[i], '°')

        l.append(st.sidebar.slider('Lightness:', 0.0, 1.0, hsl[i][1]))
        st.sidebar.write("Lightness is ", '{:.2%}'.format(l[i]))

        s.append(st.sidebar.slider('Saturation:', 0.0, 1.0, hsl[i][2]))
        st.sidebar.write("Saturation is ", '{:.2%}'.format(s[i]))

    HEX_change = []

    for i in range(0,5):
        rgb_change = colorsys.hls_to_rgb(h[i]/360,l[i],s[i])
        r = int(rgb_change[0]*255)
        g = int(rgb_change[1]*255)
        b = int(rgb_change[2]*255)

        HEX_change.append('#%02X%02X%02X' % (r, g, b))

    palette00 = Image.new('RGB',(100,100), color = HEX_change[0])
    palette01 = Image.new('RGB',(100,100), color = HEX_change[1])
    palette02 = Image.new('RGB',(100,100), color = HEX_change[2])
    palette03 = Image.new('RGB',(100,100), color = HEX_change[3])
    palette04 = Image.new('RGB',(100,100), color = HEX_change[4])

    st.image([palette00,palette01,palette02,palette03,palette04])

    st.subheader('Please copy the palette HEX that you are satisfied with: ') 
    st.write(HEX_change[0],HEX_change[1],HEX_change[2],HEX_change[3],HEX_change[4])




if genre == '3: Colorization':
    st.write('colorization!')






st.sidebar.subheader('Upload your image here:')
f = st.sidebar.file_uploader('上传您想要上色的图片：', type=None, accept_multiple_files=False, key=None)
if f is not None:
    img = Image.open(f)
    st.sidebar.image(img,use_column_width=True)