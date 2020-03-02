import requests
import os
import zipfile
import shutil

# GAN数据包
# file_url = 'http://labfile.oss.aliyuncs.com/courses/1073/GAN_models.zip'
# file_name = 'GAN_models.zip'

# MINST数据包：
file_url = 'http://labfile.oss.aliyuncs.com/courses/1073/MNIST/data.zip'
file_name = 'data.zip'

r = requests.get(file_url)
with open(file_name, 'wb') as code:
    code.write(r.content)

f = zipfile.ZipFile(file_name, 'r')
for file in f.namelist():
    f.extract(file, './')
