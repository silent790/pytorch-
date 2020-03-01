import requests
import os
import zipfile
import shutil
file_url =  'http://labfile.oss.aliyuncs.com/courses/1073/MNIST/data.zip'
file_name = 'data.zip'
# download file with requests pkg
r = requests.get(file_url)
with open(file_name, 'wb') as code:
     code.write(r.content)

# extract file 
f = zipfile.ZipFile(file_name, 'r')
for file in f.namelist():
    f.extract(file, './')