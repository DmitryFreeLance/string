import os
import json

settings = json.load(open('settings.json'))
print(settings)

# параметры
d = settings['dimension']
l = settings['pulls'] // 3
s = settings['strength']
rect = settings['rect']
input_image_path = settings['input_image_path']
long_side = settings['long_side'] # 385 - 360, 256 - 240

# здесь ничего менять не нужно
rect = '--rect' if rect  else ''
image_name = input_image_path.split('.')[0]
output_image_path = f'result/{image_name}_result_{d}_{long_side}_{l}_{s}.jpg'
try:
    os.mkdir('result')
except:
    pass

command = f'python generate.py -i images/{input_image_path} -o {output_image_path} -d {d} -l {l} -s {s} --rgb {rect} -longside {long_side}'
os.system(command)

# pyinstaller -F -w -i 'ico.png' script.py
