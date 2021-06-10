from PIL import Image

img = Image.new(mode='RGB', size=(1296, 1944), color=0xFFFFFF)
img.save('./data/sample/shader/mask.png', format='PNG')