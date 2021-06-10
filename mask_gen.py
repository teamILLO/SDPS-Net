from PIL import Image

# image resize
def create_img(name, out_num) :
    i = Image.open('./data/dpr/obama/{}.jpg'.format(name), mode='r')
    j = i.resize((256,256), resample=0)
    j.save('./data/dpr/obama/out{}.png'.format(out_num), format='PNG')

# mask gen
def create_mask() :
    img = Image.new(mode='RGB', size=(256, 256), color=0xFFFFFF)
    img.save('./data/dpr/obama/mask.png', format='PNG')

for i in range(7) : 
    create_img('obama_0{}'.format(str(i)), str(i))
create_mask()
