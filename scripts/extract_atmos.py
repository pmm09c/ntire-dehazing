from os import walk
from PIL import Image
import numpy as np

def list_files(directory, extension):
    for (dirpath, dirnames, filenames) in walk(directory):
        return (f for f in filenames if f.endswith('.' + extension))

class Channel_value:
    val = -1.0
    intensity = -1.0

    
def find_intensity_of_atmospheric_light(img, gray):
    top_num = int(img.shape[0] * img.shape[1] * 0.001)
    toplist = [Channel_value()] * top_num
    dark_channel = find_dark_channel(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            val = img.item(y, x, dark_channel)
            intensity = gray.item(y, x)
            for t in toplist:
                if t.val < val or (t.val == val and t.intensity < intensity):
                    t.val = val
                    t.intensity = intensity
                    break

    max_channel = Channel_value()
    for t in toplist:
        if t.intensity > max_channel.intensity:
            max_channel = t

    return max_channel.intensity


def find_dark_channel(img):
    return np.unravel_index(np.argmin(img), img.shape)[2]


def clamp(minimum, x, maximum):
    return max(minimum, min(x, maximum))

directory = '/home/pmorales/projects/IRP/dataset-mini/haze'
write_directory = '/home/pmorales/projects/IRP/dataset-mini/atmos'
extension = 'png'
files = list_files(directory,extension)
atmos = Image.new("L", (800,600 ))
for f in files:
    img = Image.open(directory+'/'+f)
    grayscale = img.convert('L')
    pix_img = np.array(img.getdata()).reshape(img.size[1],img.size[0],3)
    pix_gry = np.array(grayscale.getdata()).reshape(grayscale.size[1],grayscale.size[0])
    intensity = find_intensity_of_atmospheric_light(pix_img, pix_gry)
    atmos.paste(tuple(intensity),(0,0,atmos.size[0],atmos.size[1]))
    atmos.save(write_directory+'/'+f)
