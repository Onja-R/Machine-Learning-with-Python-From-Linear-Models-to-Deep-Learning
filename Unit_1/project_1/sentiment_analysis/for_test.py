import pywhatkit as pwk

PATH = '/home/itachi/Desktop/'
FILE = '157586394_3801694053241955_4433708057970546574_o.jpg'

pwk.image_to_ascii_art(
    PATH + FILE,
    PATH + 'tupac.txt'
)
