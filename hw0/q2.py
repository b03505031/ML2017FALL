import PIL
from PIL import Image
fileName="westbrook.jpg"
origImg = Image.open(fileName)
pixels = origImg.load()

for i in range(origImg.size[0]): # for every pixel:
    for j in range(origImg.size[1]):
        pixels[i,j] = (int(pixels[i,j][0]/3), int(pixels[i,j][1]/3) ,int(pixels[i,j][2]/3))
origImg.save("Q2.png")