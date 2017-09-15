import sys
from PIL import Image
#fileName="westbrook.jpg"
#flieName = sys.argv[1]
origImg = Image.open(sys.argv[1])
pixels = origImg.load()

for i in range(origImg.size[0]): # for every pixel:
    for j in range(origImg.size[1]):
        pixels[i,j] = (int(pixels[i,j][0]/2), int(pixels[i,j][1]/2) ,int(pixels[i,j][2]/2))
origImg.save("Q2.png")
