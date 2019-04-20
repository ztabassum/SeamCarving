import helperFunctions
import matplotlib.image as mpimg
import scipy


def reducePrageWidth():
    img = mpimg.imread('inputSeamCarvingPrague.jpg')
    energy_output = helperFunctions.energy_image(img)
    for i in range(100):
        reducedWidthResults = helperFunctions.reduceWidth(img, energy_output)
        img = reducedWidthResults[0]
        energy_output= reducedWidthResults[1]

    scipy.misc.imsave('outputReduceWidthPrague.png', img)


def reduceMallWidth():
    img = mpimg.imread('inputSeamCarvingMall.jpg')
    energy_output = helperFunctions.energy_image(img)
    for i in range(100):
        reducedWidthtResults = helperFunctions.reduceWidth(img, energy_output)
        img = reducedWidthtResults[0]
        energy_output= reducedWidthtResults[1]

    scipy.misc.imsave('outputReduceWidthMall.png', img)
    return img,energy_output


if __name__ == '__main__':
  reduceMallWidth()




