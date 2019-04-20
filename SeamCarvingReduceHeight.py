import helperFunctions
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy


def reducePrageHeight():
    img = mpimg.imread('inputSeamCarvingPrague.jpg')
    energy_output = helperFunctions.energy_image(img)
    for i in range(100):
        reducedHeightResults = helperFunctions.reduceHeight(img, energy_output)
        img = reducedHeightResults[0]
        energy_output= reducedHeightResults[1]

    scipy.misc.imsave('outputReduceHeightPrague.png', img)

def reduceMallHeight():
    img = mpimg.imread('inputSeamCarvingMall.jpg')
    energy_output = helperFunctions.energy_image(img)
    for i in range(100):
        reducedHeightResults = helperFunctions.reduceHeight(img, energy_output)
        img = reducedHeightResults[0]
        energy_output = reducedHeightResults[1]

    scipy.misc.imsave('outputReduceHeightMall.png', img)


if __name__ == '__main__':
   # reduceMallHeight()
   reducePrageHeight()
