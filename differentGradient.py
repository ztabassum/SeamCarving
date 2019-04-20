import helperFunctions
import matplotlib.image as mpimg
import scipy



def reducePrageHeightDifferentMap():
    img = mpimg.imread('inputSeamCarvingPrague.jpg')
    energy_output = helperFunctions.different_energy_map(img)
    for i in range(100):
        reducedHeightResults = helperFunctions.reduceHeightDifferentMap(img, energy_output)
        img = reducedHeightResults[0]
        energy_output= reducedHeightResults[1]

    scipy.misc.imsave('outputReduceHeightPragueDifferentMap.png', img)

def reducePragueWidthDifferentMap():
    img = mpimg.imread('inputSeamCarvingPrague.jpg')
    energy_output = helperFunctions.different_energy_map(img)
    for i in range(100):
        reducedWidthResults = helperFunctions.reduceWidthDifferentMap(img, energy_output)
        img = reducedWidthResults[0]
        energy_output = reducedWidthResults[1]

    scipy.misc.imsave('outputReduceWidthPragueDifferentMap.png', img)

if __name__ == '__main__':
    reducePragueWidthDifferentMap()
