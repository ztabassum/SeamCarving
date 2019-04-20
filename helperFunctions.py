import numpy
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy
def energy_image(im):
    gray_image =  numpy.dot(im[..., :3], [0.2989, 0.5870, 0.1140])

    sobel_x=numpy.matrix([[1,0,-1],[2,0,-2],[1,0,-1]])
    sobel_y = numpy.matrix([[1,2,1],[0,0,0],[-1,-2,-1]])

    x_gradient = scipy.ndimage.convolve(gray_image,weights=sobel_x)
    y_gradient = scipy.ndimage.convolve(gray_image,weights=sobel_y)

    energyImage = numpy.abs(x_gradient) + numpy.abs(y_gradient)

    return energyImage

def cumulative_minimum_energy_map(energyImage, seamDirection):

    row, column = energyImage.shape[0:2]
    cumulativeEnergyMap = numpy.zeros((energyImage.shape[0], energyImage.shape[1]))
    if seamDirection == 'VERTICAL':
        #first row will be all the values from the energy_map
        cumulativeEnergyMap[:, 0] = energyImage[:, 0]
        for i in range(1,row):
            for j in range(column):
               #left most column
                if j == 0:
                    cumulativeEnergyMap[i][j] = energyImage[i][j] + min(cumulativeEnergyMap[i - 1][j],
                                                                      cumulativeEnergyMap[i - 1][j + 1])
                #right most column
                elif j == column -1:
                    cumulativeEnergyMap[i][j] = energyImage[i][j] + min(cumulativeEnergyMap[i - 1][j],
                                                                        cumulativeEnergyMap[i - 1][j - 1])
                else :
                    cumulativeEnergyMap[i][j] = energyImage[i][j] + min(cumulativeEnergyMap[i-1][j-1],
                                                                   cumulativeEnergyMap[i-1][j],
                                                                   cumulativeEnergyMap[i-1][j+1])
    if seamDirection == 'HORIZONTAL':
        #first column will be all the values from the energy map
        cumulativeEnergyMap[0, :] = energyImage[0, :]
        for i in range(1,column):
            for j in range(row):
               #top row
               if j == 0:
                   cumulativeEnergyMap[j][i] = energyImage[j][i] + min(cumulativeEnergyMap[j][i-1],
                                                                       cumulativeEnergyMap[j+1][i-1])
               #bottom row
               elif j == row -1:
                   cumulativeEnergyMap[j][i] = energyImage[j][i] + min(cumulativeEnergyMap[j][i-1],
                                                                cumulativeEnergyMap[j-1][i-1])
               else:
                   cumulativeEnergyMap[j][i] = energyImage[j][i] + min(cumulativeEnergyMap[j-1][i-1],
                                                                       cumulativeEnergyMap[j][i-1],
                                                                       cumulativeEnergyMap[j+1][i-1])
    return cumulativeEnergyMap

def find_optical_vertical_seam(cumulativeEnergyMap):
    #start from bottom row
    rows,columns = cumulativeEnergyMap.shape
    verticalSeam = numpy.zeros(shape=(rows))
    currentCol = numpy.where(cumulativeEnergyMap== min(cumulativeEnergyMap[rows-1,:]))[1][0]
    verticalSeam[-1] = currentCol
    for i in reversed(range(1,rows)):
        #find the minimum neighor of previous row
        if currentCol == columns-1:
            minNeighbor = min(cumulativeEnergyMap[i-1,currentCol],
                               cumulativeEnergyMap[i-1,currentCol-1])
            if minNeighbor == cumulativeEnergyMap[i-1,currentCol]:
                verticalSeam[i-1] = currentCol
            if minNeighbor == cumulativeEnergyMap[i-1,currentCol-1]:
                verticalSeam[i-1] = currentCol -1
                currentCol = currentCol -1
        elif currentCol == 0:
            minNeighbor = min(cumulativeEnergyMap[i - 1, currentCol],
                               cumulativeEnergyMap[i - 1, currentCol + 1])
            if minNeighbor == cumulativeEnergyMap[i - 1, currentCol]:
                verticalSeam[i-1] = currentCol
            if  minNeighbor == cumulativeEnergyMap[i - 1, currentCol + 1]:
                verticalSeam[i-1] = currentCol + 1
                currentCol = currentCol + 1
        else:
            minNeighbor = min(cumulativeEnergyMap[i - 1, currentCol],
                               cumulativeEnergyMap[i - 1, currentCol + 1],
                               cumulativeEnergyMap[i-1,currentCol - 1])
            if minNeighbor == cumulativeEnergyMap[i - 1, currentCol]:
                verticalSeam[i-1] = currentCol
            if minNeighbor == cumulativeEnergyMap[i - 1, currentCol + 1]:
                verticalSeam[i-1] = currentCol + 1
                currentCol = currentCol + 1
            if minNeighbor == cumulativeEnergyMap[i - 1, currentCol - 1]:
                verticalSeam[i-1] = currentCol - 1
                currentCol = currentCol - 1
    return verticalSeam

def find_optimal_horizontal_seam(cumulativeEnergyMap):
    rows, columns = cumulativeEnergyMap.shape
    horizontalSeam = numpy.zeros(shape=(columns))
    currentRow = numpy.where(cumulativeEnergyMap == min(cumulativeEnergyMap[:,columns-1]))[0][0]
    horizontalSeam[-1] = currentRow

    for i in reversed(range(1,columns)):
        if currentRow == rows-1:
            minNeighbor = min(cumulativeEnergyMap[currentRow,i-1],
                               cumulativeEnergyMap[currentRow - 1,i-1])
            if minNeighbor == cumulativeEnergyMap[currentRow,i-1]:
                horizontalSeam[i-1] = currentRow
            if minNeighbor == cumulativeEnergyMap[currentRow - 1,i-1]:
               horizontalSeam[i-1] = currentRow - 1
               currentRow = currentRow - 1
        elif currentRow == 0:
            minNeighbor = min(cumulativeEnergyMap[currentRow,i-1],
                               cumulativeEnergyMap[currentRow+1, i-1])
            if minNeighbor == cumulativeEnergyMap[currentRow,i-1]:
                horizontalSeam[i - 1] = currentRow
            if minNeighbor == cumulativeEnergyMap[currentRow+1, i-1]:
                horizontalSeam[i - 1] = currentRow + 1
                currentRow = currentRow + 1
        else:
            minNeighbor = min(cumulativeEnergyMap[currentRow, i-1],
                               cumulativeEnergyMap[currentRow+1, i-1],
                               cumulativeEnergyMap[currentRow-1, i-1])
            if minNeighbor == cumulativeEnergyMap[currentRow, i-1]:
               horizontalSeam[i - 1] = currentRow
            if minNeighbor == cumulativeEnergyMap[currentRow+1, i-1]:
                horizontalSeam[i - 1] = currentRow + 1
                currentRow = currentRow + 1
            if minNeighbor == cumulativeEnergyMap[currentRow-1, i-1]:
               horizontalSeam[i - 1] = currentRow - 1
               currentRow = currentRow - 1
    return horizontalSeam


def displaySeam(im, seam, type):
    if type == 'VERTICAL':
       im.setflags(write=1)
       #rows
       for i in range(im.shape[0]):
           #columns
           for j in range(im.shape[1]):
               if float(j) == seam[i]:
                im[i][j] = [255,0,0]
       implot = plt.imshow(im)
       # plt.scatter(x=seam, y=range(len(seam)), c='r', s=1)
       plt.show()


    if type == 'HORIZONTAL':
        im.setflags(write=1)
        for i in range(im.shape[1]):
            for j in range(im.shape[0]):
                if float(j) == seam[i]:
                    im[j][i] = [0,0,255]
        implot = plt.imshow(im)
        # plt.scatter(x=range(len(seam)), y=seam, c='r', s=1)
        plt.show()

def reduceWidth(im, energyImage):
    verticalEnergyMap = cumulative_minimum_energy_map(energyImage, 'VERTICAL')
    verticalSeam =  find_optical_vertical_seam(verticalEnergyMap)

    mask = numpy.ones((im.shape[0], im.shape[1], 3), dtype=numpy.bool)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if float(j) == verticalSeam[i]:
                mask[i,j,:] = False

    im = im[mask].reshape((im.shape[0],im.shape[1]-1, 3))
    return im, energy_image(im)


def reduceHeight(im, energyImage):
    im.setflags(write=1)
    horizontalEnergyMap = cumulative_minimum_energy_map(energyImage, 'HORIZONTAL')
    horizontalSeam = find_optimal_horizontal_seam(horizontalEnergyMap)
    im = numpy.rot90(im, k=1, axes=(0, 1))
    mask = numpy.ones((im.shape[0],im.shape[1],3),dtype=numpy.bool)

    horizontalSeam = horizontalSeam[::-1]
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if float(j) == horizontalSeam[i]:
                mask[i][j][:] = False

    im = im[mask].reshape((im.shape[0],im.shape[1]-1, 3))
    im = numpy.rot90(im,k=3,axes=(0,1))
    return im, energy_image(im)

def reduceHeightDifferentMap(im,energyImage):
        im.setflags(write=1)
        horizontalEnergyMap = cumulative_minimum_energy_map(energyImage, 'HORIZONTAL')
        horizontalSeam = find_optimal_horizontal_seam(horizontalEnergyMap)
        im = numpy.rot90(im, k=1, axes=(0, 1))
        mask = numpy.ones((im.shape[0], im.shape[1], 3), dtype=numpy.bool)

        horizontalSeam = horizontalSeam[::-1]
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                if float(j) == horizontalSeam[i]:
                    mask[i][j][:] = False

        im = im[mask].reshape((im.shape[0], im.shape[1] - 1, 3))
        im = numpy.rot90(im, k=3, axes=(0, 1))
        return im, different_energy_map(im)

def reduceWidthDifferentMap(im, energyImage):
    verticalEnergyMap = cumulative_minimum_energy_map(energyImage, 'VERTICAL')
    verticalSeam =  find_optical_vertical_seam(verticalEnergyMap)

    mask = numpy.ones((im.shape[0], im.shape[1], 3), dtype=numpy.bool)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if float(j) == verticalSeam[i]:
                mask[i,j,:] = False

    im = im[mask].reshape((im.shape[0],im.shape[1]-1, 3))
    return im, different_energy_map(im)

def different_energy_map(im):
    gray_image = numpy.dot(im[..., :3], [0.2989, 0.5870, 0.1140])

    sobel_x = numpy.matrix([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = numpy.matrix([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    x_gradient = scipy.ndimage.convolve(gray_image, weights=sobel_x)
    y_gradient = scipy.ndimage.convolve(gray_image, weights=sobel_y)

    #take square root instead of absolute valued sum
    energyMap = numpy.sqrt(numpy.power(x_gradient,2) + numpy.power(y_gradient,2))
    return energyMap

if __name__ == '__main__':
    img = mpimg.imread('inputSeamCarvingPrague.jpg')
    energyMap = energy_image(img)
    implot = plt.imshow(energyMap, cmap='gray')
    plt.show()
    # verticalMap = cumulative_minimum_energy_map(energyMap, 'VERTICAL')
    # vertical= find_optical_vertical_seam(verticalMap)
    #
    # horizontalMap = cumulative_minimum_energy_map(energyMap, 'HORIZONTAL')
    # horizontal = find_optimal_horizontal_seam(horizontalMap)

    # displaySeam(img,vertical, 'VERTICAL')
    # displaySeam(img,vertical,'VERTICAL')
    # displaySeam(img,horizontal,'HORIZONTAL')

    energyMapSecond = different_energy_map(img)
    implot = plt.imshow(energyMapSecond,cmap='gray')
    plt.show()









