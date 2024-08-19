import numpy as np
import matplotlib.pyplot as plt
import cv2
########### data 2 ###########

image = np.array([[88, 99, 148],
                  [110, 130, 165],
                  [150, 140, 180]] ,dtype=np.uint8 ) 

############ data 1 ##################
# image = np.array([[100, 84, 240],
#                   [10, 75, 250],
#                   [140, 158, 11]], dtype=np.uint8)

######### My histogtam equal #############
def Histogramequalize(image):
    histogram = np.zeros(256, dtype=int)
    for pixel in image.flatten():
        histogram[pixel] += 1

    cdf = np.zeros(256, dtype=int)
    cdf[0] = histogram[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + histogram[i]

    cdf_min = np.min(cdf[cdf > 0])
    cdf_max = np.max(cdf)
    cdf_normalized = ((cdf - cdf_min) / (cdf_max - cdf_min) * 255).astype(np.uint8)

    equalized_image = np.zeros_like(image, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            equalized_image[i, j] = cdf_normalized[image[i, j]]
    
    return equalized_image

#########  show my def #######
plt.imshow(Histogramequalize(image), cmap='gray')
plt.axis('off')  
plt.show()
##################################

########## show cv2.equalizeHist def #########
equalizedd_image = cv2.equalizeHist(image)
plt.imshow(equalizedd_image, cmap='gray')
plt.axis('off') 
plt.show()
###########################################



############ CLAHE ###########
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(6, 6))
clahe2 = clahe.apply(image)
plt.imshow(clahe2, cmap='gray')
plt.axis('off') 
plt.show()
##########################################