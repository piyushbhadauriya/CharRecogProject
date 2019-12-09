import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np
from os import listdir

from skimage import color,io,morphology,filters,util,transform 
from skimage.measure import label, regionprops

IMAGE_DIR = os.path.dirname(__file__)

###### grayscale = morphology.erosion(img1)

def getImageList():
    global IMAGE_DIR
    ImageFilesList = []
    Allfiles = listdir(IMAGE_DIR)
    for afile in Allfiles:
        if ".png" in afile or ".PNG" in afile:
            ImageFilesList.append(afile)
    return ImageFilesList

def plot(imageList):
    fig, axes = plt.subplots(len(imageList))
    
    ax = axes.ravel()
    for i in range(len(imageList)):
        plt.axis('off')
        ax[i].imshow(imageList[i],cmap='gray')
    # ax[1].imshow(newimage,cmap='gray')

    # fig.tight_layout()
    plt.savefig('Image/display.png')
    image = Image.open('Image/display.png')
    image.show()


def process_Image(imageName):
    image = io.imread(os.path.join(IMAGE_DIR,imageName),as_gray=True)
    # RGB image to grayscale 
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            image = color.rgba2rgb(image)
        image = color.rgb2gray(image)
        # print("image size:",image.shape)
    
    # Erosion highlights darker areas in image 
    tmp = morphology.erosion(image)
    
    # Segmentation
    tmp = util.invert(tmp)
    thresh = filters.threshold_otsu(tmp)
    bw = tmp > thresh
    label_image = label(bw)
    image_label_overlay = color.label2rgb(label_image, image=image)
    # plt.imshow(image_label_overlay,cmap='gray')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay,cmap='gray')

    boxes = []
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 100:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
              
            chars = region.image
            chars = util.img_as_ubyte(chars)
            pMin = min(chars.shape)
            # pMax = max(chars.shape)
            if pMin < 28 :
                newchar = np.copy(chars)
                r = chars.shape[0]
                c = chars.shape[1]
                if c == pMin:
                    diff = r-c
                    diffhalf = diff//2
                    myChar = np.concatenate((np.zeros((r,diffhalf)), newchar, np.zeros((r,diff - diffhalf))),axis=1)
                elif r == pMin:
                    diff = c-r
                    diffhalf = diff//2
                    myChar = np.concatenate((np.zeros((diffhalf,c)), newchar, np.zeros((diff - diffhalf,c))),axis=0)
                chars = myChar
            
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
                     
            chars = transform.resize(chars,(28,28),mode='constant')
            chars = np.copy(chars)
            chars[chars > 0] = 255
            chars[chars <= 0] = 0

            # chars = util.img_as_ubyte(chars)

            # chars = util.invert(chars)
            # print(np.ptp(chars,axis=1))
            
            boxes.append(chars)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    # print(boxes[1])
    return boxes


# if __name__ == "__main__":
#     images = []
#     images.append(io.imread(os.path.join(IMAGE_DIR,'img-1.png'),as_gray=True))
#     # images.append(io.imread(os.path.join(IMAGE_DIR,imageName),as_gray=True))
#     # images.append(io.imread(os.path.join(IMAGE_DIR,'img-2.png')))
#     for image in images:
#         newimages=[image]
#         print("image size:",image.shape)
        
#         # RGB image to grayscale 
#         if len(image.shape) == 3:
#             if image.shape[2] == 4:
#                 image = color.rgba2rgb(image)
#             image = color.rgb2gray(image)
#             print("image size:",image.shape)
#         # else :
#             # print("image format not supported")
            

#         # erosion highlights darker areas in image 
#         tmp = morphology.erosion(image)
#         newimages.append(tmp)
#         # tmp = morphology.opening(image)
#         # newimages.append(tmp)

#         tmp = util.invert(tmp)
#         # np.sum(array,axis=1).tolist()
#         newimages.append(tmp)
#         # # histogram
#         # hist, hist_centers = exposure.histogram(tmp)
#         # print(hist[0])
#         # print(type(hist))
#         # print(hist.shape)
#         # plt.plot(hist_centers[1:], hist[1:], lw=2)
#         # plt.show()

#         # # apply threshold
#         thresh = filters.threshold_otsu(tmp)
#         bw = tmp > thresh
#         # plt.imshow(bw)
#         # remove artifacts connected to image border
#         # cleared = clear_border(bw)

#         # label image regions
#         label_image = label(bw)
#         image_label_overlay = color.label2rgb(label_image, image=image)

#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.imshow(image_label_overlay,cmap='gray')
#         boxes = []
#         for region in regionprops(label_image):
#             # take regions with large enough areas
#             if region.area >= 100:
#                 # i = i+1
#                 # draw rectangle around segmented coins
#                 minr, minc, maxr, maxc = region.bbox
#                 rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
#                                         fill=False, edgecolor='red', linewidth=2)
#                 # boxes.append(region.image)
#                 print("image size:",region.image.shape)
#                 # n = max(region.image.shape)

#                 chars = transform.resize(region.image,(28,28),mode='constant')
#                 ax.add_patch(rect)
#                 chars = util.img_as_ubyte(chars)
#                 boxes.append(chars)
#                 # images.append(tmp[minr:maxr,minc:maxc])
#         # print(i)
#         ax.set_axis_off()
#         plt.tight_layout()
#         plt.show()
#         # newimages.append(thresh)
#         # markers = np.zeros(tmp.shape, dtype=np.uint)
#         # markers[tmp < -0.95] = 1
#         # markers[tmp > 0.95] = 2
#         # # newimages.append(markers)
#         # # newimages.append(segmentation.random_walker(tmp, markers, beta=10, mode='bf'))
#         print(boxes[0])
#         plot(boxes[:6])

#     # def resize(image):

#     #     print("image size:",image.shape)
#     #     w = image.shape[0]
#     #     h = image.shape[1]
#     #     n = max(image.shape)
#     #     print("Max:",n)
        
#     #     if w == n:

#     #     image.
