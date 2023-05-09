import numpy as np
# import skimage.io as io
import cv2



class preprocess:

    def __init__(self, img) -> None:
        self.img = img

    
    def remove_lines():
        imgs=[]
        for i in range(0, 29):
            list=[6,8,14,15,16,22,23,25,26,27,28]
            if i in list:
                imgs.append((cv2.imread("../croppedDir/cropped_{}.png".format(i))))
                # print(imgs)
       
        for i in imgs:
            #This detected all horizontal and vertical

            img = cv2.imread(i)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            img = cv2.bitwise_not(img)
            th2 = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
            # cv2.imshow("th2", th2)
            # cv2.imwrite("th2.jpg", th2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            horizontal = th2
            vertical = th2
            rows = horizontal.shape[1]
            cols = horizontal.shape[0]
            horizontalsize = cols / 30
            horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
            horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
            horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
            cv2.imshow("horizontal", horizontal)
            cv2.imwrite("horizontal.jpg", horizontal)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            verticalsize = rows / 30
            verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
            vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
            vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
            ve = vertical
            cv2.imshow("vertical", vertical)
            cv2.imwrite("vertical.jpg", vertical)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            vertical = cv2.bitwise_not(vertical)
            cv2.imshow("vertical_bitwise_not", vertical)
            cv2.imwrite("vertical_bitwise_not.jpg", vertical)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            #step1
            edges = cv2.adaptiveThreshold(vertical,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,-2)
            cv2.imshow("edges", edges)
            cv2.imwrite("edges.jpg", edges)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            #step2
            kernel = np.ones((2, 2), dtype = "uint8")
            dilated = cv2.dilate(edges, kernel)
            cv2.imshow("dilated", dilated)
            cv2.imwrite("dilated.jpg", dilated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # step3
            smooth = vertical.copy()

            #step 4
            smooth = cv2.blur(smooth, (4,4))
            cv2.imshow("smooth", smooth)
            cv2.imwrite("smooth.jpg", smooth)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            #step 5
            (rows, cols) = np.where(img == 0)
            vertical[rows, cols] = smooth[rows, cols]
            vertical1=cv2.bitwise_not(vertical)
            cv2.imshow("vertical_final", vertical)
            cv2.imwrite("vertical_final.jpg", vertical)
            mask = cv2.bitwise_or(vertical1, horizontal) #vertical1 or ve 
            #75-83 commented but necessary
            #mask = cv2.imread('vertical.jpg',0)
            dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
            cv2.imwrite("original_unmasked_lastTest1.png", dst)

            inverted=cv2.bitwise_not(dst)
            cv2.imshow("invertes", inverted)
            cv2.imwrite("Inverts1.png", inverted) #inverted has vertical mask


            #92-100 lines added here 
            # # Resize horizontal image to match the size of img
            # horizontal = cv2.resize(horizontal, (img.shape[1], img.shape[0]))
            # cv2.imshow("how",horizontal)
            # Apply bitwise operation to mask the original image with horizontal and vertical images
            #masked = cv2.bitwise_and(img, img, mask=horizontal)
            #masked = cv2.bitwise_and(masked, masked, mask=vertical)






            # mask = cv2.imread('inverted.png',0)
            # dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
            # cv2.imshow("final", dst)

            # mask = cv2.imread('horizontal.jpg',0)
            # dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
            # cv2.imwrite("original_unmasked_lastTest1.png", dst)
            # inverted=cv2.bitwise_not(dst)
            # cv2.imshow("invertes", inverted)


            # #inverse the image, so that lines are black for masking
            # horizontal_inv = cv2.bitwise_not(horizontal)
            # #perform bitwise_and to mask the lines with provided mask
            # masked_img = cv2.bitwise_and(img, img, mask=horizontal_inv)
            # #reverse the image back to normal
            # masked_img_inv = cv2.bitwise_not(masked_img)
            # cv2.imshow("masked img", masked_img_inv)
            # cv2.imwrite("result2.jpg", masked_img_inv)

            # #inverse the image, so that lines are black for masking
            # vertical_inv = cv2.bitwise_not(vertical)
            # #perform bitwise_and to mask the lines with provided mask
            # masked_img1 = cv2.bitwise_and(img, img, mask=vertical_inv)
            # #reverse the image back to normal
            # masked_img_inv1 = cv2.bitwise_not(masked_img1)
            # cv2.imshow("masked img", masked_img_inv1)
            # cv2.imwrite("result3.jpg", masked_img_inv1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    
    def preprocess(self, img):

        img = self.cv2.imread('D:\Ranjila Main\Backend-SimpleHTR\croppedDir\cropped_0.png', 0)
        #cv2.imshow('pre',img)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        alpha = 1.5 # contrast control
        beta = 0 # brightness control
        new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        #cv2.imshow('pre1',new_image)
        #cv2.waitKey()
        cv2.imwrite('output/6.png', new_image)
        # io.imsave('output.png', img)
