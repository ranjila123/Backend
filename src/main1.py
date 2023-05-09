import cv2
import numpy as np
# import skimage as sk
# from skimage import io
import os

per = 25
pixelThreshold = 500 
# Result ='C:/2023_trial_1/SegmentationPart/Results/'
croppedDir = "D:/Ranjila Main/Backend-SimpleHTR/croppedDir"

# croppedDir='C:/2023_trial_1/SegmentationPart/cropped/'
##For angel rai
roi4=[[(1100, 300), (3900, 545), 'text', 'National College'],
    [(267, 4900), (1325, 5045), 'text', 'Personal Details'],
    [(267, 2490), (1325, 2630), 'text', 'Academic Details'],
    [(3190, 6472), (3415, 6595), 'text', 'Zone'],
    [(265, 6472), (655, 6595), 'text', 'District']]
#]
# roi = []
roi=[[(775, 2915), (1680, 3033), 'text', 'rollno'],
    [(2160, 2915), (2745, 3033), 'text', 'Rank'],
    [(3430, 2915), (4300, 3025), 'text', 'Score'],
    [(930, 3355), (1680, 3480), 'text', 'SymbolNo.'],
    [(2310, 3355), (3050, 3470), 'text', 'PassedYEAR2'],
    [(3900, 3355), (4370, 3470), 'text', 'GPA2'],
    [(935, 3520), (1600, 3645), 'text', 'Board2'],
    [(3425,3520), (4300, 3640), 'text', 'ExtraMaths'],
    [(1110, 3690), (4300, 3820), 'text', 'Institite/College'],
    [(1063, 3935), (1160, 4045), 'Box', 'GovermentSEE'], 
    [(1983, 3935), (2080, 4045), 'Box', 'PrivateSEE'],
    [(935, 4140), (1680, 4260), 'text', 'SymbolNo1'], 
    [(2300, 4140), (3050, 4260), 'text', 'Passed year1'], 
    [(3925, 4148), (4370, 4257), 'text', 'GPA/PER1'],
    [(935, 4310), (1600, 4440), 'text', 'Board1'],
    [(2220, 4310), (4370, 4440), 'text', 'School'],
    [(1220, 5170), (4370, 5310), 'text', 'Name'],
    [(1210, 5705), (1348, 5850), 'Box', 'Male'], 
    [(1575, 5705), (1723,  5850), 'Box', 'Female'],
    [(970, 5940), (1950, 6075), 'text', 'Contact'], 
    [(1430, 6275), (2600, 6410), 'text', 'Municipality'],
    [(3010, 6275), (3190, 6415), 'text', 'Wardno'], 
    [(3495, 6275), (4440, 6450), 'text', 'Tole'],
    [(675, 6465), (1690, 6610), 'text', 'District'], 
    [(2110, 6470), (3020, 6610), 'text', 'Province'], 
    [(3465, 6470), (4370, 6650), 'text', 'Zone'],
    [(325, 1810), (470, 1960), 'Box', 'Civil'], 
    [(1565, 1810), (1710, 1960), 'Box', 'Electrical'], 
    [(325, 1985), (470, 2125), 'Box', 'Computer'], 
    [(1565, 1985), (1710, 2125), 'Box', 'Electronics']]

# roi=[[(108, 412), (268, 438), 'text', 'rollno'],
#     [(300, 412), (450, 438), 'text', 'Rank'],
#     [(482, 412), (642, 438), 'text', 'Score'],
#     [(134, 478), (260, 504), 'text', 'SymbolNo.'], 
#     [(318, 478), (462, 500), 'text', 'PassedYEAR2'], 
#     [(550, 478), (652, 500), 'text', 'GPA2'],
#     [(134, 503), (260, 528), 'text', 'Board2'], 
#     [(490, 503), (652, 528), 'text', 'ExtraMaths'],
#     [(150, 530), (652, 546), 'text', 'Institite/Colledge'],
#     [(150, 560), (165, 575), 'Box', 'GovermentSEE'], 
#     [(281, 560), (296, 575), 'Box', 'PrivateSEE'],
#     [(134, 588), (260, 610), 'text', 'SymbolNo1'], 
#     [(320, 588), (462, 610), 'text', 'Passedyear1'], 
#     [(550, 588), (652, 610), 'text', 'GPA/PER1'],
#     [(134, 614), (260, 640), 'text', 'Board1'], 
#     [(318, 614), (652, 640), 'text', 'School'],
#     [(165, 734), (652, 760), 'text', 'Name'], 
#     [(165, 784), (340, 810), 'text', 'DOB'], 
#     [(450, 784), (630, 810), 'text', 'DOBAD'],
#     [(170, 817), (190, 834), 'Box', 'Male'], 
#     [(226, 817), (246, 834), 'Box', 'Female'],
#     [(132, 850), (275, 876), 'text', 'Contact'], 
#     [(370, 850), (652, 876), 'text', 'Email'],
#     [(192, 895), (365, 921), 'text', 'Municipality'], 
#     [(425, 895), (465, 921), 'text', 'Wardno'], 
#     [(485, 895), (652, 921), 'text', 'telno'], 
#     [(90, 927), (250, 953), 'text', 'disctrict'], 
#     [(300, 927), (460,953), 'text', 'Province'], 
#     [(495, 927), (652, 953), 'text', 'Zone'],
#     [(45, 258), (67, 282), 'Box', 'Civil'], 
#     [(220, 258), (242, 282), 'Box', 'Electrical'], 
#     [(45, 283), (67, 307), 'Box', 'Computer'], 
#     [(220, 283), (242, 307), 'Box', 'Electronics'], 
#     [(550, 177), (652, 310), 'Imahe', 'Photo']]

########################### Original ##############################################
# imgQ = cv2.imread('Query.jpg')
imgQ = cv2.imread('D:/Ranjila Main/Backend-SimpleHTR/OriginalForm/Original.jpg')

h, w, c = imgQ.shape
print(h)
print(w)
# imgQ = cv2.resize(imgQ,(w//7,h//7),interpolation = cv2.INTER_AREA)
# print("Query divided")
# print(h//7)
# print(w//7)
# cv2.imshow("query",imgQShow)
orb = cv2.ORB_create(7000)
kp1, des1 = orb.detectAndCompute(imgQ, None)

impkp1 = cv2.drawKeypoints(imgQ,kp1,None)
impkp1 = cv2.resize(impkp1,(w//7,h//7),interpolation = cv2.INTER_AREA)
# cv2.imshow("key points",impkp1)


########################### Query ##############################################
path = 'UserForms'
myPicList = os.listdir(path)
print(myPicList)

for j, y in enumerate(myPicList):
    img = cv2.imread(path + "/" + y)
    # img = cv2.resize(img, (w, h))
    # cv2.imshow(y, img)
    # print(j)
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = True)
    matches = bf.match(des2, des1)
    matches = sorted(matches, key=lambda x: x.distance)
    good=[]
    # for m,n in matches:
    #     if m.distance <0.7*n.distance:
    #         good.apppend(m)

    good = matches[:int(len(matches)*(per/100))]
    #<---------------condtion if the form is ours or not----------->
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:400], None, flags=2)
    imgShow1 = cv2.resize(imgMatch, (w//7, h//7))
    # cv2.imshow(y, imgShow1)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, _ =cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)

    imgScan = cv2.warpPerspective(img, M, (w, h))
    imgScan2 = cv2.resize(imgScan, (w//7 ,h//7))
    cv2.imshow(y + "lol", imgScan2)

    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)
     
    
    # imageName = Result +"Result"+str(y)
    # sk.io.imsave(imageName, imgScan)
    # cv2.imwrite(imageName,imgScan)



  

# imageName = croppedDir+"cropped_"+str(x)+".png"
# sk.io.imsave(imageName, imgCrop)
# print('cropped')
# cv2.imshow(str(x), imgCrop)
    myData = []
    print(f'#################Extracting Data from Form {j}###########')
    for x, r in enumerate(roi):
        cv2.rectangle(imgMask, ((r[0][0]), r[0][1]), ((r[1][0]), r[1][1]), (0, 255, 0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        # cv2.imshow(str(x), imgCrop)
        
       

        print('cropped')

        if r[2] == 'text':
                # print('{}:{}'.format(r[3], pytesseract.image_to_string(imgCrop)))
                if len(imgCrop.shape) != 2:
                    gray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
                else:
                    gray = imgCrop

                gray = cv2.bitwise_not(gray)
                bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

                horizontal = np.copy(bw)

                cols = horizontal.shape[1]
                horizontal_size = cols // 30

                horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

                horizontal = cv2.erode(horizontal, horizontalStructure)
                horizontal = cv2.dilate(horizontal, horizontalStructure)

                # mask = cv2.imread('horizontal_lines_extracted5.png',0)
                # Create a mask array directly in memory
                mask = np.zeros(imgCrop.shape[:2], dtype=np.uint8)
                mask[horizontal > 0] = 255

                dst = cv2.inpaint(imgCrop, mask, 3, cv2.INPAINT_TELEA)
                # cv2.imwrite("original_unmasked5.png", dst)
                imageName = croppedDir+"cropped_"+str(x)+".png"
                # sk.io.imsave(imageName, imgCrop)
                cv2.imwrite(imageName,dst)
                myData.append("text")
                print('text')

        if r[2] == 'Box':
             imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGRA2GRAY)
             imgThresh = cv2.threshold(imgGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
             totalPixels = cv2.countNonZero(imgThresh)
             #print(totalPixels)
             if totalPixels > pixelThreshold : totalPixels = 1;
             else:totalPixels = 0
             print(f'{r[3]}:{totalPixels}')
             myData.append(totalPixels)
             
    # cv2.putText(imgShow, str(myData[x]), (r[0][0], r[0][1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    with open('output.csv', 'a+') as f:
        for data in myData:
            f.write((str(data)+','))
        f.write('\n')
    # cv2.imshow(y,imgShow)

cv2.waitKey(0)

