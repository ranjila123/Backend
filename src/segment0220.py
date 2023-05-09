import cv2
import numpy as np
import pytesseract
from main import load_different_image, main
from model import DecoderType, Model
from difflib import SequenceMatcher
import os


# #####roi for mobile #####
roi4=[[(1100, 300), (3900, 545), 'text', 'National College'],
    [(267, 4900), (1325, 5045), 'text', 'Personal Details'],
    [(267, 2490), (1325, 2630), 'text', 'Academic Details'],
    [(3150, 6472), (3415, 6595), 'text', 'Zone'],
    [(265, 6472), (655, 6595), 'text', 'District']]

roi=[[(775, 2915), (1680, 3033), 'text', 'rollno'],
    [(2160, 2915), (2745, 3033), 'text', 'Rank'],
    [(3430, 2910), (4300, 3025), 'text', 'Score'],
    [(930, 3355), (1680, 3480), 'text', 'SymbolNo.'],
    [(2310, 3355), (3050, 3470), 'text', 'PassedYEAR2'],
    [(3900, 3350), (4370, 3470), 'text', 'GPA2'],
    [(935, 3520), (1600, 3645), 'text', 'Board2'],
    [(3425,3520), (4300, 3640), 'text', 'ExtraMaths'],
    [(1110, 3690), (4300, 3820), 'text', 'Institite/College'],
    [(1063, 3935), (1160, 4045), 'Box', 'GovermentSEE'], 
    [(1983, 3935), (2080, 4045), 'Box', 'PrivateSEE'],
    [(935, 4140), (1680, 4260), 'text', 'SymbolNo1'], 
    [(2300, 4140), (3050, 4260), 'text', 'Passed year1'], 
    [(3925, 4142), (4370, 4257), 'text', 'GPA/PER1'],
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



def difference(string1, string2):
    similarity_ratio = SequenceMatcher(None, string1, string2).ratio()
    return similarity_ratio


# @app.route('/text')
def segment():
    per =25
    pixelThreshold = 10000

    croppedDir = "D:/Ranjila Main/Backend-SimpleHTR/croppedDir"
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    # pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR-old\\tesseract.exe'


    imgQ = cv2.imread('D:/Ranjila Main/Backend-SimpleHTR/OriginalForm/Original.jpg')
    h, w, c = imgQ.shape
    
    # imgQ = cv2.resize(imgQ,(w//7,h//7),interpolation = cv2.INTER_AREA)

    #cv2.imshow("query",imgQ)

    orb = cv2.ORB_create(7000)
    kp1, des1 = orb.detectAndCompute(imgQ, None)
    impkp1 = cv2.drawKeypoints(imgQ,kp1,None)
    # cv2.imwrite('keypoints.png',impkp1)
    #cv2.imshow("key points",impkp1)

    path = 'D:/Ranjila Main/Backend-SimpleHTR/src/UserForms'
    myPicList = os.listdir(path)
    print(myPicList)
    count=255


    img = cv2.imread(path + "/" + myPicList[0])
        #img = cv2.resize(img, (w//7, h//7))
        #cv2.imshow(y, img)
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
    imgMatch1 = cv2.resize(imgMatch, (w//6, h//7))
    # cv2.imshow(y, imgMatch1)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, _ =cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)

    imgScan = cv2.warpPerspective(img, M, (w, h))
    imgScan2 = cv2.resize(imgScan, (w//7, h//7))
    # cv2.imshow( "lol", imgScan2)

    # h1, w1, c = imgScan.shape
        # print(h1)
        # print(w1)
    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)
        
        #cv2.imshow('mask',imgMask)

    os.remove(path + "/" + myPicList[0])



    myDataFormValidation = []
    print(f'#################Extracting Data from Form## Form Validation#########')
    for x, r in enumerate(roi4):
        cv2.rectangle(imgMask, ((r[0][0]), r[0][1]), ((r[1][0]), r[1][1]), (0, 255, 0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)
    #         imgShow = cv2.resize(imgShow,(w//7 , h//7))
    #         cv2.imshow("Validation",imgShow)
            
        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        # cv2.imshow(str(x), imgCrop)
        # cv2.waitKey(0)
        

        if r[2] == 'text':
            print('{}:{}'.format(r[3], pytesseract.image_to_string(imgCrop)))
            myDataFormValidation.append(pytesseract.image_to_string(imgCrop))
    
    # Driver code to call a function
    usr_str1 = 'National College of Engineering'
    usr_str2 = myDataFormValidation[0]
    output1 = difference(usr_str1, usr_str2)
    print(output1)

    usr_str3 = 'ACADEMIC DETAILS'
    usr_str4 = myDataFormValidation[1]
    output2 = difference(usr_str3, usr_str4)
    print(output2)

    usr_str5 = 'PERSONAL DETAILS'
    usr_str6 = myDataFormValidation[2]
    output3 = difference(usr_str5, usr_str6)
    print(output3)

    usr_str7 = 'Zone'
    usr_str8 = myDataFormValidation[3]
    output4 = difference(usr_str7, usr_str8)
    print(output4)

    usr_str9 = 'District'
    usr_str10 = myDataFormValidation[4]
    output5 = difference(usr_str9, usr_str10)
    print(output5)

    if(output1>0.4 and output2>0.4 and output3>0.4 and output4>0.4 and output5>0.4):
        print("Right form")
        myData = []
        print(f'#################Extracting Data from Form ###########')
        for x, r in enumerate(roi):
            cv2.rectangle(imgMask, ((r[0][0]), r[0][1]), ((r[1][0]), r[1][1]), (0, 255, 0), cv2.FILLED)
            imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

            imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
# 
            # imageName = "cropped_"+str(x)+".png"
            # cv2.imwrite(os.path.join(croppedDir , imageName), imgCrop)

            if r[2] == 'text':
                # imageName = croppedDir+"cropped_"+str(x)+".png"
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
                imageName = "cropped_"+str(x)+".png"
                # sk.io.imsave(imageName, imgCrop)
                cv2.imwrite(os.path.join(croppedDir , imageName), dst)
                # cv2.imwrite(imageName, imgCrop)    
                    #cv2.imshow(str(x), imgCrop)
                
            if r[2] == 'Box':
                imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGRA2GRAY)
                imgThresh = cv2.threshold(imgGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
                totalPixels = cv2.countNonZero(imgThresh)
                print(totalPixels)
                if totalPixels > pixelThreshold : totalPixels = 1;
                else:totalPixels = 0
                print(f'{r[3]}:{totalPixels}')
                myData.append(totalPixels)
        
        # main()
    # Number = [0,1,2,3,4,5,7,11,12,13,16,17,20,23]
        recognized = main()
        print("From segment")
        print (recognized[0])   

    # arrayListforNumber = [0,1,2,3,4,5,7,11,12,13,16,17,20,23]
    # arrayListForText = [6,8,14,15,21,22,24,25,26,27]
    # arrayListforBox = [9,10,18,19,28,29,30,31]
        list = [None] * 30
        # for i in range(31):
        list[0] = recognized[0]
        list[1] = recognized[1]
        list[2] = recognized[2]
        list[3] = recognized[3]
        list[4] = recognized[4]
        list[5] = recognized[5]
        list[6] = recognized[6]
        list[7] = recognized[7]
        list[8] = recognized[8]
        list[9] = myData[0]
        list[10] = myData[1]
        list[11] = recognized[9]
        list[12] = recognized[10]
        list[13] = recognized[11]
        list[14] = recognized[12]
        list[15] = recognized[13]
        list[16] = recognized[14]
        list[17] = myData[2]
        list[18] = myData[3]
        list[19] = recognized[15]
        list[20] = recognized[16]
        list[21] = recognized[17]
        list[22] = recognized[18]
        list[23] = recognized[19]
        list[24] = recognized[20]
        list[25] = recognized[21]
        list[26] = myData[4]
        list[27] = myData[5]
        list[28] = myData[6]
        list[29] = myData[7]

        print(list)

        #   list = recognized + myData

        # header = ["Roll no.","Rank","Score","Symbol No.","Passed YEAR2","GPA2","Board2","ExtraMaths","Institite/College","GovermentSEE","PrivateSEE","SymbolNo1","Passedyear1","GPA/PER1","Board1","School","Name","DOB","DOBAD","Male","Female","Contact","Email","Municipality","Wardno","Tol no","Disctrict","Province","Zone","Civil","Electrical","Computer","Electronics","Photo"] 
        with open('output.csv', 'a+') as f:
            # writer = csv.writer(f)
            # writer.writerow(header)
            for data in     list:
                f.write((str(data)+','))
            f.write('\n')
        
        return 1
        # return "Image Uploaded Successfully. Waiting for the result...."   
    else:
        print("The given form is not NCE Admisson Form. Please upload correct form.")
#         # return "The given form is not NCE Admisson Form. Please upload correct form."
        return 0
    


if __name__ == '__main__':
    segment()