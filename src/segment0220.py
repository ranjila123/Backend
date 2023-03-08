import cv2
import numpy as np
import pytesseract
import csv
# import skimage as sk
# from skimage import io
from main import load_different_image, main
from model import DecoderType, Model
from difflib import SequenceMatcher
import os
# import flask
# #####roi for mobile #####
# roi=[[(767, 2917), (1685, 3035), 'text', 'rollno'],
#     [(2147, 2917), (2750, 3035), 'text', 'Rank'],
#     [(3430, 2917), (4300, 3035), 'text', 'Score'],
#     [(925, 3360), (1680, 3480), 'text', 'SymbolNo.'],
#     [(2300, 3360), (3050, 3480), 'text', 'PassedYEAR2'],
#     [(3860, 3360), (4370, 3490), 'text', 'GPA2'],
#     [(925, 3522), (1600, 3645), 'text', 'Board2'],
#     [(3400,3522), (4300, 3645), 'text', 'ExtraMaths'],
#     [(1110, 3695), (4300, 3825), 'text', 'Institite/College'],
#     [(1063, 3945), (1160, 4042), 'Box', 'GovermentSEE'], 
#     [(1983, 3945), (2080, 4042), 'Box', 'PrivateSEE'],
#     [(925, 4145), (1680, 4255), 'text', 'SymbolNo1'], 
#     [(2300, 4145), (3050, 4255), 'text', 'Passed year1'], 
#     [(3880, 4145), (4370, 4255), 'text', 'GPA/PER1'],
#     [(935, 4315), (1600, 4435), 'text', 'Board1'],
#     [(2210, 4315), (4370, 4435), 'text', 'School'],
#     [(1220, 5175), (4370, 5305), 'text', 'Name'],
#     [(1210, 5510), (2330, 5645), 'text', 'DOB'], 
#     [(3220, 5510), (4180, 5645), 'text', 'DOBAD'],
#     [(1210, 5710), (1348, 5855), 'Box', 'Male'], 
#     [(1575, 5710), (1723,  5855), 'Box', 'Female'],
#     [(970, 5937), (1950, 6060), 'text', 'Contact'], 
#     [(2640, 5937), (4370, 6060), 'text', 'Email'],
#     [(1430, 6275), (2600, 6395), 'text', 'Municipality'],
#     [(2980, 6275), (3174, 6395), 'text', 'Wardno'], 
#     [(3495, 6275), (4440, 6395), 'text', 'Tole'],
#     [(665, 6472), (1690, 6595), 'text', 'District'], 
#     [(2100, 6472), (3020, 6595), 'text', 'Province'], 
#     [(3425, 6472), (4370, 6595), 'text', 'Zone'],
#     [(325, 1815), (470, 1956), 'Box', 'Civil'], 
#     [(1565, 1815), (1710, 1966), 'Box', 'Electrical'], 
#     [(325, 1990), (470, 2130), 'Box', 'Computer'], 
#     [(1565, 1990), (1710, 2130), 'Box', 'Electronics']]
# app = flask.Flask(__name__)

# roi4=[[(130, 26), (550, 75), 'text', 'national'],
#     [(38, 355), (180, 376), 'text', 'academic'], 
#     [(38, 700), (185, 720), 'text', 'personal']]

# roi=[[(108, 412), (268, 438), 'number', 'rollno'],
#         [(300, 412), (450, 438), 'number', 'Rank'],
#         [(482, 412), (642, 438), 'number', 'Score'],
#         [(134, 478), (260, 504), 'number', 'SymbolNo.'], 
#         [(318, 478), (462, 500), 'number', 'PassedYEAR2'], 
#         [(550, 478), (652, 500), 'number', 'GPA2'],
#         [(134, 503), (260, 528), 'text', 'Board2'], 
#         [(490, 503), (652, 528), 'number', 'ExtraMaths'],
#         [(150, 530), (652, 546), 'text', 'Institite/College'],
#         [(150, 560), (165, 575), 'Box', 'GovermentSEE'], 
#         [(281, 560), (296, 575), 'Box', 'PrivateSEE'],
#         [(134, 588), (260, 610), 'number', 'SymbolNo1'], 
#         [(320, 588), (462, 610), 'number', 'Passedyear1'], 
#         [(550, 588), (652, 610), 'number', 'GPA/PER1'],
#         [(134, 614), (260, 640), 'text', 'Board1'], 
#         [(318, 614), (652, 640), 'text', 'School'],
#         [(165, 734), (652, 760), 'text', 'Name'], 
#         [(165, 784), (340, 810), 'number', 'DOB'], 
#         [(450, 784), (630, 810), 'number', 'DOBAD'],
#         [(170, 817), (190, 834), 'Box', 'Male'], 
#         [(226, 817), (246, 834), 'Box', 'Female'],
#         [(132, 850), (275, 876), 'number', 'Contact'], 
#         [(370, 850), (652, 876), 'text', 'Email'],
#         [(192, 895), (365, 921), 'text', 'Municipality'], 
#         [(425, 895), (465, 921), 'number', 'Wardno'], 
#         [(485, 895), (652, 921), 'text', 'Tole'], 
#         [(90, 927), (250, 953), 'text', 'disctrict'], 
#         [(300, 927), (460,953), 'text', 'Province'], 
#         [(495, 927), (652, 953), 'text', 'Zone'],
#         [(45, 258), (67, 282), 'Box', 'Civil'], 
#         [(220, 258), (242, 282), 'Box', 'Electrical'], 
#         [(45, 283), (67, 307), 'Box', 'Computer'], 
#         [(220, 283), (242, 307), 'Box', 'Electronics'], 
#         [(550, 177), (652, 310), 'Image', 'Photo']]

def difference(string1, string2):
    similarity_ratio = SequenceMatcher(None, string1, string2).ratio()
    return similarity_ratio


# @app.route('/text')
def segment():
    per =25
    pixelThreshold = 150

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
        # good=[]
        # for m,n in matches:
        #     if m.distance <0.7*n.distance:
        #         good.apppend(m)

    good = matches[:int(len(matches)*(per/100))]
        #<---------------condtion if the form is ours or not----------->
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:400], None, flags=2)
    # imgMatch = cv2.resize(imgMatch, (w//6, h//7))
        #cv2.imshow(y, imgMatch)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, _ =cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)

    imgScan = cv2.warpPerspective(img, M, (w, h))
    # imgScan = cv2.resize(imgScan, (w//7, h//7))
    # cv2.imshow(y + "lol", imgScan)

    h1, w1, c = imgScan.shape
        # print(h1)
        # print(w1)
    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)
        
        #cv2.imshow('mask',imgMask)

    # os.remove(path + "/" + myPicList[0])



    # myDataFormValidation = []
    # print(f'#################Extracting Data from Form## Form Validation#########')
    # for x, r in enumerate(roi4):
    #         cv2.rectangle(imgMask, ((r[0][0]), r[0][1]), ((r[1][0]), r[1][1]), (0, 255, 0), cv2.FILLED)
    #         imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)
    #         imgShow = cv2.resize(imgShow,(w//7 , h//7))
    #         cv2.imshow("Validation",imgShow)
            
    #         imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
    #         # cv2.imshow(str(x), imgCrop)
        

    #         if r[2] == 'text':
    #             print('{}:{}'.format(r[3], pytesseract.image_to_string(imgCrop)))
    #             myDataFormValidation.append(pytesseract.image_to_string(imgCrop))
    
    # # Driver code to call a function
    # usr_str1 = 'National College of Engineering'
    # usr_str2 = myDataFormValidation[0]
    # output1 = difference(usr_str1, usr_str2)
    # print(output1)

    # usr_str3 = 'ACADEMIC DETAILS'
    # usr_str4 = myDataFormValidation[1]
    # output2 = difference(usr_str3, usr_str4)
    # print(output2)

    # usr_str5 = 'PERSONAL DETAILS'
    # usr_str6 = myDataFormValidation[2]
    # output3 = difference(usr_str5, usr_str6)
    # print(output3)

    # if(output1>0.5 or output2>0.5 or output3>0.5):
    #     print("Right form")
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
                imageName = "cropped_"+str(x)+".png"
                # sk.io.imsave(imageName, imgCrop)
                cv2.imwrite(os.path.join(croppedDir , imageName), imgCrop)
                # cv2.imwrite(imageName, imgCrop)    
                    #cv2.imshow(str(x), imgCrop)
                
            if r[2] == 'Box':
                imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGRA2GRAY)
                imgThresh = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY_INV)[1]
                totalPixels = cv2.countNonZero(imgThresh)
                    #print(totalPixels)
                if totalPixels > pixelThreshold : totalPixels = 1;
                else:totalPixels = 0
                print(f'{r[3]}:{totalPixels}')
                myData.append(totalPixels)
        
        # main()
    Number = [0,1,2,3,4,5,7,11,12,13,16,17,20,23]
    recognized = main()
    print("From segment")
    print (recognized[0])   

    arrayListforNumber = [0,1,2,3,4,5,7,11,12,13,16,17,20,23]
    arrayListForText = [6,8,14,15,21,22,24,25,26,27]
    arrayListforBox = [9,10,18,19,28,29,30,31]
    list = [None] * 33
        # for i in range(33):
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
    list[17] = recognized[15]
    list[18] = recognized[16]
    list[19] = myData[2]
    list[20] = myData[3]
    list[21] = recognized[17]
    list[22] = recognized[18]
    list[23] = recognized[19]
    list[24] = recognized[20]
    list[25] = recognized[21]
    list[26] = recognized[22]
    list[27] = recognized[23]
    list[28] = myData[4]
    list[29] = myData[5]
    list[30] = myData[6]
    list[31] = myData[7]
    list[32] = "Photo"


        #list = recognized + myData

        # header = ["Roll no.","Rank","Score","Symbol No.","Passed YEAR2","GPA2","Board2","ExtraMaths","Institite/College","GovermentSEE","PrivateSEE","SymbolNo1","Passedyear1","GPA/PER1","Board1","School","Name","DOB","DOBAD","Male","Female","Contact","Email","Municipality","Wardno","Tol no","Disctrict","Province","Zone","Civil","Electrical","Computer","Electronics","Photo"] 
    with open('output.csv', 'a+') as f:
            # writer = csv.writer(f)
            # writer.writerow(header)
            for data in list:
                f.write((str(data)+','))
            f.write('\n')
        
    # return 1
        # return "Image Uploaded Successfully. Waiting for the result...."   
# else:
#     print("The given form is not NCE Admisson Form. Please upload correct form.")
#         # return "The given form is not NCE Admisson Form. Please upload correct form."
#         return 0
    
    # cv2.waitKey(0)

if __name__ == '__main__':
    segment()