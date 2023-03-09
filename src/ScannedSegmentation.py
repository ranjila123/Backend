import cv2
import numpy as np
# import skimage as sk
import pytesseract
from difflib import SequenceMatcher
from main import load_different_image, main
# from skimage import io
import os

per = 75
pixelThreshold = 1000


#####For Ranjus Printer########  # 





roi4 = [[(925, 350), (3740, 580), 'text', 'National College of Engineering'],
        [(245, 2500), (1245, 2645), 'text', 'ACADEMIC DETAILS'],
        [(258, 4875), (1245, 5015), 'text', 'PERSONAL DETAILS'],
        [(3003, 6385), (3255, 6510), 'text', 'Zone'],
        [(320, 6405), (635, 6530), 'text', 'District']]



    # [(1140, 5450), (2200, 5585), 'text', 'DOB'], 
    # [(3190, 5450), (4123, 5575), 'text', 'DOBAD'],
        # [(2510, 5865), (4300, 5980), 'text', 'Email'],

# for Scan
roi=[[(767, 2908), (1685, 3015), 'text', 'rollno'],
    [(2110, 2900), (2750, 3017), 'text', 'Rank'],
    [(3350, 2900), (4300, 3015), 'text', 'Score'],
    [(870, 3333), (1680, 3448), 'text', 'SymbolNo.'],
    [(2230, 3345), (3050, 3455), 'text', 'PassedYEAR2'],
    [(3670, 3340), (4300, 3450), 'text', 'GPA2'],
    [(870, 3495), (1680, 3615), 'text', 'Board2'],
    [(3390,3492), (4315, 3610), 'text', 'ExtraMaths'],
    [(1040, 3663), (4300, 3785), 'text', 'Institite/College'],
    [(1000, 3908), (1085, 3995), 'Box', 'GovermentSEE'], 
    [(1875, 3908), (1955, 3995), 'Box', 'PrivateSEE'],
    [(870, 4110), (1680, 4225), 'text', 'SymbolNo1'],
    [(2230, 4110), (3050, 4225), 'text', 'Passed year1'], 
    [(3740, 4110), (4300, 4225), 'text', 'GPA/PER1'],
    [(870, 4273), (1685, 4395), 'text', 'Board1'],
    [(2130, 4270), (4300, 4395), 'text', 'School'],
    [(1150, 5120), (4300, 5238), 'text', 'Name'],
    [(1140, 5643), (1267, 5771), 'Box', 'Male'], 
    [(1490, 5643), (1625,  5771), 'Box', 'Female'],
    [(865, 5865), (1835, 5985), 'text', 'Contact'], 
    [(1300, 6191), (2460, 6310), 'text', 'Municipality'], 
    [(2840, 6186), (3030, 6305), 'text', 'Wardno'], 
    [(3345, 6186), (4315, 6305), 'text', 'Tole'],
    [(630, 6386), (1635, 6500), 'text', 'District'], 
    [(1985,  6380), (2890, 6495), 'text', 'Province'], 
    [(3255,  6375), (4310, 6495), 'text', 'Zone'],
    [(302, 1830), (435, 1970), 'Box', 'Civil'], 
    [(1475,  1830), (1605, 1970), 'Box', 'Electrical'], 
    [(302, 2000), (435, 2138), 'Box', 'Computer'], 
    [(1475,  2000), (1605, 2138), 'Box', 'Electronics']]




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



def difference(string1, string2):
    similarity_ratio = SequenceMatcher(None, string1, string2).ratio()
    return similarity_ratio
   
def segment():
    
    croppedDir = "D:/Ranjila Main/Backend-SimpleHTR/croppedDir"
    # Result ='C:/2023_trial_1/SegmentationPart/Results/'

    imgQ = cv2.imread('D:/Ranjila Main/Backend-SimpleHTR/OriginalForm/Original.jpg')

    h, w, c = imgQ.shape


    path = 'Userforms'
    myPicList = os.listdir(path)
    print(myPicList)

    for j, y in enumerate(myPicList):
        img = cv2.imread(path + "/" + y)
        img = cv2.resize(img, (w, h))
   
        imgScan= img

    
        imgShow = imgScan.copy()
        imgMask = np.zeros_like(imgShow)
     


        os.remove(path + "/" + myPicList[0])

        myDataFormValidation = []
        print(f'#################Extracting Data from Form## Form Validation#########')
        for x, r in enumerate(roi4):
            cv2.rectangle(imgMask, ((r[0][0]), r[0][1]), ((r[1][0]), r[1][1]), (0, 255, 0), cv2.FILLED)
            imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)
            # imgShow = cv2.resize(imgShow,(w//7 , h//7))
            # cv2.imshow("Validation",imgShow)
            
            imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
            # cv2.imshow(str(x), imgCrop)
        

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



    if(output1>0.5 and output2>0.5 and output3>0.5):
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
                imageName = "cropped_"+str(x)+".png"
                # sk.io.imsave(imageName, imgCrop)
                cv2.imwrite(os.path.join(croppedDir , imageName), imgCrop)
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
        # recognized = [0,1,2,3,4,5,7,11,12,13,16,17,20,23]
        recognized = main()
        print("From segment")
        print (recognized[0])  
        print (recognized[15])
        print (recognized[16]) 
        print (recognized[17])

        # arrayListforrecognized = [0,1,2,3,4,5,7,11,12,13,16,17,20,23]
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
        # list = recognized + myData

        with open('output.csv', 'a+') as f:
            for data in list:
                f.write((str(data)+','))
            f.write('\n')
        
        return 1
        # return "Image Uploaded Successfully. Waiting for the result...."   
    else:
        print("The given form is not NCE Admisson Form. Please upload correct form.")
        # return "The given form is not NCE Admisson Form. Please upload correct form."
        return 0
    
    # cv2.waitKey(0)

if __name__ == '__main__':
    segment()