import os

f_train = open('my_train.txt', 'w')
f_test = open('my_test.txt', 'w')

#size_1 = 34
#size_2 = 38


cou = 0
for dirPath, dirNames, fileNames in os.walk('crop\\1'):
    #print(dirPath, dirNames, fileNames)
    s = 'D:\\Dataset\\Action\\my_dataset\\' + dirPath + ' ' + str(len(fileNames)) + ' 1\n'
    if len(fileNames) != 0:
        if cou % 3 == 0:
            f_test.write(s)
            cou += 1
        else:
            f_train.write(s)
            cou += 1

cou = 0
for dirPath, dirNames, fileNames in os.walk('crop\\2'):
    #print(dirPath, dirNames, fileNames)
    s = 'D:\\Dataset\\Action\\my_dataset\\' + dirPath + ' ' + str(len(fileNames)) + ' 2\n'
    if len(fileNames) != 0:
        if cou % 3 == 0:
            f_test.write(s)
            cou += 1
        else:
            f_train.write(s)
            cou += 1

f_train.close()
f_test.close()