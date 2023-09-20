import os
import random


def text():
    list_train=[]
    list_test=[]
    for i in os.listdir('new_recordings/'):
        list=os.listdir('new_recordings/'+i)
        random.shuffle(list)
        for j in range(len(list)):
            filepath = 'new_recordings/' + i + '/' + list[j]
            leibie = i
            if j < len(os.listdir('new_recordings/'+i))*0.9:
                list_train.append(filepath+'\t'+leibie)
            else:
                list_test.append(filepath + '\t' + leibie)
    random.shuffle(list_train)
    random.shuffle(list_test)
    train_file=open('new_recordings_train.txt',encoding='utf-8',mode='w')
    test_file = open('new_recordings_test.txt', encoding='utf-8', mode='w')
    for i in list_train:
        train_file.write(i+'\n')
    train_file.close()

    for i in list_test:
        test_file.write(i+'\n')
    test_file.close()

if __name__=='__main__':
    text()