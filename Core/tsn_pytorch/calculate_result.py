import numpy as np


def calculate_no_dir():
    with open('result.txt', 'r') as f:
        Lines = f.readlines()
    print(int(len(Lines)/4))

    correct = 0
    for i in range(len(Lines)):
        if int(Lines[i].split('\\')[7]) == 0:
            arr = np.array([0,0,0,0,0,0,0])
            for j in range(len(Lines)):
                if Lines[j].split('\\')[6] == Lines[i].split('\\')[6] and Lines[j].split('\\')[8].split(' ')[0] == Lines[i].split('\\')[8].split(' ')[0]:
                    arr[int(Lines[j].split(' ')[-1].replace('\n', ''))] += 1
                    #print(Lines[j], end='')
            #print(np.argmax(arr))
            if int(Lines[i].split('\\')[6]) == np.argmax(arr):
                correct += 1

    print(correct)
    print(correct/(len(Lines)/4)*100)

def calculate():
    with open('result.txt', 'r') as f:
        Lines = f.readlines()
    print(int(len(Lines)/4))

    correct = 0
    for i in range(len(Lines)):
        if int(Lines[i].split('\\')[7]) == 0:
            arr = np.array([0,0,0,0,0,0,0])
            for j in range(len(Lines)):
                if Lines[j].split('\\')[6] == Lines[i].split('\\')[6] and Lines[j].split('\\')[8].split(' ')[0] == Lines[i].split('\\')[8].split(' ')[0]:
                    action_label = int((int(Lines[j].split(' ')[-1].replace('\n', '')) + 2)/4) + 1
                    arr[action_label] += 1
                    #print(Lines[j], end='')
            #print(np.argmax(arr))
            if int(Lines[i].split('\\')[6]) == np.argmax(arr):
                correct += 1

    print(correct)
    print(correct/(len(Lines)/4)*100)

if __name__ == "__main__":
    # choose one function
    calculate()
    #calculate_no_dir()