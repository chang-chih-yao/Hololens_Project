with open('my_test.txt', 'r') as f:
    lines = f.readlines()

print(len(lines))

f_0 = open('my_test_0.txt', 'w')
f_1 = open('my_test_1.txt', 'w')
f_2 = open('my_test_2.txt', 'w')
f_3 = open('my_test_3.txt', 'w')

for i in range(len(lines)):
    if lines[i].split(' ')[0].split('\\')[-2] == '0':
        f_0.write(lines[i])
    elif lines[i].split(' ')[0].split('\\')[-2] == '1':
        f_1.write(lines[i])
    elif lines[i].split(' ')[0].split('\\')[-2] == '2':
        f_2.write(lines[i])
    elif lines[i].split(' ')[0].split('\\')[-2] == '3':
        f_3.write(lines[i])

f_0.close()
f_1.close()
f_2.close()
f_3.close()