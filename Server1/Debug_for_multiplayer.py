import tkinter
import time
import threading
import tkinter.font as font

action = 1



class GUI_Test(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global action

        mainWin = tkinter.Tk()
        mainWin.title("Debug for multiplayer")
        mainWin.geometry("600x600")

        myFont = font.Font(family='Helvetica', size=16, weight='bold')

        self.my_str = tkinter.StringVar()

        status_label = tkinter.Label(mainWin, text='Status :', width=10, height=1).place(x=0, y=0)

        self.my_str.set(action)
        status_label_ = tkinter.Label(mainWin, textvariable=self.my_str, font=myFont, width=10, height=1).place(x=100, y=0)

        action_1 = tkinter.Button(mainWin, text="1", font=myFont, command=self.action_1_CallBack)
        action_1.pack()

        action_2 = tkinter.Button(mainWin, text="2", font=myFont, command=self.action_2_CallBack)
        action_2.pack()

        action_3 = tkinter.Button(mainWin, text="3", font=myFont, command=self.action_3_CallBack)
        action_3.pack()

        action_4 = tkinter.Button(mainWin, text="4", font=myFont, command=self.action_4_CallBack)
        action_4.pack()

        action_5 = tkinter.Button(mainWin, text="5", font=myFont, command=self.action_5_CallBack)
        action_5.pack()

        action_6 = tkinter.Button(mainWin, text="6", font=myFont, command=self.action_6_CallBack)
        action_6.pack()

        action_7 = tkinter.Button(mainWin, text="7", font=myFont, command=self.action_7_CallBack)
        action_7.pack()

        action_8 = tkinter.Button(mainWin, text="8", font=myFont, command=self.action_8_CallBack)
        action_8.pack()

        action_9 = tkinter.Button(mainWin, text="9", font=myFont, command=self.action_9_CallBack)
        action_9.pack()

        action_10 = tkinter.Button(mainWin, text="10", font=myFont, command=self.action_10_CallBack)
        action_10.pack()

        action_11 = tkinter.Button(mainWin, text="11", font=myFont, command=self.action_11_CallBack)
        action_11.pack()

        mainWin.mainloop()
    
    def action_1_CallBack(self):
        global action
        action = 1
        self.my_str.set(action)
        with open('Debug_for_multiplayer.txt', 'w') as f:
            f.write(str(action))

    def action_2_CallBack(self):
        global action
        action = 2
        self.my_str.set(action)
        with open('Debug_for_multiplayer.txt', 'w') as f:
            f.write(str(action))

    def action_3_CallBack(self):
        global action
        action = 3
        self.my_str.set(action)
        with open('Debug_for_multiplayer.txt', 'w') as f:
            f.write(str(action))

    def action_4_CallBack(self):
        global action
        action = 4
        self.my_str.set(action)
        with open('Debug_for_multiplayer.txt', 'w') as f:
            f.write(str(action))

    def action_5_CallBack(self):
        global action
        action = 5
        self.my_str.set(action)
        with open('Debug_for_multiplayer.txt', 'w') as f:
            f.write(str(action))
    
    def action_6_CallBack(self):
        global action
        action = 6
        self.my_str.set(action)
        with open('Debug_for_multiplayer.txt', 'w') as f:
            f.write(str(action))

    def action_7_CallBack(self):
        global action
        action = 7
        self.my_str.set(action)
        with open('Debug_for_multiplayer.txt', 'w') as f:
            f.write(str(action))
    
    def action_8_CallBack(self):
        global action
        action = 8
        self.my_str.set(action)
        with open('Debug_for_multiplayer.txt', 'w') as f:
            f.write(str(action))
    
    def action_9_CallBack(self):
        global action
        action = 9
        self.my_str.set(action)
        with open('Debug_for_multiplayer.txt', 'w') as f:
            f.write(str(action))
    
    def action_10_CallBack(self):
        global action
        action = 10
        self.my_str.set(action)
        with open('Debug_for_multiplayer.txt', 'w') as f:
            f.write(str(action))

    def action_11_CallBack(self):
        global action
        action = 11
        self.my_str.set(action)
        with open('Debug_for_multiplayer.txt', 'w') as f:
            f.write(str(action))
        

if __name__ == "__main__":
    
    with open('Debug_for_multiplayer.txt', 'w') as f:
        f.write(str(action))
    
    GUI_Test().start()

    # while(True):
    #     with open('Debug_for_multiplayer.txt', 'r') as f:
    #         test = f.readline()
    #     print(test.strip())
    #     time.sleep(1)