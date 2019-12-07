import tkinter
import time
import threading
import tkinter.font as font

pre_z = -20



class GUI_Test(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global pre_z

        mainWin = tkinter.Tk()
        mainWin.title("Server 1")
        mainWin.geometry("600x600")

        myFont = font.Font(family='Helvetica', size=16, weight='bold')

        self.my_str = tkinter.StringVar()

        status_label = tkinter.Label(mainWin, text='Status :', width=10, height=1).place(x=0, y=0)

        self.my_str.set(pre_z)
        status_label_ = tkinter.Label(mainWin, textvariable=self.my_str, font=myFont, width=10, height=1).place(x=100, y=0)

        button_plus_5 = tkinter.Button(mainWin, text="+5 (變遠)", font=myFont, command=self.button_plus_5_CallBack)
        button_plus_5.pack()

        button_plus_1 = tkinter.Button(mainWin, text="+1 (變遠)", font=myFont, command=self.button_plus_1_CallBack)
        button_plus_1.pack()

        button_minus_1 = tkinter.Button(mainWin, text="-1 (變近)", font=myFont, command=self.button_minus_1_CallBack)
        button_minus_1.pack()

        button_minus_5 = tkinter.Button(mainWin, text="-5 (變近)", font=myFont, command=self.button_minus_5_CallBack)
        button_minus_5.pack()

        mainWin.mainloop()
    
    def button_plus_5_CallBack(self):
        global pre_z
        pre_z += 5
        self.my_str.set(pre_z)
        with open('Debug_for_3D_position_estimation.txt', 'w') as f:
            f.write(str(pre_z))

    def button_plus_1_CallBack(self):
        global pre_z
        pre_z += 1
        self.my_str.set(pre_z)
        with open('Debug_for_3D_position_estimation.txt', 'w') as f:
            f.write(str(pre_z))

    def button_minus_1_CallBack(self):
        global pre_z
        pre_z -= 1
        self.my_str.set(pre_z)
        with open('Debug_for_3D_position_estimation.txt', 'w') as f:
            f.write(str(pre_z))

    def button_minus_5_CallBack(self):
        global pre_z
        pre_z -= 5
        self.my_str.set(pre_z)
        with open('Debug_for_3D_position_estimation.txt', 'w') as f:
            f.write(str(pre_z))
        

if __name__ == "__main__":
    
    with open('Debug_for_3D_position_estimation.txt', 'w') as f:
        f.write(str(pre_z))
    
    GUI_Test().start()

    
    # while(True):
    #     print(pre_z)
    #     with open('Debug_for_3D_position_estimation.txt', 'r') as f:
    #         test = f.readline()
    #     print(test.strip())
    #     time.sleep(1)
        