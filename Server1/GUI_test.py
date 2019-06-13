import tkinter
import time
import threading

global_action = 0



class GUI_Test(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global global_action

        mainWin = tkinter.Tk()
        mainWin.title("Server 1")
        mainWin.geometry("600x600")

        self.my_str = tkinter.StringVar()

        status_label = tkinter.Label(mainWin, text='Status :', width=10, height=1).place(x=0, y=0)
        status_label_ = tkinter.Label(mainWin, textvariable=self.my_str, bg='gray', width=10, height=1).place(x=100, y=0)
        buttonVariable = tkinter.Button(mainWin, text="這是按鈕", command=self.btnCallBack)
        buttonVariable.pack()

        mainWin.mainloop()
    
    def btnCallBack(self):
        global global_action
        global_action += 1
        self.my_str.set(global_action)
        

if __name__ == "__main__":
    
    GUI_Test().start()

    
    while(True):
        print(global_action)
        time.sleep(1)
        