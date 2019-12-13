## HoloLens for Unity

要從Unity編譯到HoloLens裡面，需進行兩段編譯。Unity編譯時的設定如下：

![image](../etcs/Unity_Setting.JPG)

務必確認：
1. Scenes In Bulid
2. Build Settings
3. Player Settings 裡面的 Capabilities

此時在 "Scenes In Build" 裡面把這 5 個Scenes按照順序新增(要按照順序，右邊有index)，新增完之後右邊的Player Settings要需要調整一下，都設定好了之後按下 Build 開始建置，這個步驟可能需要花2分鐘左右，這部分就是第一段編譯。

建置完畢後後打開資料夾，選擇 ".sln" 執行Visual Studio，到裡面進行第二段編譯。

![image](../etcs/Second_Compile.JPG)

到Visual Studio裡面，上面哪一排設定要設定好，選擇Release x86，Device選擇HoloLens，我自己是透過USB讓HoloLens跟PC連接。

![image](../etcs/Second_Compile_VS.JPG)

編譯完之後，HoloLens就會自動執行程式了，這就是第二段編譯，整個過程大概需要2分鐘，然後在這2分鐘的過程，不能讓HoloLens睡眠，螢幕須一直亮著，否則會error。

到這邊HoloLens端的程式已經建立完成。若[Server1](../Server1/README.md)的環境也都建立好，就可以開始遊戲。

Note : Server1.py 需要先在PC上運行，也只需要運行一次(一直開著)，HoloLens可隨時開啟遊戲與PC串接。