using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgcodecsModule;
using OpenCVForUnity.ImgprocModule;

using System.Net.Sockets;

public class Demo : MonoBehaviour
{
    private ClientThread ct;
    private bool isSend;
    private bool isReceive;
    public RawImage rawImage;
    public GameObject connect;

    private void Start()
    {
        ct = new ClientThread(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp, "192.168.208.120", 9000);
        ct.StartConnect();
        isSend = true;
        var texture = new Texture2D(448, 252, TextureFormat.RGB24, false);
        rawImage.texture = texture;
    }

    private void Update()
    {
        if (!ct.begin_flag)
        {
            if (!ct.rec)
            {
                ct.Receive();
            }
        }
        if (ct.receiveMessage != null)
        {
            Debug.Log("Server:" + ct.receiveMessage);
            ct.receiveMessage = null;
        }
        /*if (ct.receive_joint != null)
        {
            Debug.Log("Server:" + ct.receive_joint);
            ct.receive_joint = null;
        }*/
        if (ct.recieve_flag != 0)
        {
            //Debug.Log("Server:" + ct.frame_size.ToString());
            if (ct.frame_img.Length != ct.dataLength_frame)
            {
                Debug.Log("Receive : " + ct.frame_img.Length.ToString() + ", len : " + ct.dataLength.ToString());
            }

            Mat mat_img = new Mat(1, ct.frame_img.Length, CvType.CV_8U);
            mat_img.put(0, 0, ct.frame_img);
            Mat frame_img_mat = Imgcodecs.imdecode(mat_img, 1);
            Imgproc.cvtColor(frame_img_mat, frame_img_mat, Imgproc.COLOR_BGR2RGB);
            //Debug.Log(frame_img_mat.size());
            byte[] image = new byte[252 * 448 * 3];
            frame_img_mat.get(0, 0, image);

            var texture = rawImage.texture as Texture2D;
            texture.LoadRawTextureData(image); //TODO: Should be able to do this: texture.LoadRawTextureData(pointerToImage, 1280 * 720 * 4);
            texture.Apply();
            ct.recieve_flag = 0;
            ct.rec = false;
        }
    }
    

    private void OnApplicationQuit()
    {
        ct.Send("bye");
        ct.StopConnect();
    }


    public void on_Click()
    {
        //string my_s = "Unity_demo";
        //ct.Send(my_s);

        ct.Receive();
        //connect.enabled = false;
        connect.SetActive(false);
        //Demo_Effect.connect = true;
    }
}
