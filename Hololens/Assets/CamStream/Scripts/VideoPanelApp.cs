//  
// Copyright (c) 2017 Vulcan, Inc. All rights reserved.  
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.
//

using UnityEngine;
using UnityEngine.UI;
using System.Text;
using System.IO;
using System;

using HoloLensCameraStream;

using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgcodecsModule;
using OpenCVForUnity.VideoModule;

#if !UNITY_EDITOR
    using Windows.Networking;
    using Windows.Networking.Sockets;
    using Windows.Storage.Streams;
#endif

/// <summary>
/// This example gets the video frames at 30 fps and displays them on a Unity texture,
/// which is locked the User's gaze.
/// </summary>
public class VideoPanelApp : MonoBehaviour
{
    byte[] _latestImageBytes;
    byte[] final;


    HoloLensCameraStream.Resolution _resolution;

    //"Injected" objects.
    VideoPanel _videoPanelUI;
    VideoCapture _videoCapture;

    IntPtr _spatialCoordinateSystemPtr;

    private int flag = 1;
    private int frame_ready = 0;
    private string str_text;

    public Text debug_1;
    public Text debug_2;
    public Text debug_3;
    
    public Image joint1;
    public Image joint2;
    public Image joint3;
    public Image joint4;
    public Image joint5;
    public Image joint6;
    public Image joint7;
    public Image Skill_1;
    public Image Skill_2;
    public Image Skill_3;
    public Image Skill_4;
    public Image Skill_5;
    public Image Skill_6;
    public GameObject panel;
    public GameObject can;
    public Material line_mat;

    private float LWrist_x;
    private float LWrist_y;
    private float LElbow_x;
    private float LElbow_y;
    private float LShoulder_x;
    private float LShoulder_y;
    private float RWrist_x;
    private float RWrist_y;
    private float RElbow_x;
    private float RElbow_y;
    private float RShoulder_x;
    private float RShoulder_y;
    private float Neck_x;
    private float Neck_y;

    private int action;
    private long local_fps;
    private int recv_frame;
    private int recv_frame_old;
    private int frame_cou;
    private double fps;
    private int img_len;
    private float z;

#if !UNITY_EDITOR
    //StreamSocket socket;
    StreamSocketListener listener;
    //public DataWriter writer;
    //public DataReader reader;
    String port;
    String message;
    String IP;

    //StreamSocket streamSocket;
#endif

    void Start()
    {
        LWrist_x = 0f;
        LWrist_y = 0f;
        LElbow_x = 0f;
        LElbow_y = 0f;
        LShoulder_x = 0f;
        LShoulder_y = 0f;
        RWrist_x = 0f;
        RWrist_y = 0f;
        RElbow_x = 0f;
        RElbow_y = 0f;
        RShoulder_x = 0f;
        RShoulder_y = 0f;
        Neck_x = 0f;
        Neck_y = 0f;

        action = 0;
        local_fps = 0;
        recv_frame = 1;
        recv_frame_old = 0;
        frame_cou = 0;

        z = 90f;

        panel.GetComponent<Transform>().localPosition = new Vector3(0f, 0f, z);

        //Fetch a pointer to Unity's spatial coordinate system if you need pixel mapping
        _spatialCoordinateSystemPtr = UnityEngine.XR.WSA.WorldManager.GetNativeISpatialCoordinateSystemPtr();

        //Call this in Start() to ensure that the CameraStreamHelper is already "Awake".
        CameraStreamHelper.Instance.GetVideoCaptureAsync(OnVideoCaptureCreated);
        //You could also do this "shortcut":
        //CameraStreamManager.Instance.GetVideoCaptureAsync(v => videoCapture = v);

        _videoPanelUI = GameObject.FindObjectOfType<VideoPanel>();

        KalmanFilter KF = new KalmanFilter(4,2,0, CvType.CV_32F);
        MatOfFloat trans = new MatOfFloat(1f, 0f, 1f, 0f, 0f, 1f, 0f, 1f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f);
        KF.set_transitionMatrix(trans);
        Mat measurement = Mat.zeros(1, 1, CvType.CV_32F);
        Mat statePre = new Mat(2, 1, CvType.CV_32F, new Scalar(0));
        statePre.put(0, 0, 0f);
        statePre.put(1, 0, 0f);
        KF.set_statePre(statePre);


#if !UNITY_EDITOR
        port = "9000";
        IP = "192.168.1.106";
        
        listener = new StreamSocketListener();
        
        listener.ConnectionReceived += Listener_send;
        listener.Control.KeepAlive = false;

        Listener_Start();
#endif
    }

    private void OnDestroy()
    {
        if (_videoCapture != null)
        {
            _videoCapture.FrameSampleAcquired -= OnFrameSampleAcquired;
            _videoCapture.Dispose();
        }
#if !UNITY_EDITOR
        //streamSocket.Dispose();
        listener.Dispose();
#endif
    }

    void OnVideoCaptureCreated(VideoCapture videoCapture)
    {
        if (videoCapture == null)
        {
            Debug.LogError("Did not find a video capture object. You may not be using the HoloLens.");
            return;
        }
        
        this._videoCapture = videoCapture;

        //Request the spatial coordinate ptr if you want fetch the camera and set it if you need to 
        CameraStreamHelper.Instance.SetNativeISpatialCoordinateSystemPtr(_spatialCoordinateSystemPtr);

        _resolution = CameraStreamHelper.Instance.GetLowestResolution();
        float frameRate = CameraStreamHelper.Instance.GetHighestFrameRate(_resolution);
        videoCapture.FrameSampleAcquired += OnFrameSampleAcquired;

        //You don't need to set all of these params.
        //I'm just adding them to show you that they exist.
        CameraParameters cameraParams = new CameraParameters();
        cameraParams.cameraResolutionHeight = _resolution.height;
        cameraParams.cameraResolutionWidth = _resolution.width;
        cameraParams.frameRate = Mathf.RoundToInt(frameRate);
        cameraParams.pixelFormat = CapturePixelFormat.BGRA32;
        //cameraParams.rotateImage180Degrees = true; //If your image is upside down, remove this line.
        cameraParams.enableHolograms = false;

        UnityEngine.WSA.Application.InvokeOnAppThread(() => { _videoPanelUI.SetResolution(_resolution.width, _resolution.height); }, false);

        videoCapture.StartVideoModeAsync(cameraParams, OnVideoModeStarted);
    }

    void OnVideoModeStarted(VideoCaptureResult result)
    {
        if (result.success == false)
        {
            Debug.LogWarning("Could not start video mode.");
            return;
        }

        Debug.Log("Video capture started.");
    }

    void OnFrameSampleAcquired(VideoCaptureSample sample)
    {
        //When copying the bytes out of the buffer, you must supply a byte[] that is appropriately sized.
        //You can reuse this byte[] until you need to resize it (for whatever reason).
        if (_latestImageBytes == null || _latestImageBytes.Length < sample.dataLength)
        {
            _latestImageBytes = new byte[sample.dataLength];
        }
        sample.CopyRawImageDataIntoBuffer(_latestImageBytes);

        sample.Dispose();

        //This is where we actually use the image data
        UnityEngine.WSA.Application.InvokeOnAppThread(() =>
        {
            _videoPanelUI.SetBytes(_latestImageBytes);
        }, false);

        int cou = 0;
        byte[] no_alpha = new byte[448 * 252 * 3];
        for (int x = 0; x < 504; x += 2)
        {
            for (int y = 0; y < 896; y += 2)
            {
                no_alpha[cou] = _latestImageBytes[(x * 896 + y) * 4];
                cou += 1;
                no_alpha[cou] = _latestImageBytes[(x * 896 + y) * 4 + 1];
                cou += 1;
                no_alpha[cou] = _latestImageBytes[(x * 896 + y) * 4 + 2];
                cou += 1;
            }
        }

        Mat mat_img = new Mat(252, 448, CvType.CV_8UC3);
        mat_img.put(0, 0, no_alpha);
        MatOfByte matofbyte = new MatOfByte();
        MatOfInt compressParm;
        compressParm = new MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 80);
        Imgcodecs.imencode(".jpg", mat_img, matofbyte, compressParm);
        byte[] img_encode = matofbyte.toArray();

        img_len = img_encode.Length;
        byte[] len_byte = BitConverter.GetBytes(img_len);
        if (BitConverter.IsLittleEndian)
            Array.Reverse(len_byte);
        final = new byte[img_len + 4];
        Array.Copy(len_byte, 0, final, 0, 4);
        Array.Copy(img_encode, 0, final, 4, img_len);

    }

    private void Update()
    {
        debug_1.text = action.ToString();
        debug_2.text = ((int)fps).ToString();
        if ((int)fps < 30)
        {
            debug_2.color = Color.red;
        }
        else
        {
            debug_2.color = Color.green;
        }

        debug_3.text = recv_frame.ToString();
        if(recv_frame == recv_frame_old)
        {
            if(frame_cou > 6)
            {
                debug_3.color = Color.red;
            }
            frame_cou += 1;
        }
        else
        {
            debug_3.color = Color.green;
            recv_frame_old = recv_frame;
            frame_cou = 0;
        }


        joint4.GetComponent<RectTransform>().localPosition = new Vector3(LWrist_x / 10.0f - 22.4f, -(LWrist_y / 10.0f - 12.6f), 0f);
        joint2.GetComponent<RectTransform>().localPosition = new Vector3(LElbow_x / 10.0f - 22.4f, -(LElbow_y / 10.0f - 12.6f), 0f);
        joint3.GetComponent<RectTransform>().localPosition = new Vector3(LShoulder_x / 10.0f - 22.4f, -(LShoulder_y / 10.0f - 12.6f), 0f);

        joint1.GetComponent<RectTransform>().localPosition = new Vector3(RWrist_x / 10.0f - 22.4f, -(RWrist_y / 10.0f - 12.6f), 0f);
        joint5.GetComponent<RectTransform>().localPosition = new Vector3(RElbow_x / 10.0f - 22.4f, -(RElbow_y / 10.0f - 12.6f), 0f);
        joint6.GetComponent<RectTransform>().localPosition = new Vector3(RShoulder_x / 10.0f - 22.4f, -(RShoulder_y / 10.0f - 12.6f), 0f);

        joint7.GetComponent<RectTransform>().localPosition = new Vector3(Neck_x / 10.0f - 22.4f, -(Neck_y / 10.0f - 12.6f), 0f);

        if (action == 1)
        {
            Skill_1.GetComponent<RectTransform>().localPosition = new Vector3(RWrist_x / 10.0f - 22.5f, -(RWrist_y / 10.0f - 12.6f) + 0.5f, 0f);
            joint1.GetComponent<RectTransform>().localPosition = new Vector3(-22.4f, 12.6f, 0f);
        }
        else
        {
            Skill_1.GetComponent<RectTransform>().localPosition = new Vector3(-30.4f, 20.6f, 0f);
        }
    }


#if !UNITY_EDITOR

    private async void Listener_Start()
    {
        Debug.Log("Listener started");
        try
        {
            var hostName = new Windows.Networking.HostName(IP);
            await listener.BindEndpointAsync(hostName, port);
        }
        catch (Exception e)
        {
            Debug.Log("Error: " + e.Message);
        }

        Debug.Log("Listening");
        local_fps = System.Diagnostics.Stopwatch.GetTimestamp();
    }

    private async void Listener_send(StreamSocketListener sender, StreamSocketListenerConnectionReceivedEventArgs args)
    {
        Debug.Log("Connection send");

        try
        {
            while (true) {
    
                using (var dw = new DataWriter(args.Socket.OutputStream))
                {
                    dw.WriteBytes(final);
                    await dw.StoreAsync();
                    dw.DetachStream();
                }


                using (var dr = new DataReader(args.Socket.InputStream))
                {
                    /*
                    LWrist_x_old = LWrist_x;
                    LWrist_y_old = LWrist_y;
                    LElbow_x_old = LElbow_x;
                    LElbow_y_old = LElbow_y;
                    LShoulder_x_old = LShoulder_x;
                    LShoulder_y_old = LShoulder_y;
                    RWrist_x_old = RWrist_x;
                    RWrist_y_old = RWrist_y;
                    RElbow_x_old = RElbow_x;
                    RElbow_y_old = RElbow_y;
                    RShoulder_x_old = RShoulder_x;
                    RShoulder_y_old = RShoulder_y;
                    Neck_x_old = Neck_x;
                    Neck_y_old = Neck_y;
                    */
                    
                    dr.InputStreamOptions = InputStreamOptions.Partial;
                    uint uintBytes = await dr.LoadAsync(1024);
                    string input = dr.ReadString(uintBytes);
                    dr.DetachStream();
                    //Debug.Log("received :" + input);

                    str_text = input;
                    string[] arr = input.Split(',');

                    LWrist_x = float.Parse(arr[0]);
                    LWrist_y = float.Parse(arr[1]);
                    LElbow_x = float.Parse(arr[2]);
                    LElbow_y = float.Parse(arr[3]);
                    LShoulder_x = float.Parse(arr[4]);
                    LShoulder_y = float.Parse(arr[5]);
                    RWrist_x = float.Parse(arr[6]);
                    RWrist_y = float.Parse(arr[7]);
                    RElbow_x = float.Parse(arr[8]);
                    RElbow_y = float.Parse(arr[9]);
                    RShoulder_x = float.Parse(arr[10]);
                    RShoulder_y = float.Parse(arr[11]);
                    Neck_x = float.Parse(arr[12]);
                    Neck_y = float.Parse(arr[13]);
                    recv_frame = int.Parse(arr[14]);
                    action = int.Parse(arr[15]);
    
                    frame_ready = 1;
                    
                    fps = 10000000.0 / (System.Diagnostics.Stopwatch.GetTimestamp() - local_fps);
                    local_fps = System.Diagnostics.Stopwatch.GetTimestamp();
                }
            }
        }
        catch (Exception e)
        {
            Debug.Log("connect error!!!!!!!! " + e);
        }

    }

#endif

}
