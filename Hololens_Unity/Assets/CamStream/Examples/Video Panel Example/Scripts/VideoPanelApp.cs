//  
// Copyright (c) 2017 Vulcan, Inc. All rights reserved.  
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.
//

using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using System.Text;
using System.IO;
using System;
using System.Collections;

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
    byte[] game_status;
    byte[] final;
    private int lock_data = 0;

    // status = "被對方的技能(SK2)打到, 被對方的技能(SK3)打到, 被對方的技能(SK4)打到, 被對方的技能(SK5)打到, end game, start new game"
    private string status="0,0,0,0,0,0";


    HoloLensCameraStream.Resolution _resolution;

    //"Injected" objects.
    VideoPanel _videoPanelUI;
    VideoCapture _videoCapture;

    IntPtr _spatialCoordinateSystemPtr;
    
    private int frame_ready = 0;
    private string str_text;
    private Boolean connect = false;
    private Boolean animation_flag = false;
    private Boolean SK2_animation = false;
    //private Boolean SK2_fail_flag = false;
    private Boolean my_SK2_animation = false;
    //private Boolean my_SK2_fail_flag = false;
    private Boolean blood_animation = false;
    private Boolean defense_animation = false;
    private Boolean failed_attack_animation = false;
    private Boolean end_game_flag = false;


    public Text debug_1;
    public Text debug_2;
    public Text debug_3;
    public Text debug_4;

    public Image[] Joints;
    public Image P0_HP;
    public Image P1_HP;

    public Text Skill_1;
    //public Image Skill_2;
    public Text Skill_3;
    public Text Skill_4;
    public Text Skill_5;
    public Text Skill_6;
    public GameObject Skill_2;
    public GameObject my_Skill_2;
    public GameObject panel;
    public GameObject can;
    public Material line_mat;
    public GameObject win_img;
    public GameObject lose_img;
    public GameObject start_new_game;
    public GameObject successful_defense;
    public GameObject failed_attack;

    public SpriteRenderer blood;
    
    private float[] Joints_co;

    private string player;
    private int action = 0;
    private int my_action = 0;
    private int holo_action_p0 = -1;
    private int holo_action_p1 = -1;
    private float gamepoint_p0 = 10f;
    private float gamepoint_p1 = 10f;
    private float cur_gamepoint_p0 = 10f;
    private float cur_gamepoint_p1 = 10f;
    private int p0_win_lose = 0;
    private int p1_win_lose = 0;
    private int defense_skill_2_p0 = 0;
    private int defense_skill_2_p1 = 0;
    private int blood_effect_p0 = 0;
    private int blood_effect_p1 = 0;

    private long local_fps = 0;
    private int recv_frame;
    private int recv_frame_old;
    private int frame_cou = 0;
    private double fps;
    private int img_len;
    private float z = 60;

    private float status_s2_hit_time;
    private float defense_time;
    private float failed_attack_time;
    //private float status_start_new_game_time;
    //private float status_end_game_time;

    private float P0_HP_X;
    private float P1_HP_X;
    private float SK2_reduce_x;
    private float SK2_reduce_y;
    private float SK2_reduce_z;
    private float SK2_limit;

#if !UNITY_EDITOR
    String port;
    String IP;
    StreamSocket streamSocket;
#endif

    void Start()
    {
        Joints_co = new float[Joints.Length * 2];
        for (int i = 0; i < Joints.Length * 2; i++)
        {
            Joints_co[i] = 0f;
        }
        
        recv_frame = 1;
        recv_frame_old = 0;
        P0_HP_X = P0_HP.GetComponent<RectTransform>().localPosition.x;
        P1_HP_X = P1_HP.GetComponent<RectTransform>().localPosition.x;

        player = Game_Stats.PlayerID;
        game_status = new byte[256];

        panel.GetComponent<Transform>().localPosition = new Vector3(0f, 0f, z);

        //Fetch a pointer to Unity's spatial coordinate system if you need pixel mapping
        _spatialCoordinateSystemPtr = UnityEngine.XR.WSA.WorldManager.GetNativeISpatialCoordinateSystemPtr();

        //Call this in Start() to ensure that the CameraStreamHelper is already "Awake".
        CameraStreamHelper.Instance.GetVideoCaptureAsync(OnVideoCaptureCreated);
        //You could also do this "shortcut":
        //CameraStreamManager.Instance.GetVideoCaptureAsync(v => videoCapture = v);

        _videoPanelUI = GameObject.FindObjectOfType<VideoPanel>();
        
#if !UNITY_EDITOR
        port = "9000";
        IP = Game_Stats.IP;
        StartClient(streamSocket);
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
        streamSocket.Dispose();
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
        final = new byte[img_len + 4 + game_status.Length];
        lock_data = 1;
        Array.Copy(len_byte, 0, final, 0, 4);
        Array.Copy(img_encode, 0, final, 4, img_len);
        byte[] status_temp = Encoding.ASCII.GetBytes(status);
        Array.Copy(status_temp, 0, game_status, 0, status_temp.Length);
        Array.Copy(game_status, 0, final, 4 + img_len, game_status.Length);
        lock_data = 0;
    }

    private Vector3 joint_coordinate_trans(float x, float y)
    {
        return new Vector3(x / 10.0f - 22.4f + 2f, -(y / 10.0f - 12.6f + 0.7f), 0f);
    }

    private Vector3 Skill_2_coordinate_trans(float RWrist_x, float RWrist_y, float RElbow_x, float RElbow_y)
    {
        Vector3 RWrist = new Vector3(RWrist_x / 10.0f - 22.4f + 2f, -(RWrist_y / 10.0f - 12.6f + 0.7f), 0f);
        Vector3 RElbow = new Vector3(RElbow_x / 10.0f - 22.4f + 2f, -(RElbow_y / 10.0f - 12.6f + 0.7f), 0f);
        Vector3 my_vec = RWrist - RElbow;
        return RElbow + 1.2f * my_vec;
    }

    private IEnumerator Lose_Game()
    {
        status = replace_char(status, 10, "1");

        My_Line.Draw_skeleton = false;
        lose_img.SetActive(true);
        yield return new WaitForSecondsRealtime(2.5f);
        lose_img.SetActive(false);
        start_new_game.SetActive(true);
        yield return new WaitForSecondsRealtime(2.5f);
        start_new_game.SetActive(false);

        // 把loadscene拿掉，改成 reset 雙方血量變數
        P0_HP.GetComponent<RectTransform>().sizeDelta = new Vector2(10, P0_HP.GetComponent<RectTransform>().sizeDelta.y);
        P1_HP.GetComponent<RectTransform>().sizeDelta = new Vector2(10, P1_HP.GetComponent<RectTransform>().sizeDelta.y);
        P0_HP.GetComponent<RectTransform>().localPosition = new Vector3(P0_HP_X, P0_HP.GetComponent<RectTransform>().localPosition.y,
                    P0_HP.GetComponent<RectTransform>().localPosition.z);
        P1_HP.GetComponent<RectTransform>().localPosition = new Vector3(P1_HP_X, P1_HP.GetComponent<RectTransform>().localPosition.y,
                    P1_HP.GetComponent<RectTransform>().localPosition.z);
        
        cur_gamepoint_p0 = 10f;
        cur_gamepoint_p1 = 10f;
        My_Line.Draw_skeleton = true;

        status = replace_char(status, 10, "0");
    }

    private IEnumerator Win_Game()
    {
        status = replace_char(status, 10, "1");

        My_Line.Draw_skeleton = false;
        win_img.SetActive(true);
        yield return new WaitForSecondsRealtime(2.5f);
        win_img.SetActive(false);
        start_new_game.SetActive(true);
        yield return new WaitForSecondsRealtime(2.5f);
        start_new_game.SetActive(false);

        // 把loadscene拿掉，改成 reset 雙方血量變數
        P0_HP.GetComponent<RectTransform>().sizeDelta = new Vector2(10, P0_HP.GetComponent<RectTransform>().sizeDelta.y);
        P1_HP.GetComponent<RectTransform>().sizeDelta = new Vector2(10, P1_HP.GetComponent<RectTransform>().sizeDelta.y);
        P0_HP.GetComponent<RectTransform>().localPosition = new Vector3(P0_HP_X, P0_HP.GetComponent<RectTransform>().localPosition.y,
                    P0_HP.GetComponent<RectTransform>().localPosition.z);
        P1_HP.GetComponent<RectTransform>().localPosition = new Vector3(P1_HP_X, P1_HP.GetComponent<RectTransform>().localPosition.y,
                    P1_HP.GetComponent<RectTransform>().localPosition.z);
        
        cur_gamepoint_p0 = 10f;
        cur_gamepoint_p1 = 10f;
        My_Line.Draw_skeleton = true;

        status = replace_char(status, 10, "0");
    }

    private string replace_char(string s, int index, string p)
    {
        s = s.Remove(index, 1).Insert(index, p);
        return s;
    }
    

    private void Update()
    {
        if (connect == true) {
            //if (status[0] == '1' && ((Time.time - status_s2_hit_time) >= 0.4f))
                //status = replace_char(status, 0, "0");
            
            //z = z - 0.01f;
            //panel.GetComponent<Transform>().localPosition = new Vector3(0f, 0f, z);
            //debug_4.text = "blood_effect\n" + blood_effect_p0.ToString() + ", " + blood_effect_p1.ToString();
            debug_1.text = action.ToString() + ", " + holo_action_p0.ToString() + "|" + holo_action_p1.ToString() + ", " + gamepoint_p0.ToString() + "|" + gamepoint_p1.ToString() + ", " + player;
            debug_2.text = ((int)fps).ToString();
            if ((int)fps < 10)
                debug_2.color = Color.red;
            else if ((int)fps < 30 && (int)fps >= 10)
                debug_2.color = Color.yellow;
            else
                debug_2.color = Color.green;

            debug_3.text = recv_frame.ToString();
            if (recv_frame == recv_frame_old)
            {
                if (frame_cou > 6)
                    debug_3.color = Color.red;
                frame_cou += 1;
            }
            else
            {
                debug_3.color = Color.green;
                recv_frame_old = recv_frame;
                frame_cou = 0;
            }

            //  calculate HP
            if (cur_gamepoint_p0 - gamepoint_p0 > 0.9f)     // if gamepoint_p0 != cur_gamepoint_p0 (cannot use == or != , sometimes may cause error)
            {
                float p0_x = P0_HP.GetComponent<RectTransform>().localPosition.x - ((cur_gamepoint_p0 - gamepoint_p0) / 2f);
                P0_HP.GetComponent<RectTransform>().sizeDelta = new Vector2(gamepoint_p0, P0_HP.GetComponent<RectTransform>().sizeDelta.y);
                P0_HP.GetComponent<RectTransform>().localPosition = new Vector3(p0_x, P0_HP.GetComponent<RectTransform>().localPosition.y,
                    P0_HP.GetComponent<RectTransform>().localPosition.z);
                cur_gamepoint_p0 = gamepoint_p0;
            }
            if (cur_gamepoint_p1 - gamepoint_p1 > 0.9f)     // compare two float number (cannot use == or != , sometimes may cause error)
            {
                float p1_x = P1_HP.GetComponent<RectTransform>().localPosition.x + ((cur_gamepoint_p1 - gamepoint_p1) / 2f);
                P1_HP.GetComponent<RectTransform>().sizeDelta = new Vector2(gamepoint_p1, P1_HP.GetComponent<RectTransform>().sizeDelta.y);
                P1_HP.GetComponent<RectTransform>().localPosition = new Vector3(p1_x, P1_HP.GetComponent<RectTransform>().localPosition.y,
                    P1_HP.GetComponent<RectTransform>().localPosition.z);
                cur_gamepoint_p1 = gamepoint_p1;
            }

            
            if (Joints_co[2] == 0 && Joints_co[3] == 0)   // Neck
                debug_1.text = debug_1.text + "\nNo human";

            // draw joints
            for (int i = 0; i < Joints.Length; i++)
            {
                Joints[i].GetComponent<RectTransform>().localPosition = joint_coordinate_trans(Joints_co[i * 2], Joints_co[i * 2 + 1]);
                if (Joints_co[i * 2] == 0 && Joints_co[i * 2 + 1] == 0)
                    Joints[i].enabled = false;
                else
                    Joints[i].enabled = true;
            }


            if (animation_flag)       // 進入動畫模式
            {
                if (SK2_animation)
                {
                    float SK2_x = Skill_2.GetComponent<Transform>().localPosition.x;
                    float SK2_y = Skill_2.GetComponent<Transform>().localPosition.y;
                    float SK2_z = Skill_2.GetComponent<Transform>().localPosition.z;
                    SK2_x = SK2_x - SK2_reduce_x;
                    SK2_y = SK2_y - SK2_reduce_y;
                    SK2_z = SK2_z - SK2_reduce_z;
                    Skill_2.GetComponent<Transform>().localPosition = new Vector3(SK2_x, SK2_y, SK2_z);
                    
                    if (SK2_z <= SK2_limit)
                    {
                        status = replace_char(status, 0, "1");
                        status_s2_hit_time = Time.time;
                        animation_flag = false;
                        SK2_animation = false;
                        Skill_2.GetComponent<Transform>().localPosition = new Vector3(40f, 0f, 0f);
                        Skill_2.SetActive(false);
                    }
                }
                if (blood_animation)
                {
                    float blood_alpha = blood.color.a;
                    blood_alpha = blood_alpha - 0.02f;
                    blood.color = new Color(blood.color.r, blood.color.g, blood.color.b, blood_alpha);
                    if(blood_alpha <= 0f)
                    {
                        animation_flag = false;
                        blood_animation = false;
                    }
                }
                if (defense_animation)
                {
                    if ((Time.time - defense_time) >= 1.5f)
                    {
                        animation_flag = false;
                        defense_animation = false;
                        successful_defense.SetActive(false);
                    }
                }
                if (failed_attack_animation)
                {
                    if ((Time.time - failed_attack_time) >= 1.5f)
                    {
                        animation_flag = false;
                        failed_attack_animation = false;
                        failed_attack.SetActive(false);
                    }
                }

                if (my_SK2_animation)
                {
                    float SK2_z = my_Skill_2.GetComponent<Transform>().localPosition.z;
                    SK2_z = SK2_z + 1.5f;   // z 從 -40f 到 0f
                    my_Skill_2.GetComponent<Transform>().localPosition = new Vector3(1f, 0f, SK2_z);

                    if (SK2_z >= 0f)
                    {
                        animation_flag = false;
                        my_SK2_animation = false;
                        my_Skill_2.GetComponent<Transform>().localPosition = new Vector3(1f, 0f, -43f);
                        my_Skill_2.SetActive(false);
                    }
                }
            }
            else
            {
                if (action == 2)
                {
                    Skill_2.SetActive(true);
                    if (Joints_co[2] != 0 || Joints_co[3] != 0)         // 有人，(Neck_x,Neck_y)=(0,0)我們訂為沒偵測到人
                    {
                        if (Joints_co[8] != 0f || Joints_co[9] != 0f)   // 右手有偵測到才去改Skill 2的動畫位置，(RWrist_x,RWrist_y)=(0,0)代表沒偵測到
                        {
                            Skill_2.GetComponent<Transform>().localPosition = Skill_2_coordinate_trans(Joints_co[8], Joints_co[9], Joints_co[6], Joints_co[7]);  //RWrist
                        }
                    }
                }
                else if (action == 3)
                {
                    if (Skill_2.activeSelf == true)
                    {
                        //Skill_2.GetComponent<Transform>().localPosition = joint_coordinate_trans(Joints_co[8], Joints_co[9]);  //RWrist
                        animation_flag = true;
                        SK2_animation = true;

                        SK2_limit = -40f;
                        SK2_reduce_z = 1.5f;
                        SK2_reduce_x = SK2_reduce_z / -SK2_limit * Skill_2.GetComponent<Transform>().localPosition.x;
                        SK2_reduce_y = SK2_reduce_z / -SK2_limit * Skill_2.GetComponent<Transform>().localPosition.y;
                    }
                }
                else   // 其餘的action
                {
                    Skill_2.SetActive(false);
                }


                if (my_action == 2)
                {
                    my_Skill_2.SetActive(true);
                    my_Skill_2.GetComponent<Transform>().localPosition = new Vector3(100f, 0f, 0f);
                }
                else if (my_action == 3)
                {
                    if (my_Skill_2.activeSelf == true)
                    {
                        //Skill_2.GetComponent<Transform>().localPosition = joint_coordinate_trans(Joints_co[8], Joints_co[9]);  //RWrist
                        animation_flag = true;
                        my_SK2_animation = true;
                        my_Skill_2.GetComponent<Transform>().localPosition = new Vector3(2f, 0f, -40f);
                    }
                }
                else
                {
                    my_Skill_2.SetActive(false);
                }
            }

            if (player == "holo_P1")
            {
                if (p1_win_lose == 2 && end_game_flag == false)      // lose
                    StartCoroutine(Lose_Game());
                else if (p1_win_lose == 1 && end_game_flag == false)  // win
                    StartCoroutine(Win_Game());

                if (defense_skill_2_p1 == 1 && defense_animation == false)
                {
                    animation_flag = true;
                    defense_animation = true;
                    successful_defense.SetActive(true);
                    defense_time = Time.time;
                }
                else if (defense_skill_2_p0 == 1 && defense_animation == false)
                {
                    animation_flag = true;
                    failed_attack_animation = true;
                    failed_attack.SetActive(true);
                    failed_attack_time = Time.time;
                }

                if (blood_effect_p1 == 1 && blood_animation == false)
                {
                    animation_flag = true;
                    blood_animation = true;
                    blood.color = new Color(blood.color.r, blood.color.g, blood.color.b, 1f);
                }
            }
        }
    }


#if !UNITY_EDITOR

    /*
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
    */

    private async void StartClient(StreamSocket streamSocket)
    {
        try
        {
            using (streamSocket = new Windows.Networking.Sockets.StreamSocket()) {

                var hostName = new Windows.Networking.HostName(IP);
                await streamSocket.ConnectAsync(hostName, port);

                // tell server who you are
                using (var dw = new DataWriter(streamSocket.OutputStream))
                {
                    dw.WriteBytes(Encoding.ASCII.GetBytes(player));
                    await dw.StoreAsync();
                    dw.DetachStream();
                }
                using (var dr = new DataReader(streamSocket.InputStream))
                {
                    dr.InputStreamOptions = InputStreamOptions.Partial;
                    uint uintBytes = await dr.LoadAsync(1024);
                    string input = dr.ReadString(uintBytes);
                    dr.DetachStream();
                }

                // wait for final
                while(true) {
                    if (final == null){
                        await System.Threading.Tasks.Task.Delay(200);
                        continue;
                    }
                    else{
                        break;
                    }
                }

                connect = true;

                while (true) {
                    using (var dw = new DataWriter(streamSocket.OutputStream))
                    {
                        while(true){
                            if (lock_data == 1)
                                continue;
                            else{
                                dw.WriteBytes(final);
                                await dw.StoreAsync();
                                dw.DetachStream();
                                break;
                            }
                        }
                    }
                    
                    
                    using (var dr = new DataReader(streamSocket.InputStream))
                    {
                        dr.InputStreamOptions = InputStreamOptions.Partial;
                        uint uintBytes = await dr.LoadAsync(1024);
                        string input = dr.ReadString(uintBytes);
                        dr.DetachStream();

                        str_text = input;
                        string[] arr = input.Split(',');

                        for (int i = 0 ; i < Joints.Length * 2 ; i++) {
                            Joints_co[i] = float.Parse(arr[i]);
                        }
                        
                        recv_frame = int.Parse(arr[36]);
                        holo_action_p0 = int.Parse(arr[37]);
                        holo_action_p1 = int.Parse(arr[38]);
                        gamepoint_p0 = float.Parse(arr[39]);
                        gamepoint_p1 = float.Parse(arr[40]);
                        p0_win_lose = int.Parse(arr[41]);
                        p1_win_lose = int.Parse(arr[42]);
                        defense_skill_2_p0 = int.Parse(arr[43]);
                        defense_skill_2_p1 = int.Parse(arr[44]);
                        blood_effect_p0 = int.Parse(arr[45]);
                        blood_effect_p1 = int.Parse(arr[46]);

                        if(player == "holo_P0"){
                            action = holo_action_p1;
                            my_action = holo_action_p0;
                            debug_4.text = arr[47] + "," + arr[48] + "," + arr[49] + "," + arr[50] + "," + arr[51] + "," + arr[52];
                            if (arr[47] == "1"){                          // arr[47] is status[0]
                                status = replace_char(status, 0, "0");
                            }

                            if (p0_win_lose == 2)                         // lose
                                StartCoroutine(Lose_Game());
                            else if (p0_win_lose == 1)                    // win
                                StartCoroutine(Win_Game());

                            if (defense_skill_2_p0 == 1) {
                                animation_flag = true;
                                defense_animation = true;
                                successful_defense.SetActive(true);
                                defense_time = Time.time;
                            }
                            else if (defense_skill_2_p1 == 1) {
                                animation_flag = true;
                                failed_attack_animation = true;
                                failed_attack.SetActive(true);
                                failed_attack_time = Time.time;
                            }

                            if (blood_effect_p0 == 1) {
                                animation_flag = true;
                                blood_animation = true;
                                blood.color = new Color(blood.color.r, blood.color.g, blood.color.b, 1f);
                            }
                        }
                        else if(player == "holo_P1"){
                            action = holo_action_p0;
                            my_action = holo_action_p1;
                        }
    

                        frame_ready = 1;
                    
                        fps = 10000000.0 / (System.Diagnostics.Stopwatch.GetTimestamp() - local_fps);
                        local_fps = System.Diagnostics.Stopwatch.GetTimestamp();
    
                    }
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
