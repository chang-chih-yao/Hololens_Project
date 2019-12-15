﻿//  
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

    // status = "被對方的技能(ATK1)打到, 被對方的技能(ATK2)打到, 被對方的技能(ATK3)打到, 被對方的技能(ATK4)打到, end game, start new game"
    private string status="0,0,0,0,0,0";


    HoloLensCameraStream.Resolution _resolution;

    //"Injected" objects.
    VideoPanel _videoPanelUI;
    VideoCapture _videoCapture;

    IntPtr _spatialCoordinateSystemPtr;
    
    private string str_text;
    private Boolean connect = false;
    private Boolean animation_flag = false;
    private Boolean ATK_1_animation = false;
    private Boolean my_ATK_1_animation = false;
    //private Boolean SK2_fail_flag = false;
    private Boolean my_SK2_animation = false;
    //private Boolean my_SK2_fail_flag = false;
    private Boolean blood_animation = false;
    private Boolean defense_animation = false;
    private Boolean failed_attack_animation = false;
    private Boolean end_game_flag = false;                // 之後要刪掉，把P1的移到下方去


    public Text debug_1;
    public Text debug_2;
    public Text debug_3;
    public Text debug_4;

    public Image[] Joints;
    public Image P0_HP;
    public Image P1_HP;

    public GameObject Debug_Spere;
    //public GameObject Debug_Spere2;

    public GameObject Skill_Ref;
    public GameObject my_Skill_Ref;
    public GameObject Atk_1;
    public GameObject Atk_2;
    public GameObject Atk_3;
    public GameObject Atk_4;
    public GameObject Def_1;
    public GameObject Def_2;
    public GameObject Def_3;
    public GameObject Def_4;

    private GameObject skill_temp;
    private GameObject my_skill_temp;
    private Rasengan rasengan;
    private Rasengan my_rasengan;
    private Orb orb;
    private Orb my_orb;
    private MagicCircle magicCircle;
    private MagicCircle my_magicCircle;

    public GameObject panel;
    public GameObject image_panel;
    public GameObject canvas_openpose;
    public GameObject win_img;
    public GameObject lose_img;
    public GameObject start_new_game;
    public GameObject successful_defense;
    public GameObject failed_attack;

    public Material line_mat;

    public SpriteRenderer blood;
    
    private float[] Joints_co;
    private float[] height_window;

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
    private float video_panel_z;
    private float pre_z;
    private float temp_pre_z;

    private float status_s2_hit_time;
    private float defense_time;
    private float failed_attack_time;

    private float P0_HP_X;
    private float P1_HP_X;

#if !UNITY_EDITOR
    String port;
    String IP;
    StreamSocket streamSocket;
#endif

    void Start()
    {
        skill_temp = Instantiate(Atk_1);
        rasengan = skill_temp.GetComponent<Rasengan>();
        my_rasengan = skill_temp.GetComponent<Rasengan>();
        Destroy(skill_temp);

        skill_temp = Instantiate(Def_1);
        magicCircle = skill_temp.GetComponent<MagicCircle>();
        my_magicCircle = skill_temp.GetComponent<MagicCircle>();
        Destroy(skill_temp);

        Joints_co = new float[Joints.Length * 2];
        for (int i = 0; i < Joints.Length * 2; i++)
            Joints_co[i] = 0f;
        
        recv_frame = 1;
        recv_frame_old = 0;
        P0_HP_X = P0_HP.GetComponent<RectTransform>().localPosition.x;
        P1_HP_X = P1_HP.GetComponent<RectTransform>().localPosition.x;

        player = Game_Stats.PlayerID;
        game_status = new byte[256];

        height_window = new float[5];
        for (int i = 0; i < 5; i++)
            height_window[i] = 175f;

        video_panel_z = 80f;
        panel.GetComponent<Transform>().localPosition = new Vector3(0f, 0f, video_panel_z);

        if (Game_Stats.DEMO)     // 有 image panel
        {
            pre_z = 0f;
            canvas_openpose.GetComponent<Transform>().localPosition = new Vector3(0, 0, pre_z);
            image_panel.SetActive(true);
        }
        else
        {
            pre_z = Game_Stats.Pre_z;
            canvas_openpose.GetComponent<Transform>().localPosition = new Vector3(0, 0, pre_z);
            image_panel.SetActive(false);
        }

        if (Game_Stats.Debug)
        {
            debug_1.enabled = true;
            debug_2.enabled = true;
            debug_3.enabled = true;
            debug_4.enabled = true;
        }
        else
        {
            debug_1.enabled = false;
            debug_2.enabled = false;
            debug_3.enabled = false;
            debug_4.enabled = false;
        }

        if (Game_Stats.Draw_skeleton)
            My_Line.Draw_skeleton = true;
        else
            My_Line.Draw_skeleton = false;



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
        if (Game_Stats.DEMO)
            return new Vector3(x / 10.0f - 22.4f, -(y / 10.0f - 12.6f), 0f);
        else
            return new Vector3(x / 10.0f - 22.4f + 2f, -(y / 10.0f - 12.6f + 0.7f), 0f);
        
    }

    private Vector3 Skill_2_coordinate_trans(Vector3 RWrist, Vector3 RElbow)
    {
        /*if (Game_Stats.DEMO)
        {
            Vector3 RWrist = new Vector3(RWrist_x / 10.0f - 22.4f, -(RWrist_y / 10.0f - 12.6f), pre_z);
            Vector3 RElbow = new Vector3(RElbow_x / 10.0f - 22.4f, -(RElbow_y / 10.0f - 12.6f), pre_z);
            Vector3 my_vec = RWrist - RElbow;
            return RElbow + 1.2f * my_vec;
        }
        else
        {
            Vector3 RWrist = new Vector3(RWrist_x / 10.0f - 22.4f + 2f, -(RWrist_y / 10.0f - 12.6f + 0.7f), pre_z);
            Vector3 RElbow = new Vector3(RElbow_x / 10.0f - 22.4f + 2f, -(RElbow_y / 10.0f - 12.6f + 0.7f), pre_z);
            Vector3 my_vec = RWrist - RElbow;
            return RElbow + 1.2f * my_vec;
        }*/
        Vector3 new_RWrist = new Vector3(RWrist.x, RWrist.y, pre_z);
        Vector3 new_RElbow = new Vector3(RElbow.x, RElbow.y, pre_z);
        Vector3 my_vec = new_RWrist - new_RElbow;
        return new_RElbow + 1.2f * my_vec;
    }

    private void Destroy()
    {
        Skill_Ref.GetComponent<Transform>().localPosition = new Vector3(0f, 0f, pre_z);
        if (rasengan.isAlive() == true)
            rasengan.finish(0.1f);
        else if (skill_temp.name == "Orb(Clone)")   // 之後要改成 orb.isAlive() == true
            orb.finish(0.1f);
        else if (skill_temp.name == "MagicCircle_Red(Clone)" && magicCircle.isAlive() == true)
            magicCircle.finish(0.1f);
        else if (skill_temp.name == "MagicCircle_Purple(Clone)" && magicCircle.isAlive() == true)
            magicCircle.finish(0.1f);
        else if (skill_temp.name == "MagicCircle_Orange(Clone)" && magicCircle.isAlive() == true)
            magicCircle.finish(0.1f);
        else if (skill_temp.name == "MagicCircle_Green(Clone)" && magicCircle.isAlive() == true)
            magicCircle.finish(0.1f);
    }

    private void my_Destroy()
    {
        my_Skill_Ref.GetComponent<Transform>().localPosition = new Vector3(0f, 0f, pre_z);
        if (my_rasengan.isAlive() == true)
            my_rasengan.finish(0.1f);
        else if (my_skill_temp.name == "Orb(Clone)")   // 之後要改成 orb.isAlive() == true
            my_orb.finish(0.1f);
        else if (my_skill_temp.name == "MagicCircle_Red(Clone)" && my_magicCircle.isAlive() == true)
            my_magicCircle.finish(0.1f);
        else if (my_skill_temp.name == "MagicCircle_Purple(Clone)" && my_magicCircle.isAlive() == true)
            my_magicCircle.finish(0.1f);
        else if (my_skill_temp.name == "MagicCircle_Orange(Clone)" && my_magicCircle.isAlive() == true)
            my_magicCircle.finish(0.1f);
        else if (my_skill_temp.name == "MagicCircle_Green(Clone)" && my_magicCircle.isAlive() == true)
            my_magicCircle.finish(0.1f);
    }

    private IEnumerator Lose_Game()
    {
        status = replace_char(status, 10, "1");

        //My_Line.Draw_skeleton = false;
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
        //My_Line.Draw_skeleton = true;

        status = replace_char(status, 10, "0");
    }

    private IEnumerator Win_Game()
    {
        status = replace_char(status, 10, "1");

        //My_Line.Draw_skeleton = false;
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
        //My_Line.Draw_skeleton = true;

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
            //pre_z = pre_z - 0.01f;
            //panel.GetComponent<Transform>().localPosition = new Vector3(0f, 0f, pre_z);
            //debug_4.text = "blood_effect\n" + blood_effect_p0.ToString() + ", " + blood_effect_p1.ToString();
            canvas_openpose.GetComponent<Transform>().localPosition = new Vector3(0, 0, pre_z);
            //Debug_Spere2.GetComponent<Transform>().localPosition = new Vector3(2f, 2f, pre_z);

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


            // draw joints
            for (int i = 0; i < Joints.Length; i++)
            {
                Joints[i].GetComponent<RectTransform>().localPosition = joint_coordinate_trans(Joints_co[i * 2], Joints_co[i * 2 + 1]);
                if (Joints_co[i * 2] == 0 && Joints_co[i * 2 + 1] == 0)
                    Joints[i].enabled = false;
                else
                    Joints[i].enabled = true;
            }

            if (Joints_co[2] == 0 && Joints_co[3] == 0 && Joints_co[16] == 0 && Joints_co[17] == 0 && Joints_co[22] == 0 && Joints_co[23] == 0)  // Neck, RHip, LHip
                debug_1.text = "No human";
            else
            {
                if ((Joints_co[20] == 0 && Joints_co[21] == 0) || (Joints_co[26] == 0 && Joints_co[27] == 0) || (Joints_co[0] == 0 && Joints_co[1] == 0))
                    debug_1.text = "Cannot capture the entire human";
                else
                {
                    float avg_foot_y = (Joints_co[21] + Joints_co[27]) / 2f;
                    float human_height = avg_foot_y - Joints_co[1];                  // Human height = Feet - Nose
                    height_window[0] = height_window[1];
                    height_window[1] = height_window[2];
                    height_window[2] = height_window[3];
                    height_window[3] = height_window[4];
                    height_window[4] = human_height;
                    float avg_height = (height_window[0] + height_window[1] + height_window[2] + height_window[3] + height_window[4]) / 5f;
                    debug_4.text = avg_height.ToString();
                    if (avg_height <= 100f)
                        debug_1.text = "Please stand closer";
                    if (Game_Stats.DEMO == false)
                        pre_z = (((video_panel_z + Game_Stats.Pre_z) * (Game_Stats.Height / 10f)) / (avg_height / 10f)) - video_panel_z;
                }
            }

            debug_1.text += "\n";
            debug_1.text += pre_z.ToString();

            if (animation_flag)       // 進入動畫模式
            {
                if (ATK_1_animation)
                {
                    Skill_Ref.GetComponent<Transform>().localPosition = Vector3.MoveTowards(Skill_Ref.GetComponent<Transform>().localPosition, new Vector3(0f, 0f, -video_panel_z), 2f);
                    rasengan.setPosition(Skill_Ref.GetComponent<Transform>().position);
                    if (Skill_Ref.GetComponent<Transform>().localPosition.z < (5f - video_panel_z))     // 在鏡頭前面5單位
                    {
                        rasengan.finish(0.1f);
                        Skill_Ref.GetComponent<Transform>().localPosition = new Vector3(0f, 0f, pre_z);
                        status = replace_char(status, 0, "1");
                        status_s2_hit_time = Time.time;
                        animation_flag = false;
                        ATK_1_animation = false;
                    }
                }
                if (my_ATK_1_animation)
                {
                    my_Skill_Ref.GetComponent<Transform>().localPosition = Vector3.MoveTowards(my_Skill_Ref.GetComponent<Transform>().localPosition, new Vector3(0f, 0f, 0f), 2f);
                    my_rasengan.setPosition(my_Skill_Ref.GetComponent<Transform>().position);
                    if (my_Skill_Ref.GetComponent<Transform>().localPosition.z > (temp_pre_z - 5f))
                    {
                        my_rasengan.finish(0.1f);
                        my_Skill_Ref.GetComponent<Transform>().localPosition = new Vector3(0f, 0f, pre_z);
                        animation_flag = false;
                        my_ATK_1_animation = false;
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
                    /*
                    float SK2_z = my_Skill_2.GetComponent<Transform>().localPosition.z;
                    SK2_z = SK2_z + 1.5f;   // z 從 -40f 到 0f
                    my_Skill_2.GetComponent<Transform>().localPosition = new Vector3(1f, 0f, SK2_z);

                    if (SK2_z >= 0f)
                    {
                        animation_flag = false;
                        my_SK2_animation = false;
                        my_Skill_2.GetComponent<Transform>().localPosition = new Vector3(1f, 0f, -43f);
                        my_Skill_2.SetActive(false);
                    }*/
                }
            }
            else
            {
                // **************************** 對手的 action **************************** //
                if (action == 2)
                {
                    /*Debug_Spere.SetActive(true);
                    if (Joints_co[2] != 0 || Joints_co[3] != 0)         // 有人，(Neck_x,Neck_y)=(0,0)我們訂為沒偵測到人
                    {
                        if (Joints_co[8] != 0f || Joints_co[9] != 0f)   // 右手有偵測到才去改Skill 2的動畫位置，(RWrist_x,RWrist_y)=(0,0)代表沒偵測到
                        {
                            Skill_Ref.GetComponent<Transform>().localPosition =
                                    Skill_2_coordinate_trans(Joints[4].GetComponent<Transform>().localPosition, Joints[3].GetComponent<Transform>().localPosition);
                            Debug_Spere.GetComponent<Transform>().localPosition = Skill_Ref.GetComponent<Transform>().localPosition;
                        }
                    }*/

                    if (skill_temp == null)
                    {
                        skill_temp = Instantiate(Atk_1);
                        rasengan = skill_temp.GetComponent<Rasengan>();
                        rasengan.ready(Skill_Ref.GetComponent<Transform>().position, new Vector3(0f, 0f, 1f), 0.1f, 3.0f);
                    }
                    else
                    {
                        if (rasengan.isAlive() == true)
                        {
                            if (Joints_co[8] != 0f || Joints_co[9] != 0f)   // 右手有偵測到才去改Skill 2的動畫位置，(RWrist_x,RWrist_y)=(0,0)代表沒偵測到
                            {
                                Skill_Ref.GetComponent<Transform>().localPosition = 
                                    Skill_2_coordinate_trans(Joints[4].GetComponent<Transform>().localPosition, Joints[3].GetComponent<Transform>().localPosition);
                                rasengan.setPosition(Skill_Ref.GetComponent<Transform>().position);
                            }
                        }
                        else
                            Destroy();
                    }
                }
                else if (action == 3)
                {
                    if (skill_temp != null)
                    {
                        if (rasengan.isAlive() == true)
                        {
                            animation_flag = true;
                            ATK_1_animation = true;
                        }
                        else
                            Destroy();
                    }
                }
                else if (action == 8)
                {
                    if (skill_temp == null)
                    {
                        skill_temp = Instantiate(Def_1);
                        magicCircle = skill_temp.GetComponent<MagicCircle>();
                        magicCircle.ready(Skill_Ref.GetComponent<Transform>().position, new Vector3(0f, 0f, 1f), 0.1f, 3.0f);
                    }
                    else
                    {
                        if (skill_temp.name == "MagicCircle_Purple(Clone)" && magicCircle.isAlive() == true)
                        {
                            if (Joints_co[2] != 0f || Joints_co[3] != 0f)
                            {
                                Skill_Ref.GetComponent<Transform>().localPosition = 
                                    new Vector3(Joints[1].GetComponent<Transform>().localPosition.x, Joints[1].GetComponent<Transform>().localPosition.y, pre_z);
                                magicCircle.setPosition(Skill_Ref.GetComponent<Transform>().position);
                            }
                        }
                        else
                            Destroy();
                    }
                }
                else if (action == 9)
                {
                    if (skill_temp == null)
                    {
                        skill_temp = Instantiate(Def_2);
                        magicCircle = skill_temp.GetComponent<MagicCircle>();
                        magicCircle.ready(Skill_Ref.GetComponent<Transform>().position, new Vector3(0f, 0f, 1f), 0.1f, 3.0f);
                    }
                    else
                    {
                        if (skill_temp.name == "MagicCircle_Orange(Clone)" && magicCircle.isAlive() == true)
                        {
                            if (Joints_co[2] != 0f || Joints_co[3] != 0f)
                            {
                                Skill_Ref.GetComponent<Transform>().localPosition =
                                    new Vector3(Joints[1].GetComponent<Transform>().localPosition.x, Joints[1].GetComponent<Transform>().localPosition.y, pre_z);
                                magicCircle.setPosition(Skill_Ref.GetComponent<Transform>().position);
                            }
                        }
                        else
                            Destroy();
                    }
                }
                else if (action == 10)
                {
                    if (skill_temp == null)
                    {
                        skill_temp = Instantiate(Def_3);
                        magicCircle = skill_temp.GetComponent<MagicCircle>();
                        magicCircle.ready(Skill_Ref.GetComponent<Transform>().position, new Vector3(0f, 0f, 1f), 0.1f, 3.0f);
                    }
                    else
                    {
                        if (skill_temp.name == "MagicCircle_Red(Clone)" && magicCircle.isAlive() == true)
                        {
                            if (Joints_co[2] != 0f || Joints_co[3] != 0f)
                            {
                                Skill_Ref.GetComponent<Transform>().localPosition =
                                    new Vector3(Joints[1].GetComponent<Transform>().localPosition.x, Joints[1].GetComponent<Transform>().localPosition.y, pre_z);
                                magicCircle.setPosition(Skill_Ref.GetComponent<Transform>().position);
                            }
                        }
                        else
                            Destroy();
                    }
                }
                else if (action == 11)
                {
                    if (skill_temp == null)
                    {
                        skill_temp = Instantiate(Def_4);
                        magicCircle = skill_temp.GetComponent<MagicCircle>();
                        magicCircle.ready(Skill_Ref.GetComponent<Transform>().position, new Vector3(0f, 0f, 1f), 0.1f, 3.0f);
                    }
                    else
                    {
                        if (skill_temp.name == "MagicCircle_Green(Clone)" && magicCircle.isAlive() == true)
                        {
                            if (Joints_co[2] != 0f || Joints_co[3] != 0f)
                            {
                                Skill_Ref.GetComponent<Transform>().localPosition =
                                    new Vector3(Joints[1].GetComponent<Transform>().localPosition.x, Joints[1].GetComponent<Transform>().localPosition.y, pre_z);
                                magicCircle.setPosition(Skill_Ref.GetComponent<Transform>().position);
                            }
                        }
                        else
                            Destroy();
                    }
                }
                else if (action == 1)   // 其餘的action
                {
                    //Debug_Spere.SetActive(false);
                    if (skill_temp != null)
                    {
                        Destroy();
                    }
                }


                // **************************** 自己的 action **************************** //
                if (my_action == 2)
                {
                    if (my_skill_temp == null)
                    {
                        my_skill_temp = Instantiate(Atk_1);
                        my_rasengan = my_skill_temp.GetComponent<Rasengan>();
                        my_Skill_Ref.GetComponent<Transform>().localPosition = new Vector3(1.5f, -1f, 5f - video_panel_z);     // 在鏡頭前面5單位
                        my_rasengan.ready(my_Skill_Ref.GetComponent<Transform>().position, new Vector3(0f, 0f, 1f), 0.1f, 3.0f);
                    }
                    else
                    {
                        if (my_rasengan.isAlive() == true)
                        {
                            my_Skill_Ref.GetComponent<Transform>().localPosition = new Vector3(1.5f, -1f, 5f - video_panel_z);     // 在鏡頭前面5單位
                            my_rasengan.setPosition(my_Skill_Ref.GetComponent<Transform>().position);
                        }
                        else
                            my_Destroy();
                    }
                }
                else if (my_action == 3)
                {
                    if (my_skill_temp != null)
                    {
                        if (my_rasengan.isAlive() == true)
                        {
                            animation_flag = true;
                            my_ATK_1_animation = true;
                            temp_pre_z = pre_z;
                        }
                        else
                            my_Destroy();
                    }
                }
                else if (my_action == 8)
                {
                    if (my_skill_temp == null)
                    {
                        my_skill_temp = Instantiate(Def_1);
                        my_magicCircle = my_skill_temp.GetComponent<MagicCircle>();
                        my_Skill_Ref.GetComponent<Transform>().localPosition = new Vector3(0f, -0.4f, 5f - video_panel_z);     // 在鏡頭前面5單位
                        my_magicCircle.ready(my_Skill_Ref.GetComponent<Transform>().position, new Vector3(0f, 0f, -1f), 0.1f, 0.7f);
                    }
                    else
                    {
                        if (my_skill_temp.name == "MagicCircle_Purple(Clone)" && my_magicCircle.isAlive() == true)
                        {
                            my_Skill_Ref.GetComponent<Transform>().localPosition = new Vector3(0f, -0.4f, 5f - video_panel_z);     // 在鏡頭前面5單位
                            my_magicCircle.setPosition(my_Skill_Ref.GetComponent<Transform>().position);
                        }
                        else
                            my_Destroy();
                    }
                }
                else if (my_action == 9)
                {
                    if (my_skill_temp == null)
                    {
                        my_skill_temp = Instantiate(Def_2);
                        my_magicCircle = my_skill_temp.GetComponent<MagicCircle>();
                        my_Skill_Ref.GetComponent<Transform>().localPosition = new Vector3(0f, -0.4f, 5f - video_panel_z);     // 在鏡頭前面5單位
                        my_magicCircle.ready(my_Skill_Ref.GetComponent<Transform>().position, new Vector3(0f, 0f, -1f), 0.1f, 0.7f);
                    }
                    else
                    {
                        if (my_skill_temp.name == "MagicCircle_Orange(Clone)" && my_magicCircle.isAlive() == true)
                        {
                            my_Skill_Ref.GetComponent<Transform>().localPosition = new Vector3(0f, -0.4f, 5f - video_panel_z);     // 在鏡頭前面5單位
                            my_magicCircle.setPosition(my_Skill_Ref.GetComponent<Transform>().position);
                        }
                        else
                            my_Destroy();
                    }
                }
                else if (my_action == 10)
                {
                    if (my_skill_temp == null)
                    {
                        my_skill_temp = Instantiate(Def_3);
                        my_magicCircle = my_skill_temp.GetComponent<MagicCircle>();
                        my_Skill_Ref.GetComponent<Transform>().localPosition = new Vector3(0f, -0.4f, 5f - video_panel_z);     // 在鏡頭前面5單位
                        my_magicCircle.ready(my_Skill_Ref.GetComponent<Transform>().position, new Vector3(0f, 0f, -1f), 0.1f, 0.7f);
                    }
                    else
                    {
                        if (my_skill_temp.name == "MagicCircle_Red(Clone)" && my_magicCircle.isAlive() == true)
                        {
                            my_Skill_Ref.GetComponent<Transform>().localPosition = new Vector3(0f, -0.4f, 5f - video_panel_z);     // 在鏡頭前面5單位
                            my_magicCircle.setPosition(my_Skill_Ref.GetComponent<Transform>().position);
                        }
                        else
                            my_Destroy();
                    }
                }
                else if (my_action == 11)
                {
                    if (my_skill_temp == null)
                    {
                        my_skill_temp = Instantiate(Def_4);
                        my_magicCircle = my_skill_temp.GetComponent<MagicCircle>();
                        my_Skill_Ref.GetComponent<Transform>().localPosition = new Vector3(0f, -0.4f, 5f - video_panel_z);     // 在鏡頭前面5單位
                        my_magicCircle.ready(my_Skill_Ref.GetComponent<Transform>().position, new Vector3(0f, 0f, -1f), 0.1f, 0.7f);
                    }
                    else
                    {
                        if (my_skill_temp.name == "MagicCircle_Green(Clone)" && my_magicCircle.isAlive() == true)
                        {
                            my_Skill_Ref.GetComponent<Transform>().localPosition = new Vector3(0f, -0.4f, 5f - video_panel_z);     // 在鏡頭前面5單位
                            my_magicCircle.setPosition(my_Skill_Ref.GetComponent<Transform>().position);
                        }
                        else
                            my_Destroy();
                    }
                }
                else if (my_action == 1)   // 其餘的action
                {
                    //my_Skill_2.SetActive(false);
                    if (my_skill_temp != null)
                    {
                        my_Destroy();
                    }
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
                        //pre_z = float.Parse(arr[47]);

                        if(player == "holo_P0"){
                            action = holo_action_p1;
                            my_action = holo_action_p0;

                            //debug_4.text = arr[47] + "," + arr[48] + "," + arr[49] + "," + arr[50] + "," + arr[51] + "," + arr[52];
                            if (arr[47] == "1"){                          // arr[47] is status[0]
                                status = replace_char(status, 0, "0");
                            }
                            if (arr[48] == "1"){                          // arr[48] is status[2]
                                status = replace_char(status, 2, "0");
                            }
                            if (arr[49] == "1"){                          // arr[49] is status[4]
                                status = replace_char(status, 4, "0");
                            }
                            if (arr[50] == "1"){                          // arr[50] is status[6]
                                status = replace_char(status, 6, "0");
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

                            //debug_4.text = arr[53] + "," + arr[54] + "," + arr[55] + "," + arr[56] + "," + arr[57] + "," + arr[58];
                            if (arr[53] == "1"){                          // arr[53] is status[0]
                                status = replace_char(status, 0, "0");
                            }
                            if (arr[54] == "1"){                          // arr[54] is status[2]
                                status = replace_char(status, 2, "0");
                            }
                            if (arr[55] == "1"){                          // arr[55] is status[4]
                                status = replace_char(status, 4, "0");
                            }
                            if (arr[56] == "1"){                          // arr[56] is status[6]
                                status = replace_char(status, 6, "0");
                            }

                            if (p1_win_lose == 2)                         // lose
                                StartCoroutine(Lose_Game());
                            else if (p1_win_lose == 1)                    // win
                                StartCoroutine(Win_Game());

                            if (defense_skill_2_p1 == 1) {
                                animation_flag = true;
                                defense_animation = true;
                                successful_defense.SetActive(true);
                                defense_time = Time.time;
                            }
                            else if (defense_skill_2_p0 == 1) {
                                animation_flag = true;
                                failed_attack_animation = true;
                                failed_attack.SetActive(true);
                                failed_attack_time = Time.time;
                            }

                            if (blood_effect_p1 == 1) {
                                animation_flag = true;
                                blood_animation = true;
                                blood.color = new Color(blood.color.r, blood.color.g, blood.color.b, 1f);
                            }
                        }
                    
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