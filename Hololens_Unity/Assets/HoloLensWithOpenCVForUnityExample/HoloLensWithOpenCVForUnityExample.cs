using UnityEngine;
using System.Collections;
using UnityEngine.UI;
using OpenCVForUnity.CoreModule;

namespace HoloLensWithOpenCVForUnityExample
{
    /// <summary>
    /// HoloLensWithOpenCVForUnity Example
    /// </summary>
    public class HoloLensWithOpenCVForUnityExample : ExampleSceneBase
    {
        public Text exampleTitle;
        public Text versionInfo;
        public Text Start_Game;
        public ScrollRect scrollRect;
        static float verticalNormalizedPosition = 1f;

        // Use this for initialization
        protected override void Start ()
        {
            base.Start ();

            if (Game_Stats.IP_SET == true)
            {
                string s = Start_Game.text;
                if (Game_Stats.PlayerID == "holo_P0")
                    s = s + ", IP : " + Game_Stats.IP + " (You select P1)";
                else
                    s = s + ", IP : " + Game_Stats.IP + " (You select P2)";
                Start_Game.text = s;
            }
            else
            {
                string s = Start_Game.text;
                s = s + ", IP : None";
                Start_Game.text = s;
            }

            //exampleTitle.text = "HoloLens MR Game " + Application.version;
            exampleTitle.text = "HoloLens MR Battle Game";

            //versionInfo.text = Core.NATIVE_LIBRARY_NAME + " " + OpenCVForUnity.UnityUtils.Utils.getVersion() + " (" + Core.VERSION + ")";
            versionInfo.text = "UnityEditor " + Application.unityVersion;
            versionInfo.text += " / ";

            #if UNITY_EDITOR
            versionInfo.text += "Editor";
            #elif UNITY_STANDALONE_WIN
            versionInfo.text += "Windows";
            #elif UNITY_STANDALONE_OSX
            versionInfo.text += "Mac OSX";
            #elif UNITY_STANDALONE_LINUX
            versionInfo.text += "Linux";
            #elif UNITY_ANDROID
            versionInfo.text += "Android";
            #elif UNITY_IOS
            versionInfo.text += "iOS";
            #elif UNITY_WSA
            versionInfo.text += "WSA";
            #elif UNITY_WEBGL
            versionInfo.text += "WebGL";
            #endif
            versionInfo.text += " ";
            #if ENABLE_MONO
            versionInfo.text += "Mono";
            #elif ENABLE_IL2CPP
            versionInfo.text += "IL2CPP";
            #elif ENABLE_DOTNET
            versionInfo.text += ".NET";
            #endif

            scrollRect.verticalNormalizedPosition = verticalNormalizedPosition;
        }
        
        // Update is called once per frame
        void Update ()
        {
            
        }

        public void OnScrollRectValueChanged()
        {
            verticalNormalizedPosition = scrollRect.verticalNormalizedPosition;
        }

        public void OnStartGame_Click()
        {
            if (Game_Stats.IP != "192.168.")
                LoadScene(2);
        }
        public void OnSetIP_Click()
        {
            LoadScene("SetIP");
        }
        public void On_Setting_Click()
        {
            LoadScene("Setting");
        }
        public void On_Tutorial_Click()
        {
            LoadScene("Tutorial_Video");
        }




        /*
        public void OnShowLicenseButtonClick ()
        {
            LoadScene ("ShowLicense");
        }

        public void OnHoloLensPhotoCaptureExampleButtonClick ()
        {
            LoadScene ("HoloLensPhotoCaptureExample");
        }

        public void OnHoloLensComicFilterExampleButtonClick ()
        {
            LoadScene ("HoloLensComicFilterExample");
        }
        
        public void OnHoloLensFaceDetectionExampleButtonClick ()
        {
            LoadScene ("HoloLensFaceDetectionExample");
        }

        public void OnHoloLensFaceDetectionOverlayExampleButtonClick ()
        {
            LoadScene ("HoloLensFaceDetectionOverlayExample");
        }

        public void OnHoloLensArUcoExampleButtonClick ()
        {
            LoadScene ("HoloLensArUcoExample");
        }

        public void OnHoloLensArUcoCameraCalibrationExampleButtonClick ()
        {
            LoadScene ("HoloLensArUcoCameraCalibrationExample");
        }
        */
    }
}