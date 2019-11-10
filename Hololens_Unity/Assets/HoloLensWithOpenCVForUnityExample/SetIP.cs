using UnityEngine;
using UnityEngine.UI;
using System.Collections;

namespace HoloLensWithOpenCVForUnityExample
{
    public class SetIP : ExampleSceneBase
    {
        private string s = "192.168.";
        public Text IP;
        public Text Player;
        // Use this for initialization
        protected override void Start ()
        {
            base.Start ();
            IP.text = s;
        }
        
        // Update is called once per frame
        void Update ()
        {
            
        }

        /// <summary>
        /// Raises the back button click event.
        /// </summary>
        public void OnBackButtonClick ()
        {
            LoadScene ("HoloLensWithOpenCVForUnityExample");
        }

        public void On_0()
        {
            s = s + "0";
            IP.text = s;
        }

        public void On_1()
        {
            s = s + "1";
            IP.text = s;
        }

        public void On_2()
        {
            s = s + "2";
            IP.text = s;
        }

        public void On_3()
        {
            s = s + "3";
            IP.text = s;
        }

        public void On_4()
        {
            s = s + "4";
            IP.text = s;
        }

        public void On_5()
        {
            s = s + "5";
            IP.text = s;
        }

        public void On_6()
        {
            s = s + "6";
            IP.text = s;
        }

        public void On_7()
        {
            s = s + "7";
            IP.text = s;
        }

        public void On_8()
        {
            s = s + "8";
            IP.text = s;
        }

        public void On_9()
        {
            s = s + "9";
            IP.text = s;
        }

        public void On_Dot_Click()
        {
            s = s + ".";
            IP.text = s;
        }

        public void On_Enter_Click()
        {
            string[] arr = s.Split('.');
            if (arr.Length == 4)
            {
                if (Game_Stats.PlayerID != "None")
                {
                    Game_Stats.IP = s;
                    LoadScene("HoloLensWithOpenCVForUnityExample");
                }
            }
            else
            {
                s = "Please input correct IP address";
                IP.text = s;
            }
        }

        public void On_BackSpace_Click()
        {
            if (s.Length != 0)
            {
                s = s.Remove(s.Length - 1, 1);
                IP.text = s;
            }
        }

        public void On_Clear_Click()
        {
            s = "";
            IP.text = s;
        }

        public void On_Custom_Click()
        {
            if (Game_Stats.PlayerID != "None")
            {
                Game_Stats.IP = "192.168.60.2";
                LoadScene("HoloLensWithOpenCVForUnityExample");
            }
        }

        public void On_P0_Click()
        {
            Game_Stats.PlayerID = "holo_P0";
            Player.text = "P0";
        }

        public void On_P1_Click()
        {
            Game_Stats.PlayerID = "holo_P1";
            Player.text = "P1";
        }
    }
}
