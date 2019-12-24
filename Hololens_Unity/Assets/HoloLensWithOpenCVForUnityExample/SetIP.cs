using UnityEngine;
using UnityEngine.UI;
using System.Collections;

namespace HoloLensWithOpenCVForUnityExample
{
    public class SetIP : ExampleSceneBase
    {
        private string s;
        public Text IP;
        public Text Player;
        public Text Enter_text;
        // Use this for initialization
        protected override void Start ()
        {
            base.Start ();

            s = Game_Stats.IP;
            IP.text = s;
            if (Game_Stats.PlayerID == "holo_P0")
                Player.text = "P1";
            else if (Game_Stats.PlayerID == "holo_P1")
                Player.text = "P2";
            else
                Player.text = "Choose Player";
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
            if(Enter_text.text == "Type Error")
            {
                Enter_text.text = "Enter";
                Enter_text.color = Color.black;
            }
            else
            {
                string[] arr = s.Split('.');
                if (arr.Length == 4)
                {
                    if (Game_Stats.PlayerID != "None")
                    {
                        Game_Stats.IP = s;
                        Game_Stats.IP_SET = true;
                        LoadScene("HoloLensWithOpenCVForUnityExample");
                    }
                }
                else
                {
                    Enter_text.text = "Tyep Error";
                    Enter_text.color = Color.red;
                }
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
                Game_Stats.IP_SET = true;
                LoadScene("HoloLensWithOpenCVForUnityExample");
            }
        }

        public void On_P0_Click()
        {
            Game_Stats.PlayerID = "holo_P0";
            Player.text = "P1";
        }

        public void On_P1_Click()
        {
            Game_Stats.PlayerID = "holo_P1";
            Player.text = "P2";
        }
    }
}
