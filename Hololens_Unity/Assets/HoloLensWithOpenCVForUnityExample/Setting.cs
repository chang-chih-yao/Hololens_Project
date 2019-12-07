using UnityEngine;
using UnityEngine.UI;
using System.Collections;

namespace HoloLensWithOpenCVForUnityExample
{
    public class Setting : ExampleSceneBase
    {
        public Text demo_text;
        public Text debug_text;
        public Text draw_skeleton;
        public Text pre_z;
        public Text height;
        protected override void Start()
        {
            base.Start();
            if (Game_Stats.DEMO)
            {
                demo_text.text = "Demo Mode : True";
                demo_text.color = Color.green;
            }
            else
            {
                demo_text.text = "Demo Mode : False";
                demo_text.color = Color.red;
            }

            if (Game_Stats.Debug)
            {
                debug_text.text = "Debug Text : True";
                debug_text.color = Color.green;
            }
            else
            {
                debug_text.text = "Debug Text : False";
                debug_text.color = Color.red;
            }

            if (Game_Stats.Draw_skeleton)
            {
                draw_skeleton.text = "Draw Skeleton : True";
                draw_skeleton.color = Color.green;
            }
            else
            {
                draw_skeleton.text = "Draw Skeleton : False";
                draw_skeleton.color = Color.red;
            }

            pre_z.text = Mathf.RoundToInt(Game_Stats.Pre_z).ToString();

            height.text = Mathf.RoundToInt(Game_Stats.Height).ToString() + " cm";
        }

        void Update()
        {

        }

        public void OnBackButtonClick()
        {
            LoadScene("HoloLensWithOpenCVForUnityExample");
        }

        public void On_Demo_Click()
        {
            if (Game_Stats.DEMO)
            {
                Game_Stats.DEMO = false;
                demo_text.text = "Demo Mode : False";
                demo_text.color = Color.red;
            }
            else
            {
                Game_Stats.DEMO = true;
                demo_text.text = "Demo Mode : True";
                demo_text.color = Color.green;
            }
        }

        public void On_Debug_Click()
        {
            if (Game_Stats.Debug)
            {
                Game_Stats.Debug = false;
                debug_text.text = "Debug Text : False";
                debug_text.color = Color.red;
            }
            else
            {
                Game_Stats.Debug = true;
                debug_text.text = "Debug Text : True";
                debug_text.color = Color.green;
            }
        }

        public void On_Skeleton_Click()
        {
            if (Game_Stats.Draw_skeleton)
            {
                Game_Stats.Draw_skeleton = false;
                draw_skeleton.text = "Draw Skeleton : False";
                draw_skeleton.color = Color.red;
            }
            else
            {
                Game_Stats.Draw_skeleton = true;
                draw_skeleton.text = "Draw Skeleton : True";
                draw_skeleton.color = Color.green;
            }
        }

        public void On_Plus1_Click()
        {
            Game_Stats.Pre_z += 1f;
            pre_z.text = Mathf.RoundToInt(Game_Stats.Pre_z).ToString();
        }

        public void On_Plus5_Click()
        {
            Game_Stats.Pre_z += 5f;
            pre_z.text = Mathf.RoundToInt(Game_Stats.Pre_z).ToString();
        }

        public void On_Minus1_Click()
        {
            Game_Stats.Pre_z -= 1f;
            pre_z.text = Mathf.RoundToInt(Game_Stats.Pre_z).ToString();
        }

        public void On_Minus5_Click()
        {
            Game_Stats.Pre_z -= 5f;
            pre_z.text = Mathf.RoundToInt(Game_Stats.Pre_z).ToString();
        }


        public void On_150_Click()
        {
            Game_Stats.Height = 150f;
            height.text = Mathf.RoundToInt(Game_Stats.Height).ToString() + " cm";
        }

        public void On_155_Click()
        {
            Game_Stats.Height = 155f;
            height.text = Mathf.RoundToInt(Game_Stats.Height).ToString() + " cm";
        }

        public void On_160_Click()
        {
            Game_Stats.Height = 160f;
            height.text = Mathf.RoundToInt(Game_Stats.Height).ToString() + " cm";
        }

        public void On_165_Click()
        {
            Game_Stats.Height = 165f;
            height.text = Mathf.RoundToInt(Game_Stats.Height).ToString() + " cm";
        }

        public void On_170_Click()
        {
            Game_Stats.Height = 170f;
            height.text = Mathf.RoundToInt(Game_Stats.Height).ToString() + " cm";
        }

        public void On_175_Click()
        {
            Game_Stats.Height = 175f;
            height.text = Mathf.RoundToInt(Game_Stats.Height).ToString() + " cm";
        }

        public void On_180_Click()
        {
            Game_Stats.Height = 180f;
            height.text = Mathf.RoundToInt(Game_Stats.Height).ToString() + " cm";
        }

        public void On_185_Click()
        {
            Game_Stats.Height = 185f;
            height.text = Mathf.RoundToInt(Game_Stats.Height).ToString() + " cm";
        }
    }
}