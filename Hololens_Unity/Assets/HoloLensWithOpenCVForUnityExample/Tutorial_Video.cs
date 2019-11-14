using UnityEngine;
using UnityEngine.UI;
using System.Collections;

namespace HoloLensWithOpenCVForUnityExample
{
    public class Tutorial_Video : ExampleSceneBase
    {
        // Use this for initialization
        protected override void Start()
        {
            base.Start();
        }

        // Update is called once per frame
        void Update()
        {

        }
        public void OnBackButtonClick()
        {
            LoadScene("HoloLensWithOpenCVForUnityExample");
        }
    }
}
