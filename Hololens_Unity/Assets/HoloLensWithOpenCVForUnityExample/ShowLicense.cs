using UnityEngine;
using System.Collections;

namespace HoloLensWithOpenCVForUnityExample
{
    public class ShowLicense : ExampleSceneBase
    {
        string s = "";
        // Use this for initialization
        protected override void Start ()
        {
            base.Start ();
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
    }
}
