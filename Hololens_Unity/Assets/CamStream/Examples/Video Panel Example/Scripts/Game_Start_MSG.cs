using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Game_Start_MSG : MonoBehaviour
{
    public static bool playerID_init;
    public Text Game_Start_Text;
    private float animation_time;

    private float rotate_time = 0.0f;
    public GameObject loading_png;
    public Text loading_text;
    void Start()
    {
        playerID_init = false;
        Game_Start_Text.enabled = false;
        loading_text.enabled = true;
        loading_png.SetActive(true);
    }

    // Update is called once per frame
    void Update()
    {
        if (VideoPanelApp.connect == false)
        {
            if (Time.time - rotate_time >= 0.05f)
            {
                loading_png.transform.Rotate(new Vector3(0, 0, -15), Space.Self);
                rotate_time = Time.time;
            }
        }
        if (VideoPanelApp.connect == true && playerID_init == false)
        {
            loading_text.enabled = false;
            loading_png.SetActive(false);
            if (Game_Stats.PlayerID == "holo_P0")
            {
                if (VideoPanelApp.player_id_err_p0 == 1)   // 代表 server 端已經偵測到playerID衝突，自動把目前的playerID設為另一個了
                {
                    Game_Stats.PlayerID = "holo_P1";
                    Game_Start_Text.enabled = true;
                    Game_Start_Text.text = "Wrong player ID selected, the system has automatically selected 'P2'";
                    playerID_init = true;
                    animation_time = Time.time;
                }
                else if (VideoPanelApp.player_id_err_p0 == 0)
                {
                    Game_Start_Text.enabled = true;
                    Game_Start_Text.text = "Game Start !";
                    playerID_init = true;
                    animation_time = Time.time;
                }
            }
            else if (Game_Stats.PlayerID == "holo_P1")
            {
                if (VideoPanelApp.player_id_err_p1 == 1)
                {
                    Game_Stats.PlayerID = "holo_P0";
                    Game_Start_Text.enabled = true;
                    Game_Start_Text.text = "Wrong player ID selected, the system has automatically selected 'P1'";
                    playerID_init = true;
                    animation_time = Time.time;
                }
                else if (VideoPanelApp.player_id_err_p1 == 0)
                {
                    Game_Start_Text.enabled = true;
                    Game_Start_Text.text = "Game Start !";
                    playerID_init = true;
                    animation_time = Time.time;
                }
            }
        }
        if (playerID_init == true && Game_Start_Text.enabled == true)
        {
            if (Time.time - animation_time >= 2f)
                Game_Start_Text.enabled = false;
        }
    }
}
