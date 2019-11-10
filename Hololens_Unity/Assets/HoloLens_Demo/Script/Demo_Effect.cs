using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using System.Text;
using System.IO;
using System;

using System.Collections;

public class Demo_Effect : MonoBehaviour
{
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

    private Boolean SK2_animation = false;
    private Boolean animation_flag = false;
    private Boolean SK2_fail_flag = false;
    private Boolean my_SK2_animation = false;
    private Boolean my_SK2_fail_flag = false;
    private Boolean blood_animation = false;
    private Boolean defense_animation = false;
    private Boolean failed_attack_animation = false;
    private Boolean end_game_flag = false;

    private float defense_time;
    private float failed_attack_time;

    private float P0_HP_X;
    private float P1_HP_X;
    private float SK2_reduce_x;
    private float SK2_reduce_y;
    private float SK2_reduce_z;
    private float SK2_limit;

    private void Start()
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
        //player = Game_Stats.PlayerID;
        player = "holo_P0";
    }

    private Vector3 joint_coordinate_trans(float x, float y)
    {
        return new Vector3(x / 10.0f - 22.4f, -(y / 10.0f - 12.6f), 0f);
    }

    private IEnumerator Lose_Game()
    {
        end_game_flag = true;

        lose_img.SetActive(true);
        yield return new WaitForSeconds(3);
        lose_img.SetActive(false);
        start_new_game.SetActive(true);
        yield return new WaitForSeconds(3);
        start_new_game.SetActive(false);

        P0_HP.GetComponent<RectTransform>().sizeDelta = new Vector2(10, P0_HP.GetComponent<RectTransform>().sizeDelta.y);
        P1_HP.GetComponent<RectTransform>().sizeDelta = new Vector2(10, P1_HP.GetComponent<RectTransform>().sizeDelta.y);
        P0_HP.GetComponent<RectTransform>().localPosition = new Vector3(P0_HP_X, P0_HP.GetComponent<RectTransform>().localPosition.y,
                    P0_HP.GetComponent<RectTransform>().localPosition.z);
        P1_HP.GetComponent<RectTransform>().localPosition = new Vector3(P1_HP_X, P1_HP.GetComponent<RectTransform>().localPosition.y,
                    P1_HP.GetComponent<RectTransform>().localPosition.z);
        cur_gamepoint_p0 = 10f;
        cur_gamepoint_p1 = 10f;
        p0_win_lose = 0;
        p1_win_lose = 0;

        end_game_flag = false;
    }

    private IEnumerator Win_Game()
    {
        end_game_flag = true;

        win_img.SetActive(true);
        yield return new WaitForSeconds(3);
        win_img.SetActive(false);
        start_new_game.SetActive(true);
        yield return new WaitForSeconds(3);
        start_new_game.SetActive(false);

        P0_HP.GetComponent<RectTransform>().sizeDelta = new Vector2(10, P0_HP.GetComponent<RectTransform>().sizeDelta.y);
        P1_HP.GetComponent<RectTransform>().sizeDelta = new Vector2(10, P1_HP.GetComponent<RectTransform>().sizeDelta.y);
        P0_HP.GetComponent<RectTransform>().localPosition = new Vector3(P0_HP_X, P0_HP.GetComponent<RectTransform>().localPosition.y,
                    P0_HP.GetComponent<RectTransform>().localPosition.z);
        P1_HP.GetComponent<RectTransform>().localPosition = new Vector3(P1_HP_X, P1_HP.GetComponent<RectTransform>().localPosition.y,
                    P1_HP.GetComponent<RectTransform>().localPosition.z);
        cur_gamepoint_p0 = 10f;
        cur_gamepoint_p1 = 10f;
        p0_win_lose = 0;
        p1_win_lose = 0;

        end_game_flag = false;
    }

    private void Update()
    {
        if (ClientThread.receive_joint != null)
        {
            Debug.Log(ClientThread.receive_joint);
            string[] arr = ClientThread.receive_joint.Split(',');

            for (int i = 0; i < Joints.Length * 2; i++)
            {
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

            if (player == "holo_P0")
            {
                action = holo_action_p1;
                my_action = holo_action_p0;
            }
            else if (player == "holo_P1")
            {
                action = holo_action_p0;
                my_action = holo_action_p1;
            }




            debug_4.text = p0_win_lose.ToString();
            debug_1.text = action.ToString() + ", " + holo_action_p0.ToString() + "|" + holo_action_p1.ToString() + ", " + gamepoint_p0.ToString() + "|" + gamepoint_p1.ToString();
            /*debug_2.text = ((int)fps).ToString();
            if ((int)fps < 30)
            {
                debug_2.color = Color.red;
            }
            else
            {
                debug_2.color = Color.green;
            }*/

            debug_3.text = recv_frame.ToString();
            if (recv_frame == recv_frame_old)
            {
                if (frame_cou > 6)
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
            {
                debug_1.text = "No human";
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


            if (animation_flag == true)
            {
                if (SK2_animation == true)
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
                        animation_flag = false;
                        SK2_animation = false;
                        Skill_2.GetComponent<Transform>().localPosition = new Vector3(0f, 0f, -1);
                        Skill_2.SetActive(false);
                    }
                }
                if (blood_animation)
                {
                    float blood_alpha = blood.color.a;
                    blood_alpha = blood_alpha - 0.02f;
                    blood.color = new Color(blood.color.r, blood.color.g, blood.color.b, blood_alpha);
                    if (blood_alpha <= 0f)
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
                    SK2_z = SK2_z + 1.5f;   // z 從 -85f 到 0f
                    my_Skill_2.GetComponent<Transform>().localPosition = new Vector3(2f, 0f, SK2_z);

                    if (SK2_z >= 0f)
                    {
                        animation_flag = false;
                        my_SK2_animation = false;
                        my_Skill_2.GetComponent<Transform>().localPosition = new Vector3(2f, 0f, -43f);
                        my_Skill_2.SetActive(false);
                    }
                }
            }
            else
            {
                if (action == 2)
                {
                    Skill_2.SetActive(true);
                    SK2_fail_flag = false;
                    if (Joints_co[2] != 0 || Joints_co[3] != 0)         // 有人，(Neck_x,Neck_y)=(0,0)我們訂為沒偵測到人
                    {
                        if (Joints_co[8] != 0f || Joints_co[9] != 0f)   // 右手有偵測到才去改Skill 2的動畫位置，(RWrist_x,RWrist_y)=(0,0)代表沒偵測到
                        {
                            Skill_2.GetComponent<Transform>().localPosition = joint_coordinate_trans(Joints_co[8], Joints_co[9]);  //RWrist
                        }
                    }

                    /*Skill_1.GetComponent<RectTransform>().localPosition = new Vector3(-40f, 30f, 0f);
                    Skill_3.GetComponent<RectTransform>().localPosition = new Vector3(-40f, 30f, 0f);
                    Skill_4.GetComponent<RectTransform>().localPosition = new Vector3(-40f, 30f, 0f);
                    Skill_5.GetComponent<RectTransform>().localPosition = new Vector3(-40f, 30f, 0f);
                    Skill_6.GetComponent<RectTransform>().localPosition = new Vector3(-40f, 30f, 0f);*/
                }
                else if (action == 3)
                {
                    if (Skill_2.activeSelf == true)
                    {
                        //Skill_2.GetComponent<Transform>().localPosition = joint_coordinate_trans(Joints_co[8], Joints_co[9]);  //RWrist
                        animation_flag = true;
                        SK2_animation = true;

                        SK2_limit = -85f;
                        SK2_reduce_z = 1.5f;
                        SK2_reduce_x = SK2_reduce_z / -SK2_limit * Skill_2.GetComponent<Transform>().localPosition.x;
                        SK2_reduce_y = SK2_reduce_z / -SK2_limit * Skill_2.GetComponent<Transform>().localPosition.y;
                    }
                }
                else
                {
                    if (SK2_fail_flag)                    // 如果是true，代表已經連續兩個frame的action都不是action2，但上上個frame是action2
                    {
                        Skill_2.SetActive(false);         // 此時我們就判定 現在動作不是action2
                        SK2_fail_flag = false;
                    }
                    else                                  // 代表此時不是action2
                    {
                        if (Skill_2.activeSelf == true)   // 代表此時不是action2，但上一個frame是action2，代表目前只有錯一個frame而已
                            SK2_fail_flag = true;
                    }
                }

                if (my_action == 2)
                {
                    my_Skill_2.SetActive(true);
                    my_SK2_fail_flag = false;
                    my_Skill_2.GetComponent<Transform>().localPosition = new Vector3(100f, 0f, 0f);
                }
                else if (my_action == 3)
                {
                    if (my_Skill_2.activeSelf == true)
                    {
                        //Skill_2.GetComponent<Transform>().localPosition = joint_coordinate_trans(Joints_co[8], Joints_co[9]);  //RWrist
                        animation_flag = true;
                        my_SK2_animation = true;
                        my_Skill_2.GetComponent<Transform>().localPosition = new Vector3(2f, 0f, -85f);
                    }
                }
                else
                {
                    if (my_SK2_fail_flag)                    // 如果是true，代表已經連續兩個frame的action都不是action2，但上上個frame是action2
                    {
                        my_Skill_2.SetActive(false);         // 此時我們就判定 現在動作不是action2
                        my_SK2_fail_flag = false;
                    }
                    else                                  // 代表此時不是action2
                    {
                        if (my_Skill_2.activeSelf == true)   // 代表此時不是action2，但上一個frame是action2，代表目前只有錯一個frame而已
                            my_SK2_fail_flag = true;
                    }
                }
            }

            if (player == "holo_P0")
            {
                if (p0_win_lose == 2 && end_game_flag == false)      // lose
                    StartCoroutine(Lose_Game());
                else if (p0_win_lose == 1 && end_game_flag == false)  // win
                    StartCoroutine(Win_Game());

                if (defense_skill_2_p0 == 1 && defense_animation == false)
                {
                    animation_flag = true;
                    defense_animation = true;
                    successful_defense.SetActive(true);
                    defense_time = Time.time;
                }
                else if (defense_skill_2_p1 == 1 && defense_animation == false)
                {
                    animation_flag = true;
                    failed_attack_animation = true;
                    failed_attack.SetActive(true);
                    failed_attack_time = Time.time;
                }

                if (blood_effect_p0 == 1 && blood_animation == false)
                {
                    animation_flag = true;
                    blood_animation = true;
                    blood.color = new Color(blood.color.r, blood.color.g, blood.color.b, 1f);
                }
            }
            else if (player == "holo_P1")
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
}
