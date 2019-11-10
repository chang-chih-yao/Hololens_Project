using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System;

public class My_animation : MonoBehaviour
{
    private Boolean animation_flag = false;
    private Boolean SK2_animation = false;
    private Boolean SK2_fail_flag = false;
    private Boolean my_SK2_animation = false;
    private Boolean my_SK2_fail_flag = false;
    private Boolean blood_animation = false;
    private Boolean defense_animation = false;
    private Boolean failed_attack_animation = false;

    public GameObject Skill_2;
    public GameObject my_Skill_2;

    void Start()
    {
        
    }

    
    void Update()
    {
        /*
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
                SK2_fail_flag = false;
                if (Joints_co[2] != 0 || Joints_co[3] != 0)         // 有人，(Neck_x,Neck_y)=(0,0)我們訂為沒偵測到人
                {
                    if (Joints_co[8] != 0f || Joints_co[9] != 0f)   // 右手有偵測到才去改Skill 2的動畫位置，(RWrist_x,RWrist_y)=(0,0)代表沒偵測到
                    {
                        Skill_2.GetComponent<Transform>().localPosition = Skill_2_coordinate_trans(Joints_co[8], Joints_co[9], Joints_co[6], Joints_co[7]);  //RWrist
                    }
                }

                //Skill_1.GetComponent<RectTransform>().localPosition = new Vector3(-40f, 30f, 0f);
                //Skill_3.GetComponent<RectTransform>().localPosition = new Vector3(-40f, 30f, 0f);
                //Skill_4.GetComponent<RectTransform>().localPosition = new Vector3(-40f, 30f, 0f);
                //Skill_5.GetComponent<RectTransform>().localPosition = new Vector3(-40f, 30f, 0f);
                //Skill_6.GetComponent<RectTransform>().localPosition = new Vector3(-40f, 30f, 0f)
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
                    my_Skill_2.GetComponent<Transform>().localPosition = new Vector3(2f, 0f, -40f);
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
        }*/
    }
}
