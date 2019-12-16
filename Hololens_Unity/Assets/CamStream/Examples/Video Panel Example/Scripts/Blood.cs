using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Blood : MonoBehaviour
{
    public SpriteRenderer my_blood;
    public SpriteRenderer blood;

    public static bool my_blood_animation = false;
    public static bool blood_animation = false;
    private bool my_init = false;
    private bool init = false;
    void Start()
    {
        my_blood_animation = false;
        my_init = false;
        blood_animation = false;
        init = false;
    }
    
    void Update()
    {
        if (my_blood_animation)
        {
            if (!my_init)
            {
                my_blood.color = new Color(my_blood.color.r, my_blood.color.g, my_blood.color.b, 1f);
                my_init = true;
            }
            else
            {
                float my_blood_alpha = my_blood.color.a;
                my_blood_alpha = my_blood_alpha - 0.02f;
                if (my_blood_alpha >= 0f)
                {
                    my_blood.color = new Color(my_blood.color.r, my_blood.color.g, my_blood.color.b, my_blood_alpha);
                }
                else
                {
                    my_blood_animation = false;
                    my_init = false;
                }
            }
        }

        if (blood_animation)
        {
            if (!init)
            {
                blood.color = new Color(blood.color.r, blood.color.g, blood.color.b, 1f);
                init = true;
            }
            else
            {
                float blood_alpha = blood.color.a;
                blood_alpha = blood_alpha - 0.02f;
                if (blood_alpha >= 0f)
                {
                    blood.color = new Color(blood.color.r, blood.color.g, blood.color.b, blood_alpha);
                }
                else
                {
                    blood_animation = false;
                    init = false;
                }
            }
        }
    }
}
