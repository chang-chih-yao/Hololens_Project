using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class My_Line : MonoBehaviour
{
    public Material line_mat;
    public static bool Draw_skeleton = true;

    public GameObject canvas_openpose;
    public GameObject Line_1;
    public GameObject Line_2;
    public GameObject Line_3;
    public GameObject Line_4;
    public GameObject Line_5;
    private static LineRenderer lineRenderer_1;
    private static LineRenderer lineRenderer_2;
    private static LineRenderer lineRenderer_3;
    private static LineRenderer lineRenderer_4;
    private static LineRenderer lineRenderer_5;
    public Image[] Joints;

    public Color skeleton_color = Color.cyan;
    public float skeleton_width = 0.15f;
    // Start is called before the first frame update
    void Start()
    {
        lineRenderer_1 = Line_1.AddComponent<LineRenderer>();
        lineRenderer_1.material = line_mat;
        lineRenderer_1.SetColors(skeleton_color, skeleton_color);
        lineRenderer_1.SetWidth(skeleton_width, skeleton_width);
        lineRenderer_1.positionCount = 2;

        lineRenderer_2 = Line_2.AddComponent<LineRenderer>();
        lineRenderer_2.material = line_mat;
        lineRenderer_2.SetColors(skeleton_color, skeleton_color);
        lineRenderer_2.SetWidth(skeleton_width, skeleton_width);
        lineRenderer_2.positionCount = 4;

        lineRenderer_3 = Line_3.AddComponent<LineRenderer>();
        lineRenderer_3.material = line_mat;
        lineRenderer_3.SetColors(skeleton_color, skeleton_color);
        lineRenderer_3.SetWidth(skeleton_width, skeleton_width);
        lineRenderer_3.positionCount = 4;

        lineRenderer_4 = Line_4.AddComponent<LineRenderer>();
        lineRenderer_4.material = line_mat;
        lineRenderer_4.SetColors(skeleton_color, skeleton_color);
        lineRenderer_4.SetWidth(skeleton_width, skeleton_width);
        lineRenderer_4.positionCount = 4;

        lineRenderer_5 = Line_5.AddComponent<LineRenderer>();
        lineRenderer_5.material = line_mat;
        lineRenderer_5.SetColors(skeleton_color, skeleton_color);
        lineRenderer_5.SetWidth(skeleton_width, skeleton_width);
        lineRenderer_5.positionCount = 4;
    }

    void LateUpdate()
    {
        if (Draw_skeleton == true)
            canvas_openpose.SetActive(true);
        else
            canvas_openpose.SetActive(false);

        if (Joints[0].enabled && Joints[1].enabled)
        {
            lineRenderer_1.positionCount = 2;
            lineRenderer_1.SetPosition(0, Joints[0].gameObject.transform.position);
            lineRenderer_1.SetPosition(1, Joints[1].gameObject.transform.position);
        }

        if (Joints[1].enabled && Joints[2].enabled)
        {
            if (Joints[3].enabled)
            {
                if (Joints[4].enabled)
                {
                    lineRenderer_2.positionCount = 4;
                    lineRenderer_2.SetPosition(0, Joints[1].gameObject.transform.position);
                    lineRenderer_2.SetPosition(1, Joints[2].gameObject.transform.position);
                    lineRenderer_2.SetPosition(2, Joints[3].gameObject.transform.position);
                    lineRenderer_2.SetPosition(3, Joints[4].gameObject.transform.position);
                }
                else
                {
                    lineRenderer_2.positionCount = 3;
                    lineRenderer_2.SetPosition(0, Joints[1].gameObject.transform.position);
                    lineRenderer_2.SetPosition(1, Joints[2].gameObject.transform.position);
                    lineRenderer_2.SetPosition(2, Joints[3].gameObject.transform.position);
                }
            }
            else
            {
                lineRenderer_2.positionCount = 2;
                lineRenderer_2.SetPosition(0, Joints[1].gameObject.transform.position);
                lineRenderer_2.SetPosition(1, Joints[2].gameObject.transform.position);
            }
        }

        if (Joints[1].enabled && Joints[5].enabled)
        {
            if (Joints[6].enabled)
            {
                if (Joints[7].enabled)
                {
                    lineRenderer_3.positionCount = 4;
                    lineRenderer_3.SetPosition(0, Joints[1].gameObject.transform.position);
                    lineRenderer_3.SetPosition(1, Joints[5].gameObject.transform.position);
                    lineRenderer_3.SetPosition(2, Joints[6].gameObject.transform.position);
                    lineRenderer_3.SetPosition(3, Joints[7].gameObject.transform.position);
                }
                else
                {
                    lineRenderer_3.positionCount = 3;
                    lineRenderer_3.SetPosition(0, Joints[1].gameObject.transform.position);
                    lineRenderer_3.SetPosition(1, Joints[5].gameObject.transform.position);
                    lineRenderer_3.SetPosition(2, Joints[6].gameObject.transform.position);
                }
            }
            else
            {
                lineRenderer_3.positionCount = 2;
                lineRenderer_3.SetPosition(0, Joints[1].gameObject.transform.position);
                lineRenderer_3.SetPosition(1, Joints[5].gameObject.transform.position);
            }
        }

        if (Joints[1].enabled && Joints[8].enabled)
        {
            if (Joints[9].enabled)
            {
                if (Joints[10].enabled)
                {
                    lineRenderer_4.positionCount = 4;
                    lineRenderer_4.SetPosition(0, Joints[1].gameObject.transform.position);
                    lineRenderer_4.SetPosition(1, Joints[8].gameObject.transform.position);
                    lineRenderer_4.SetPosition(2, Joints[9].gameObject.transform.position);
                    lineRenderer_4.SetPosition(3, Joints[10].gameObject.transform.position);
                }
                else
                {
                    lineRenderer_4.positionCount = 3;
                    lineRenderer_4.SetPosition(0, Joints[1].gameObject.transform.position);
                    lineRenderer_4.SetPosition(1, Joints[8].gameObject.transform.position);
                    lineRenderer_4.SetPosition(2, Joints[9].gameObject.transform.position);
                }
            }
            else
            {
                lineRenderer_4.positionCount = 2;
                lineRenderer_4.SetPosition(0, Joints[1].gameObject.transform.position);
                lineRenderer_4.SetPosition(1, Joints[8].gameObject.transform.position);
            }
        }

        if (Joints[1].enabled && Joints[11].enabled)
        {
            if (Joints[12].enabled)
            {
                if (Joints[13].enabled)
                {
                    lineRenderer_5.positionCount = 4;
                    lineRenderer_5.SetPosition(0, Joints[1].gameObject.transform.position);
                    lineRenderer_5.SetPosition(1, Joints[11].gameObject.transform.position);
                    lineRenderer_5.SetPosition(2, Joints[12].gameObject.transform.position);
                    lineRenderer_5.SetPosition(3, Joints[13].gameObject.transform.position);
                }
                else
                {
                    lineRenderer_5.positionCount = 3;
                    lineRenderer_5.SetPosition(0, Joints[1].gameObject.transform.position);
                    lineRenderer_5.SetPosition(1, Joints[11].gameObject.transform.position);
                    lineRenderer_5.SetPosition(2, Joints[12].gameObject.transform.position);
                }
            }
            else
            {
                lineRenderer_5.positionCount = 2;
                lineRenderer_5.SetPosition(0, Joints[1].gameObject.transform.position);
                lineRenderer_5.SetPosition(1, Joints[11].gameObject.transform.position);
            }
        }

        if (Joints[1].enabled == false)
        {
            lineRenderer_1.positionCount = 0;
            lineRenderer_2.positionCount = 0;
            lineRenderer_3.positionCount = 0;
            lineRenderer_4.positionCount = 0;
            lineRenderer_5.positionCount = 0;
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public static LineRenderer Get_line_1()
    {
        return lineRenderer_1;
    }

    public static LineRenderer Get_line_2()
    {
        return lineRenderer_2;
    }

    public static LineRenderer Get_line_3()
    {
        return lineRenderer_3;
    }

    public static LineRenderer Get_line_4()
    {
        return lineRenderer_4;
    }

    public static LineRenderer Get_line_5()
    {
        return lineRenderer_5;
    }
}
