using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DG.Tweening;

public class MagicCircle : Skill
{
    public GameObject bigCircle;
    public GameObject midCircle;
    public GameObject smallCircle;

    public GameObject bigSpark;
    public GameObject midSpark;
    public ParticleSystem blockSpark;
    public MeshRenderer[] circleMaterial; 
    Sequence s0;

    void Update()
    {
        bigSpark.transform.RotateAround(this.transform.position, this.transform.forward, 108 * Time.deltaTime);
        midSpark.transform.RotateAround(this.transform.position + 0.2f *this.transform.forward, this.transform.forward, 108 * Time.deltaTime);
        bigCircle.transform.localEulerAngles += new Vector3( 0 , 0 ,  72 * Time.deltaTime);
        midCircle.transform.localEulerAngles += new Vector3( 0 , 0 ,  -108 * Time.deltaTime);
        //smallCircle.transform.localEulerAngles += new Vector3( 0 , 0 ,  -180 * Time.deltaTime);

        //smallSpark.transform.RotateAround(this.transform.position - 0.4f * Vector3.forward, Vector3.forward, -72 * Time.deltaTime);

    }
    public override void ready(Vector3 position, Vector3 forward, float duration, float size = 1.0f)
    {
        alive = true;
        setPosition(position);
        this.transform.LookAt(this.transform.position+forward);
        this.transform.localScale = Vector3.one * 0.01f;
        this.transform.DOScale( new Vector3(1 * size , 1* size , 1) , duration).SetEase(Ease.OutCubic);

        circleMaterial = GetComponentsInChildren<MeshRenderer>();

    }
    public void blockSuccess()
    {   
        Vector3 tempScale = transform.localScale;
        if(s0 == null || !s0.IsPlaying())
        {
            Debug.Log("blocked");
            s0 = DOTween.Sequence();
            s0.Append(transform.DOScale(tempScale * 1.5f, 0.25f));
            s0.Append(transform.DOScale(tempScale, 0.25f));
            
            foreach(MeshRenderer r in circleMaterial)
            {
                if(!DOTween.IsTweening(s0))
                {
                    //r.material.DOColor(Color.white,"_Color",0.5f);
                    //Sequence s0 = DOTween.Sequence();
                    
                    Sequence s1 = DOTween.Sequence();
                    Color tempColor = r.material.color;
                    s1.Append(r.material.DOColor(new Color(0.5f,0.5f,0.5f,1),"_Color",0.25f));
                    s1.Append(r.material.DOColor(tempColor,"_Color",0.25f));
                }
            }
            Sequence mySequence = DOTween.Sequence();
        }

    }
    public override void finish(float duration)
    {
        alive = false;
        DOTween.KillAll();
        this.transform.DOScale( new Vector3(0.01f,0.01f,1),duration).SetEase(Ease.OutCubic);;
        //DOTween.To(()=> this.transform.localScale, x=> this.transform.localScale = x, new Vector3(0,0,1), 1.5f);
        Destroy(this.gameObject,duration);
    }


}
