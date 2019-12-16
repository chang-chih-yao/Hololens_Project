using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DG.Tweening;
public class Orb : Skill
{

    public GameObject lineLight;
    public GameObject initOrb;

    // Start is called before the first frame update
    public override void ready(Vector3 position, Vector3 forward, float duration , float size  = 1.0f){

        alive = true;
        setPosition(position);
        this.transform.LookAt(this.transform.position+forward);
        this.transform.localScale = Vector3.one * size;
        initOrb.transform.localScale = Vector3.one * 0.001f;
        initOrb.transform.DOScale( Vector3.one , duration);


        
    }


    public override void finish(float duration){
        alive = false;
        DOTween.KillAll();

        this.transform.DOScale( Vector3.one * 0.001f,duration);

        Destroy(this.gameObject,duration);
    }


}
