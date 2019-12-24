using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DG.Tweening;

public class Laser : Skill
{

    // Start is called before the first frame update
    public override void ready(Vector3 position, Vector3 forward, float duration , float size  = 1.0f){

        alive = true;
        setPosition(position);
        this.transform.LookAt(this.transform.position+forward);


        this.transform.localScale = new Vector3( size * 1,  size * 1, size * 1);
        
    }

    // public override void setPosition()
    // {
    //     Rigidbody rb = this.GetComponent<Rigidbody>();
    //     rb.velocity = this.transform.forward * 30f;
    // }
    public override void finish(float duration){
        alive = false;
        DOTween.KillAll();

        this.transform.DOScale( new Vector3(0.001f, 0.001f ,0.001f) , duration);

        Destroy(this.gameObject,duration);
    }
}
