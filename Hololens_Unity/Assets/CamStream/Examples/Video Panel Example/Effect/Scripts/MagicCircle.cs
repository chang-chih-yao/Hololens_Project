using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DG.Tweening;

public class MagicCircle : Skill
{

    public override void ready(Vector3 position, Vector3 forward, float duration, float size)
    {
        alive = true;
        setPosition(position);
        this.transform.LookAt(this.transform.position+forward);
        this.transform.localScale = Vector3.one * 0.01f;
        this.transform.DOScale( new Vector3(size, size, size), duration);
        //DOTween.To(()=> this.transform.localScale, x=> this.transform.localScale = x, new Vector3(1,1,1), 1.5f);
    }
    public void setScale(Vector3 s)
    {
        DOTween.KillAll();
        this.transform.localScale = s;
    }

    public override void finish(float duration)
    {
        alive = false;
        this.transform.DOScale( new Vector3(0.01f,0.01f,1),duration);
        //DOTween.To(()=> this.transform.localScale, x=> this.transform.localScale = x, new Vector3(0,0,1), 1.5f);

        Destroy(this.gameObject,duration);
    }


}
