using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DG.Tweening;

public class MagicCircle : Skill
{

    public override void ready(Vector3 position, Vector3 forward, float duration)
    {
        setPosition(position);
        this.transform.DOScale( new Vector3(1,1,1),duration);
        //DOTween.To(()=> this.transform.localScale, x=> this.transform.localScale = x, new Vector3(1,1,1), 1.5f);
    }

    public override void finish(float duration)
    {
        this.transform.DOScale( new Vector3(0.01f,0.01f,1),duration);
        //DOTween.To(()=> this.transform.localScale, x=> this.transform.localScale = x, new Vector3(0,0,1), 1.5f);

        Destroy(this.gameObject,duration);
    }


}
