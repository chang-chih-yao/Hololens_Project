using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DG.Tweening;

public class Lightning : Skill
{
    public GameObject explosion;
    // Start is called before the first frame update
    void Start()
    {
        explosion.SetActive(false);
    }
    public override void ready(Vector3 position, Vector3 forward, float duration , float size  = 1.0f){

        alive = true;
        setPosition(position);
        this.transform.localScale = Vector3.one * size;
        Invoke("Explosion",0.3f);
    }
    private void Explosion()
    {
        explosion.SetActive(true);
    }

    public override void finish(float duration){
        alive = false;
        DOTween.KillAll();

        Destroy(this.gameObject,duration);
    }
}

