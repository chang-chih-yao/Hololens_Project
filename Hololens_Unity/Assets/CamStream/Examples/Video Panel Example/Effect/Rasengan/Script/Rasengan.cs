using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DG.Tweening;

public class Rasengan : Skill
{
    public ParticleSystem particle1;
    public ParticleSystem particle2;
    public ParticleSystem particle3;
    public ParticleSystem particle4;
    float circleRadius;
    public override void ready(Vector3 position, Vector3 forward, float duration){
        setPosition(position);
        var shape = particle1.shape;
        var shape2 = particle4.shape;
        circleRadius = 1;
        DOTween.To(()=> shape.radius, x=> shape.radius = x, 0.55f, duration);
        DOTween.To(()=> shape2.radius, x=> shape2.radius = x, 90f, duration);
        DOTween.To(()=> circleRadius, x=> circleRadius = x, 3.15f, duration);
        particle3.gameObject.transform.DOScale(Vector3.one * 3.15f, duration);
    }

    void Update(){
        var main2 = particle2.main;
        main2.startSize = circleRadius * 1.5f;
    }

    public override void finish(float duration){
        // var shape = particle1.shape;
        // DOTween.To(()=> shape.radius, x=> shape.radius = x, 0.1f, duration / 2.0f);
        particle1.Stop();
        particle4.Stop();
        DOTween.To(()=> circleRadius, x=> circleRadius = x, 0.5f, duration);
        particle3.gameObject.transform.DOScale(Vector3.one * 0.01f, duration);
        Destroy(this.gameObject, duration);
    }
}
