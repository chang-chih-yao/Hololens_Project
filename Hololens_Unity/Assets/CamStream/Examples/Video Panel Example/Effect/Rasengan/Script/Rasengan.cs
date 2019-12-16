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
    public ParticleSystem particle5;
    float circleRadius;
    float m_size;

    public override void ready(Vector3 position, Vector3 forward, float duration, float size  = 1.0f){
        alive = true;
        m_size = size;
        setPosition(position);
        var shape = particle1.shape;
        var shape2 = particle4.shape;
        circleRadius = 1 * m_size;
        particle5.Stop();
        var main5 = particle5.main;
        main5.duration = duration * 0.8f;
        var shape5 = particle5.shape;
        shape5.radius = 4.02f * m_size;
        particle5.Play();

        shape.radius *= m_size;
        shape2.radius *= m_size;
        particle3.gameObject.transform.localScale = Vector3.one * m_size;

        DOTween.To(()=> shape.radius, x=> shape.radius = x, 0.33f * m_size, duration);
        DOTween.To(()=> shape2.radius, x=> shape2.radius = x, 0.6f * m_size, duration);
        DOTween.To(()=> circleRadius, x=> circleRadius = x, 1.26f * m_size, duration);
        particle3.gameObject.transform.DOScale(Vector3.one * 1.453914f * m_size, duration);
    }



    void Update(){
        var main2 = particle2.main;
        main2.startSize = circleRadius;
    }

    public override void finish(float duration){
        DOTween.Kill(particle1.gameObject);
        DOTween.Kill(particle2.gameObject);
        DOTween.Kill(particle3.gameObject);
        DOTween.Kill(particle4.gameObject);
        alive = false;
        particle1.Stop();
        particle4.Stop();
        DOTween.To(()=> circleRadius, x=> circleRadius = x, 0.5f * m_size, duration);
        particle3.gameObject.transform.DOScale(Vector3.one * 0.01f * m_size, duration);
        Destroy(this.gameObject, duration);
    }
}
