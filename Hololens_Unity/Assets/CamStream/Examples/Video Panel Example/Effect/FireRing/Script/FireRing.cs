using UnityEngine;
using DG.Tweening;

public class FireRing : Skill
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
        transform.rotation = Quaternion.LookRotation(forward, Vector3.up);
        circleRadius = 2 * m_size;
        var shape2 = particle2.shape;
        var shape4 = particle4.shape;
        var shape5 = particle5.shape;

        DOTween.To(()=> circleRadius, x=> circleRadius = x, 4.0f * m_size, duration);
        DOTween.To(()=> shape2.radius, x=> shape2.radius = x, 1.5f * m_size, duration);
        DOTween.To(()=> shape4.radius, x=> shape4.radius = x, 1.66f * m_size, duration);
        DOTween.To(()=> shape5.radius, x=> shape5.radius = x, 1.66f * m_size, duration);
    }
    void Update(){
        var main = particle1.main;
        main.startSize = circleRadius;
    }

    public override void finish(float duration){
        alive = false;
        var shape2 = particle2.shape;
        var shape4 = particle4.shape;
        var shape5 = particle5.shape;

        DOTween.To(()=> circleRadius, x=> circleRadius = x, 1.0f * m_size, duration);
        DOTween.To(()=> shape2.radius, x=> shape2.radius = x, 0.2f * m_size, duration);
        DOTween.To(()=> shape4.radius, x=> shape4.radius = x, 0.85f * m_size, duration);
        DOTween.To(()=> shape5.radius, x=> shape5.radius = x, 0.85f * m_size, duration);
        Destroy(this.gameObject, duration);
    }
}
