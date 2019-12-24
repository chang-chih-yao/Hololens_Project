using UnityEngine;
using DG.Tweening;

public class Hit : Skill
{
    public ParticleSystem particle1;
    public ParticleSystem particle2;
    public ParticleSystem particle3;
    float m_size;
    public override void ready(Vector3 position, Vector3 forward, float duration, float size  = 1.0f){
        alive = true;
        m_size = size;
        setPosition(position);
        transform.rotation = Quaternion.LookRotation(forward, Vector3.up);
        particle1.Stop();
        particle2.Stop();
        particle3.Stop();
        Color targetColor = Color.white;
        if(duration == 0){
            targetColor = Color.red;
        }else if(duration == 1){
            targetColor = new Color(1, 0.153f, 0, 1);
        }else if(duration == 2){
            targetColor = Color.green;
        }else if(duration == 3){
            targetColor = new Color(0, 0.5f, 1, 1);
        }
        var main1 = particle1.main;
        main1.startColor = targetColor;
        var main2 = particle2.main;
        main2.startColor = targetColor;
        
        this.gameObject.transform.localScale = Vector3.one * m_size;
        var shape2 = particle2.shape;
        shape2.radius = 0.31f * m_size;
        var main3 = particle3.main;
        main3.startSize = 6 * m_size;


        particle1.Play();
        particle2.Play();
        particle3.Play();

        finish(2);
    }   

    public override void finish(float duration){
        alive = false;
        Destroy(this.gameObject, duration);
    }
}
