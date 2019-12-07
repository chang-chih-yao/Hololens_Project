using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Orb : Skill
{

    public List<GameObject> initParticle;
    public List<GameObject> stableParticle;

    // Start is called before the first frame update
    public override void ready(Vector3 position, Vector3 forward, float duration){
        foreach(GameObject g in initParticle)
        {
            g.SetActive(true);
        }
                foreach(GameObject g in stableParticle)
        {
            g.SetActive(false);
        }
        Invoke("DisplayStableParticle",1.0f);
    }


    public override void finish(float duration){
        Destroy(this.gameObject);
    }

    private void DisplayStableParticle()
    {
        foreach(GameObject g in stableParticle)
        {
            g.SetActive(true);
        }
        foreach(GameObject g in initParticle)
        {
            g.SetActive(false);
        }
    }
}
