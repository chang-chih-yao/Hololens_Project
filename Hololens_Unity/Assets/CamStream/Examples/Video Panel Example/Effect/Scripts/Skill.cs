using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Skill : MonoBehaviour
{
    public virtual void ready(Vector3 position, Vector3 forward, float duration){

    }

    public virtual void setPosition(Vector3 position){
        transform.position = position;
    }

    public virtual void finish(float duration){
        
    }
}
