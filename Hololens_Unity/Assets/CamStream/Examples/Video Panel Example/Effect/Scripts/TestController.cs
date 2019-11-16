using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TestController : MonoBehaviour
{
    [Header("Attack")]
    public GameObject attack_red;
    public GameObject attack_orange;
    public GameObject attack_green;
    public GameObject attack_purple;
    [Header("Defense")]
    public GameObject defense_red;
    public GameObject defense_orange;
    public GameObject defense_green;
    public GameObject defense_purple;


    private GameObject skill_temp;
    private Rasengan rasengan;
    private MagicCircle magicCircle;
    private Orb orb;
    private Laser laser;
    private Explode explode;

    private int currentSkill;
    private enum SkillState
    {
        Ready,
        Stable,
        Finish,
        Idle
    };
    private SkillState currentSkillState = SkillState.Idle;
    bool attack_ready = false;
    bool attack_release = false;
    bool attack_finish = false;

    void Start(){
    }

    void Update(){
        switch (currentSkill)
        {
            case 0:

                break;
            case 1:
                break;
            case 2:
                break;
            case 3:
                if(currentSkillState == SkillState.Ready && skill_temp == null){
                    skill_temp = Instantiate(attack_purple); // attack1 alive
                    rasengan = skill_temp.GetComponent<Rasengan>();
                    rasengan.ready(transform.position, transform.forward, 0.5f);
                }
                else if(currentSkillState == SkillState.Stable && skill_temp != null){
                    // if(transform.position.z < 9.0f){
                    //     transform.position = transform.position + Vector3.forward * Time.deltaTime * 9.0f;
                    //     rasengan.setPosition(transform.position);
                    // }
                }
                else if(currentSkillState == SkillState.Finish && skill_temp != null){
                    rasengan.finish(0.5f);
                    currentSkillState = SkillState.Idle;
                }

                break;
            case 4:
                if(currentSkillState == SkillState.Ready && skill_temp == null){
                    skill_temp = Instantiate(defense_red); // attack1 alive
                    magicCircle = skill_temp.GetComponent<MagicCircle>();
                    magicCircle.ready(transform.position, transform.forward, 0.5f);
                }
                else if(currentSkillState == SkillState.Stable && skill_temp != null){
                    //magicCircle.finish(transform.position);
                }
                else if(currentSkillState == SkillState.Finish && skill_temp != null){
                    magicCircle.finish(0.5f);
                    currentSkillState = SkillState.Idle;
                }

                break;
            case 5:
                break;
            case 6:
                break;
            case 7:
                break;                
        }
        
        
        // if(attack_ready && skill_temp == null){
        //     skill_temp = Instantiate(defense_red); // attack1 alive
        //     skillScript = skill.GetComponent<Rasengan>();
        //     magicCircle = skill_temp.GetComponent<MagicCircle>();
        //     magicCircle.ready(transform.position, transform.forward, 1.0f);
        // }
        // else if(attack_release){
        //     if(skill){
        //         transform.position = transform.position + Vector3.forward * Time.deltaTime * 9.0f;
        //         magicCircle.setPosition(transform.position);
        //         if(transform.position.z >= 9.0f){
        //             attack_release = false;
        //             magicCircle.finish(1.0f);
        //         }
        //     }
        // }
    }

    public void controllerReady(){
        // skillScript.ready(transform.position, transform.forward, 1.0f);
        currentSkillState = SkillState.Ready;
    }

    public void controllerRelease(){
        currentSkillState = SkillState.Stable;
    }
     public void controllerFinish(){
        currentSkillState = SkillState.Finish;
    }


    public void ChangeSkill(int Name)
    {
        currentSkill = Name;
    }
}
