#define DEBUG_MODE
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LifeStealSkill : MonoBehaviour, ISkill
{
    
    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private BaseCharacter playerController;

    /* LifeSteal 관련 변수 */
    public float lifeStealPercentage = 0.3f; // 흡혈 비율 (30%)

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        playerController = player.GetComponent<BaseCharacter>();
    }
    public void Activate()
    {
        // 인터페이스를 위한 공란 함수 정의
    }

    // 기본 공격이 적에게 명중할 때 호출되는 메서드, target은 필요없는 파라미터지만, 인터페이스를 위하여 넣음
    public void SkillEffectActivate(Transform target)
    {
        float healthRecovered = playerController.damage * lifeStealPercentage;
        
        playerController.hp += healthRecovered;
        
        if (playerController.hp > playerController.maxHp) //최대체력 초과 시
        {
            playerController.hp = playerController.maxHp; //최대체력
        }

#if DEBUG_MODE
        Debug.Log("LifeSteal applied: " + healthRecovered + " HP recovered. Current health: " + playerController.hp);
#endif
    }

    public void LevelUp(float increaseAmount)
    {
        lifeStealPercentage += increaseAmount;
    }
}
