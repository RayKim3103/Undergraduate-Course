using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Summon : MonoBehaviour, ISkill
{
    [SerializeField]
    private GameObject summonPrefab;
    public bool isSummon = false;
    private GameObject clone;

    /* Summon 정보 */
    public float attackDamage; // 공격력
    public float attackRange = 3.0f; // 공격 사거리
    public float damageDecreaseAmount = 0.2f;
    public float moveSpeed = 1.0f; // 소환수의 이동 속도
    public float baseMoveSpeed = 1.0f; // 소환수의 original 이동 속도

    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private BaseCharacter playerController;

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        playerController = player.GetComponent<BaseCharacter>();
    }

    public void Activate()
    {
        if(isSummon == false)
        {
            isSummon = true;
            Debug.Log("Summon is now true!!!");
            clone = Instantiate(summonPrefab, transform.position, Quaternion.identity);
            clone.name = "BasicSummon";
            clone.transform.localScale = clone.transform.localScale * 0.5f;
        }
    }
    public void SkillEffectActivate(Transform target)
    {
        // 인터페이스를 위한 공란 함수 정의
    }

    
    public void LevelUp(float increaseAmount)
    {
        // 0렙때, 0.2f -> 1렙때, 0.4f
        damageDecreaseAmount += increaseAmount;
        this.moveSpeed = baseMoveSpeed + baseMoveSpeed * increaseAmount;
        Debug.Log("Summon damageDecreaseAmount: " + damageDecreaseAmount);
        Debug.Log("Summon moveSpeed: " + moveSpeed  );
    }

    // public void ReleaseSkill()
    // {
    //     isSummon = false;
    //     Destroy(clone);
    // }
}
