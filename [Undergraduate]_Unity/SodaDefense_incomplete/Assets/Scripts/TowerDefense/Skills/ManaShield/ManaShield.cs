using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ManaShield : MonoBehaviour, ISkill
{
    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private BaseCharacter playerController;

    /* ManaShield 정보를 담을 변수 */
    public bool manaShieldActive = false;

    public float damageReduceAmount = 0.7f; // 30% 피해감소

    // private GameObject enemy;

    [SerializeField]
    private GameObject shieldPrefab;
    private GameObject shieldInstance;

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        playerController = player.GetComponent<BaseCharacter>();
    }

    public void Activate()
    {
        if(manaShieldActive == false)
        {
            // manaShieldActivate
            manaShieldActive = true;
            Debug.Log("ManaShieldActive is now true!!!");

            // Shield Prfab 생성
            shieldInstance = Instantiate(shieldPrefab, transform.position, Quaternion.identity);
        }
    }
    public void SkillEffectActivate(Transform target)
    {
        // 인터페이스를 위한 공란 함수 정의
    }

    // public void ReleaseSkill()
    // {
    //     manaShieldActive = false;
    //     Debug.Log("ManaShield Inactive / Player Died");
    //     Destroy(shieldInstance);
    // }
    public void LevelUp(float increaseAmount)
    {
        // 피해량 감소해야 하므로 뺄셈 연산
        damageReduceAmount -= increaseAmount;
    }
}
