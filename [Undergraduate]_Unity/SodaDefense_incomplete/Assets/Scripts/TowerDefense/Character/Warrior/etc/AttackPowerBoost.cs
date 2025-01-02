using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// Player 에 넣을 스크립트
public class AttackPowerBoost : MonoBehaviour
{
    /* BasicAttack 정보를 담을 변수 */
    private float originalDamage;

    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private WarriorController playerController;

    /* ManaPowerIncrease 정보를 담을 변수들 */
    private bool attackBoost = false;
    [SerializeField]
    private float attackBoostAmount = 2.0f; // 100% 증가
    private float attackBoostMaintainanceTime = 3.0f;

    public bool isKeyCodeAssign = false;
    public KeyCode keyCodeAttackBoost; // 할당된 key 누르면 AttackBoost

    [SerializeField]
    private GameObject attackBoostPrefab;
    [SerializeField]
    private float attackBoostCooldown = 5.0f; // ManaPowerIncrease 쿨타임
    private float lastattackBoostTime; // 마지막 ManaPowerIncrease 시간
    private GameObject attackBoostInstance;

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        playerController = player.GetComponent<WarriorController>();
    }

    public void ActivateAttackBoost()
    {
        if (Input.GetKey(keyCodeAttackBoost) && Time.time >= lastattackBoostTime + attackBoostCooldown)
        {
            attackBoost = true;
            // ManaPower가 발동 되었을 때의 기존 Damage 저장
            originalDamage = playerController.damage;

            // Prefab 인스턴스화
            attackBoostInstance = Instantiate(attackBoostPrefab, player.transform.position, Quaternion.identity);

            // playerController.damage 증가
            playerController.damage = playerController.damage * attackBoostAmount;
            Debug.Log(" manaPowerIncrease for " + attackBoostAmount + " manaPower is now " + playerController.damage);

            // 마지막 사용 시간을 현재 시간으로 업데이트
            lastattackBoostTime = Time.time;
        }
        else if(attackBoost && (Time.time >= lastattackBoostTime + attackBoostMaintainanceTime))
        {
            attackBoost = false;
            playerController.damage = originalDamage;
            Destroy(attackBoostInstance);
            Debug.Log(" Player ManaPower does not Increase Anymore: " + playerController.damage);
        }
    }
}

