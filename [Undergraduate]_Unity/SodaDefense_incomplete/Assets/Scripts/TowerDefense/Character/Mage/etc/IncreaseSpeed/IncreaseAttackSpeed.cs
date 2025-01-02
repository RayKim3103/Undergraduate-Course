using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class IncreaseAttackSpeed : MonoBehaviour
{
    public bool isKeyCodeAssign = false;
    public KeyCode keyCodeSpeed;

    /* BasicAttack 정보를 담을 변수 */
    private float originalSpeed;
    private float originalCoolDown;

    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private MageController playerController;

    /* MageAttackSpeedIncrease 정보를 담을 변수들 */
    private bool attackSpeedIncrease = false;
    [SerializeField]
    private float attackSpeedIncreaseAmount = 2.0f; // 100% 증가
    [SerializeField]
    private float attackSpeedCooldownDecreaseAmount = 0.5f; // 0.5배로 기본공격 쿨타임 감소
    private float attackSpeedMaintainanceTime = 3.0f;

    [SerializeField]
    private GameObject mageAttackSpeedPrefab;
    [SerializeField]
    private float attackSpeedCooldown = 5.0f; // AttackSpeedIncrease 쿨타임
    private float lastAttackSpeedTime; // 마지막 AttackSpeedIncrease 시간
    private GameObject attackSpeedInstance;

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        playerController = player.GetComponent<MageController>();
    }

    public void AttackSpeedIncrease()
    {
        if (Input.GetKey(keyCodeSpeed) && Time.time >= lastAttackSpeedTime + attackSpeedCooldown && isKeyCodeAssign)
        {
            attackSpeedIncrease = true;
            // AttackSpeedIncrease가 발동 되었을 때의 기존 attackSpeed 저장
            originalSpeed = playerController.attackSpeed;
            originalCoolDown = playerController.fireCooldown;

            // Prefab 인스턴스화
            attackSpeedInstance = Instantiate(mageAttackSpeedPrefab, player.transform.position, Quaternion.identity);

            // playerController.attackSpeed 증가
            playerController.attackSpeed = playerController.attackSpeed * attackSpeedIncreaseAmount;
            playerController.fireCooldown = playerController.fireCooldown * attackSpeedCooldownDecreaseAmount;
            Debug.Log(" AttackSpeedIncrease for " + attackSpeedIncreaseAmount + " / AttackSpeed is now " + playerController.attackSpeed);
            Debug.Log(" AttackCooldownDecrease for " + attackSpeedCooldownDecreaseAmount + " / AttackCooldown is now " + playerController.fireCooldown);

            // 마지막 사용 시간을 현재 시간으로 업데이트
            lastAttackSpeedTime = Time.time;
        }
        else if(attackSpeedIncrease && (Time.time >= lastAttackSpeedTime + attackSpeedMaintainanceTime) && isKeyCodeAssign)
        {
            attackSpeedIncrease = false;
            playerController.attackSpeed = originalSpeed;
            playerController.fireCooldown = originalCoolDown;
            Destroy(attackSpeedInstance);
            Debug.Log(" Player AttackSpeed does not Increase Anymore: " + playerController.attackSpeed);
            Debug.Log(" Player CoolDown does not Decrease Anymore: " + playerController.fireCooldown);
        }
    }
}
