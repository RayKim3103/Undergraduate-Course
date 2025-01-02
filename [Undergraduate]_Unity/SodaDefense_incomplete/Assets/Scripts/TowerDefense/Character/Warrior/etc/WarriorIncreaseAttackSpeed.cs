using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WarriorIncreaseAttackSpeed : MonoBehaviour
{
    public bool isKeyCodeAssign = false;
    public KeyCode keyCodeSpeed;

    /* BasicAttack 정보를 담을 변수 */
    private float originalSpeed;
    private float originalCoolDown;

    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private WarriorController playerController;

    /* MageAttackSpeedIncrease 정보를 담을 변수들 */
    private bool warriorattackSpeedIncrease = false;
    [SerializeField]
    private float warriorattackSpeedIncreaseAmount = 2.0f; // 100% 증가
    [SerializeField]
    private float warriorattackSpeedCooldownDecreaseAmount = 0.5f; // 0.5배로 기본공격 쿨타임 감소
    private float warriorattackSpeedMaintainanceTime = 3.0f;

    [SerializeField]
    private GameObject warriorAttackSpeedPrefab;
    
    [SerializeField]
    private float warriorattackSpeedCooldown = 5.0f; // AttackSpeedIncrease 쿨타임
    private float warriorlastAttackSpeedTime; // 마지막 AttackSpeedIncrease 시간
    private GameObject warriorattackSpeedInstance;

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        playerController = player.GetComponent<WarriorController>();
    }

    public void AttackSpeedIncrease()
    {
        if (Input.GetKey(keyCodeSpeed) && Time.time >= warriorlastAttackSpeedTime + warriorattackSpeedCooldown && isKeyCodeAssign)
        {
            warriorattackSpeedIncrease = true;
            // AttackSpeedIncrease가 발동 되었을 때의 기존 attackSpeed 저장
            originalSpeed = playerController.attackSpeed;
            originalCoolDown = playerController.fireCooldown;

            // Prefab 인스턴스화
            warriorattackSpeedInstance = Instantiate(warriorAttackSpeedPrefab, player.transform.position, Quaternion.identity);

            // playerController.attackSpeed 증가
            playerController.attackSpeed = playerController.attackSpeed * warriorattackSpeedIncreaseAmount;
            playerController.fireCooldown = playerController.fireCooldown * warriorattackSpeedCooldownDecreaseAmount;
            Debug.Log(" AttackSpeedIncrease for " + warriorattackSpeedIncreaseAmount + " / AttackSpeed is now " + playerController.attackSpeed);
            Debug.Log(" AttackCooldownDecrease for " + warriorattackSpeedCooldownDecreaseAmount + " / AttackCooldown is now " + playerController.fireCooldown);

            // 마지막 사용 시간을 현재 시간으로 업데이트
            warriorlastAttackSpeedTime = Time.time;
        }
        else if(warriorattackSpeedIncrease && (Time.time >= warriorlastAttackSpeedTime + warriorattackSpeedMaintainanceTime) && isKeyCodeAssign)
        {
            warriorattackSpeedIncrease = false;
            playerController.attackSpeed = originalSpeed;
            playerController.fireCooldown = originalCoolDown;
            Destroy(warriorattackSpeedInstance);
            Debug.Log(" Player AttackSpeed does not Increase Anymore: " + playerController.attackSpeed);
            Debug.Log(" Player CoolDown does not Decrease Anymore: " + playerController.fireCooldown);
        }
    }
}
