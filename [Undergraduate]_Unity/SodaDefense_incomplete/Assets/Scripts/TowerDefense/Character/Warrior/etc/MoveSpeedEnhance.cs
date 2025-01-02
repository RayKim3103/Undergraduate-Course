using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveSpeedEnhance : MonoBehaviour
{
    /*  basicmove 정보를 담을 변수 */
    private float originalSpeed;
    private float originalCoolDown;

    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private WarriorController playerController;

    /* MoveSpeedIncrease 정보를 담을 변수들 */
    private bool moveSpeedIncrease = false;
    [SerializeField]
    private float moveSpeedIncreaseAmount = 2.0f; // 100% 증가
    // [SerializeField]
    // private float moveSpeedCooldownDecreaseAmount = 0.5f; // 0.5배로 기본공격 쿨타임 감소
    private float moveSpeedMaintainanceTime = 3.0f;

        public bool isKeyCodeAssign = false;
    public KeyCode keyCodemoveSpeed; // 할당된 key 누르면 MoveSpeedIncrease

    [SerializeField]
    private float moveSpeedIncreaseCooldown = 5.0f; // moveSpeedIncrease 쿨타임
    private float lastmoveSpeedIncreaseTime; // 마지막 moveSpeedIncrease 시간
    private GameObject moveSpeedIncreaseInstance;

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        playerController = player.GetComponent<WarriorController>();
    }

    public void MoveSpeedIncrease()
    {
        if (Input.GetKey(keyCodemoveSpeed) && Time.time >= lastmoveSpeedIncreaseTime + moveSpeedIncreaseCooldown)
        {
            moveSpeedIncrease = true;
            // moveSpeedIncrease가 발동 되었을 때의 기존 moveSpeed 저장
            originalSpeed = playerController.moveSpeed;
            originalCoolDown = playerController.moveSpeed;


            // playerController.moveSpeed 증가
            playerController.moveSpeed = playerController.moveSpeed * moveSpeedIncreaseAmount;

            // 아래 코드는 moveSpeed에 쿨타임이 있을 때 사용하는 부분, moveSpeed는 쿨타임이 있는게 아니니 아래 코드를 사용할 필요 없음
            // playerController.moveSpeed = playerController.moveSpeed * moveSpeedCooldownDecreaseAmount; 
            Debug.Log(" moveSpeedIncrease for " + moveSpeedIncreaseAmount + " / moveSpeed is now " + playerController.moveSpeed);
            // Debug.Log(" moveCooldownDecrease for " + moveSpeedCooldownDecreaseAmount + " / moveCooldown is now " + playerController.moveSpeed);

            // 마지막 사용 시간을 현재 시간으로 업데이트
            lastmoveSpeedIncreaseTime = Time.time;
        }
        else if(moveSpeedIncrease && (Time.time >= lastmoveSpeedIncreaseTime + moveSpeedMaintainanceTime))
        {
            moveSpeedIncrease = false;
            playerController.moveSpeed = originalSpeed;
            playerController.moveSpeed = originalCoolDown;
            Destroy(moveSpeedIncreaseInstance);
            Debug.Log(" Player moveSpeed does not Increase Anymore: " + playerController.moveSpeed);
            Debug.Log(" Player CoolDown does not Decrease Anymore: " + playerController.moveSpeed);
        }
    }
}
