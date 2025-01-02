using System.Collections;
using UnityEngine;

public class Shield : MonoBehaviour
{
    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private WarriorController playerController;

    /* Shield 정보를 담을 변수들 */
    private bool isImmune = false;
    private float originalHp; // 쉴드 사용 시점의 체력
    private float shieldHpBoost = 100.0f; // 쉴드 사용 시 체력 증가량
    [SerializeField]
    private float immunityDuration = 20.0f; // 피해 무시 지속 시간
    [SerializeField]
    private float immunityCooldown = 10.0f; // 피해 무시 쿨다운 시간
    private float lastImmunityTime; // 마지막 피해 무시 시간

    public bool isKeyCodeAssign = false;
    public KeyCode keyCodeShield; // 할당된 key 누르면 Shield

    private void Awake()
    {
        player = GameObject.FindWithTag(playerTag);
        playerController = player.GetComponent<WarriorController>();
    }

    private void Start()
    {
        // 디버그 로그 추가
        Debug.Log("Shield key code assigned: " + keyCodeShield);
    }

    public void ActivateShield()
    {
        if (Input.GetKey(keyCodeShield) && Time.time >= lastImmunityTime + immunityCooldown)
        {
            isImmune = true;
            originalHp = playerController.hp; // 현재 체력을 저장
            Debug.Log("Original Hp before shield: " + originalHp);

            playerController.hp += shieldHpBoost; // 체력을 크게 증가시킴
            Debug.Log("Shield activated. Player is immune to damage. Current Hp after boost: " + playerController.hp);

            // 마지막 사용 시간을 현재 시간으로 업데이트
            lastImmunityTime = Time.time;

            // 일정 시간이 지나면 피해 무시 상태 해제 및 체력 복원
            StartCoroutine(ShieldDurationCoroutine());
        }
    }

    private IEnumerator ShieldDurationCoroutine()
    {
        yield return new WaitForSeconds(immunityDuration);
        isImmune = false;

        // 현재 체력에서 쉴드로 증가된 체력을 빼고, 원래 체력과 비교하여 더 작은 값을 설정
        playerController.hp = Mathf.Min(originalHp, 10);
        Debug.Log("Shield deactivated. Player is no longer immune to damage. Current Hp after revert: " + playerController.hp);
    }

    public bool IsImmune()
    {
        return isImmune;
    }
}
