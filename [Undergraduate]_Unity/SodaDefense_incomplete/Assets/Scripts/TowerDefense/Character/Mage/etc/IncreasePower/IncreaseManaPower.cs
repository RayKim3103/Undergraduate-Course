using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// Player 에 넣을 스크립트
public class IncreaseManaPower : MonoBehaviour
{
    public bool isKeyCodeAssign = false;
    public KeyCode keyCodePower;

    /* BasicAttack 정보를 담을 변수 */
    private float originalDamage;

    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private MageController playerController;

    /* ManaPowerIncrease 정보를 담을 변수들 */
    private bool manaPowerIncrease = false;
    [SerializeField]
    private float manaPowerIncreaseAmount = 2.0f; // 100% 증가
    private float manaPowerMaintainanceTime = 3.0f;

    [SerializeField]
    private GameObject manaPowerPrefab;
    [SerializeField]
    private float manaPowerCooldown = 5.0f; // ManaPowerIncrease 쿨타임
    private float lastManaPowerTime; // 마지막 ManaPowerIncrease 시간
    private GameObject manaPowerInstance;

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        playerController = player.GetComponent<MageController>();
    }

    public void ManaPowerIncrease()
    {
        if (Input.GetKey(keyCodePower) && Time.time >= lastManaPowerTime + manaPowerCooldown && isKeyCodeAssign)
        {
            manaPowerIncrease = true;
            // ManaPower가 발동 되었을 때의 기존 Damage 저장
            originalDamage = playerController.damage;

            // Prefab 인스턴스화
            manaPowerInstance = Instantiate(manaPowerPrefab, player.transform.position, Quaternion.identity);

            // playerController.damage 증가
            playerController.damage = playerController.damage * manaPowerIncreaseAmount;
            Debug.Log(" manaPowerIncrease for " + manaPowerIncreaseAmount + " manaPower is now " + playerController.damage);

            // 마지막 사용 시간을 현재 시간으로 업데이트
            lastManaPowerTime = Time.time;
        }
        else if(manaPowerIncrease && (Time.time >= lastManaPowerTime + manaPowerMaintainanceTime) && isKeyCodeAssign)
        {
            manaPowerIncrease = false;
            playerController.damage = originalDamage;
            Destroy(manaPowerInstance);
            Debug.Log(" Player ManaPower does not Increase Anymore: " + playerController.damage);
        }
    }
}

