using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Heal : MonoBehaviour, ISkill
{
    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private BaseCharacter playerController;

    /* Heal 정보를 담을 변수들 */
    private float healAmount = 3.0f;
    private float baseHealAmount = 3.0f;

    [SerializeField]
    private GameObject healPrefab;
    [SerializeField]
    private float healCooldown = 10.0f; // 힐 쿨타임
    private float lasthealTime; // 마지막 힐 시간
    //private GameObject healInstance;

    /* preFab 생성 위치 변수 */
    private CapsuleCollider2D capsuleCollider2D;
    private Vector3 headPosition;

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        playerController = player.GetComponent<BaseCharacter>();
        // Debug.Log("Heal Awake / playerController: " + playerController);
    }
    public void SkillEffectActivate(Transform target)
    {
        // 인터페이스를 위한 공란 함수 정의
    }
    public void Activate()
    {
        if (Time.time >= lasthealTime + healCooldown)
        {
            // CapsuleCollider2D에서 bounds를 가져와서 머리 위치 계산
            capsuleCollider2D = GetComponentInParent<CapsuleCollider2D>();
            Bounds bounds = capsuleCollider2D.bounds;
            headPosition = new Vector2(bounds.center.x, bounds.max.y);

            // 머리 위치에 Prefab 인스턴스화
            Instantiate(healPrefab, headPosition, Quaternion.identity);

            playerController.hp += healAmount;
            if(playerController.hp >= playerController.maxHp)
            {
                playerController.hp = playerController.maxHp;
                Debug.Log(" Player Hp is Max: " + playerController.hp);
            }
            else
            {
                Debug.Log(" heals for " + healAmount + " HP is now " + playerController.hp);
            }
            

            // 마지막 사용 시간을 현재 시간으로 업데이트
            lasthealTime = Time.time;
        }
    }

    public void LevelUp(float increaseAmount)
    {
        // 0렙때, 0.2f -> 1렙때, 0.4f
        this.healAmount = baseHealAmount + baseHealAmount * increaseAmount;
    }
}
