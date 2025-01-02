using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WarriorController : BaseCharacter
{   
    /* preFab 지속시간 관리 */

    public GameObject attackInstancePrefab; // 게임화면상에 공격할 때 잠깐 출력될 프리팹
    private GameObject spawnedObject; // 생성된 오브젝트 저장

    [SerializeField]
    private float instancePrefabDuration = 0.2f; // 파괴될 때까지 대기할 시간
    
    protected override void Awake()
    {
        BaseCharacter baseCharacter = GetComponent<BaseCharacter>();
        // SpriteRenderer 컴포넌트를 가져옴
        base.Awake();
        maxDistance = 2.0f;
        damage = damage * 2.0f;
        //baseCharacter.maxDistance = 2.0f;
        // maxDistance = baseCharacter.maxDistance;
        //baseCharacter.damage = damage * 2.0f;
        // damage = baseCharacter.damage;
        // Debug.Log("WarriorController Awake");
        // Debug.Log("BaseCharacter Damge: " + baseCharacter.damage);
    }

    protected override void Update()
    {
        base.Update();
    }

    protected override void HandleAttack()
    {
        targets = Physics2D.CircleCastAll(transform.position, maxDistance, Vector2.zero, 0, targetLayer);
        nearestTarget = GetNearestTarget();

        if (nearestTarget && Time.time >= lastFireTime + fireCooldown)
        {
            Vector3 direction = (nearestTarget.position - transform.position).normalized;
            clone = Instantiate(attackPrefab, transform.position, Quaternion.Euler(new Vector3(0f, 0f, RotateToDirection(direction))));
            clone.name = "BasicAttack";
            clone.GetComponent<BasicAttack>().Setup(direction, attackSpeed, damage);
            spawnedObject = Instantiate(attackInstancePrefab, transform.position, Quaternion.Euler(new Vector3(0f, 0f, RotateToDirection(direction))));
            // Debug.Log("Nearest Target: " + nearestTarget +" Direction: " + direction);
            lastFireTime = Time.time;
        }
        Destroy(clone, instancePrefabDuration);
        Destroy(spawnedObject, instancePrefabDuration);
    }

    /* 디버깅용: player의 기본 공격범위를 표시 */
    /* 디버깅용: 발사체의 적 탐지 및 공격범위를 표시 */
    private void OnDrawGizmos()
    {
        Gizmos.color = Color.blue;
        Gizmos.DrawWireSphere(transform.position, attackPrefab.GetComponent<BasicSkillEffect>().increaseDetection);
    }
}