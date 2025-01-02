using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CurseMovement : MonoBehaviour
{
    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private BaseCharacter playerController;
    private Curse curse;

    /* Curse 정보 */

    [SerializeField]
    private float moveSpeed = 10.0f; // 소환수의 이동 속도
    [SerializeField]
    private float attackRange = 2.0f; // 공격 범위
    [SerializeField]
    private float attackCooldown = 1.0f; // 공격 대기 시간
    

    private Transform target; // 소환수가 추적할 타겟
    private float lastAttackTime;

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        playerController = player.GetComponent<BaseCharacter>();
        curse = player.GetComponentInChildren<Curse>();
    }
    
    void Start()
    {
        // 처음에는 타겟이 없으므로 null로 설정
        target = null;
        lastAttackTime = 0f;
    }

    void Update()
    {
        if (playerController.isDied == false)
        {
            // Player의 능력치에 비례하여 Curse의 damage 설정
            curse.curseDamage = playerController.damage * curse.damageDecreaseAmount;

            FindTarget(); // 타겟을 찾습니다.
            if (target != null)
            {
                if (IsTargetInRange())
                {
                    MoveTowardsTarget(); // 타겟을 향해 이동합니다.
                    AttackTarget(); // 타겟이 범위 안에 있으면 공격합니다.
                }
                else
                {
                    Destroy(gameObject);
                }
            }
            else
            {
                Destroy(gameObject);
            }
        }
        else
        {
            Destroy(gameObject);
        }
        
    }

    void FindTarget()
    {
        // 적을 찾는 로직을 구현, 태그가 "Enemy"인 모든 적을 찾음
        GameObject[] enemies = GameObject.FindGameObjectsWithTag("Enemy");
        float closestDistance = Mathf.Infinity; // Distance 무한으로 초기화
        GameObject closestEnemy = null;         // 가까운 적은 없음으로 초기화

        foreach (GameObject enemy in enemies)
        {
            // Distance: 자기 자신과 target의 거리 반환
            float distance = Vector3.Distance(transform.position, enemy.transform.position);
            
            // foreach를 이용하여 가장 가까운 거리의 적을 저장
            if (distance < closestDistance)
            {
                closestDistance = distance;
                closestEnemy = enemy;
            }
        }

        // foreach문이 끝나고 closestEnemy가 null이 아니면, target 결정
        if (closestEnemy != null)
        {
            target = closestEnemy.transform;
        }
    }

    void MoveTowardsTarget()
    {
        // 타겟을 향해 이동하는 로직
        Vector3 direction = (target.position - transform.position).normalized; // 속도를 일정하게 하기 위하여 정규화
        transform.position += direction * moveSpeed * Time.deltaTime;
    }

    bool IsTargetInRange()
    {
        // 타겟이 공격 범위 안에 있는지 확인
        return Vector3.Distance(transform.position, target.position) <= attackRange;
    }

    void AttackTarget()
    {
        // 공격 대기 시간이 지났는지 확인
        if (Time.time > lastAttackTime + attackCooldown)
        {
#if DEBUG_MODE
            // 여기서 타겟에게 데미지를 입히는 로직을 구현
            Debug.Log("Attacking target for " + attackDamage + " damage.");
#endif
            // 타겟에게 데미지를 입히는 방법은 타겟의 Enemy Script의 Takedamage()에 접근해서 체력을 줄이는 식으로 구현
            AttackEnemy enemy = target.GetComponent<AttackEnemy>();
            if (enemy != null)
            {
                enemy.TakeDamage(curse.curseDamage);
            }

            // 마지막 공격 시간을 갱신합니다.
            lastAttackTime = Time.time;
#if DEBUG_MODE
            Debug.Log("AttackTime: " + Time.time + " / Enemy HP: " + enemy.hp);
#endif
        }
    }
}
