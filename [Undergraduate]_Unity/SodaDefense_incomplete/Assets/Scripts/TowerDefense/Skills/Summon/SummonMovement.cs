// #define DEBUG_MODE
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SummonMovement : MonoBehaviour
{
    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private BaseCharacter playerController;
    public Summon summon;

    /* Summon 정보 */
    private float attackRange = 2.0f; // 공격 범위
    private float attackCooldown = 0.2f; // 공격 대기 시간
    private Transform target; // 소환수가 추적할 타겟
    private float lastAttackTime;

    [SerializeField]
    private GameObject attackPrefab;

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        playerController = player.GetComponent<BaseCharacter>();
        summon = player.GetComponentInChildren<Summon>();
        
        summon.attackDamage = playerController.damage * summon.damageDecreaseAmount * attackCooldown;
        summon.attackRange = playerController.baseMaxDistance * summon.damageDecreaseAmount * 2.0f;
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
            // Player의 능력치에 비례하여 Summon의 damage 설정
            summon.attackDamage = playerController.damage * summon.damageDecreaseAmount * attackCooldown; // 1초당 데미지를 위해 attackCooldown 곱함
            summon.attackRange = playerController.baseMaxDistance * summon.damageDecreaseAmount * 2.0f; // 일단 2배 함

            FindTarget(); // 타겟을 찾습니다.
            if (target != null)
            {
                MoveTowardsTarget(); // 타겟을 향해 이동합니다.
                if (IsTargetInRange())
                {
                    ProjectileShoot();
                    // AttackTarget(); // 타겟이 범위 안에 있으면 공격합니다.
                }
            }
        }
        else
        {
            summon.isSummon = false;
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

        // enemy하고 거리가 좀 있었으면 좋겠어서 추가 (attackRange의 80%로 잡음)
        if( Vector3.Distance(transform.position, target.position) >= summon.attackRange * 0.8f)
        {
            transform.position += direction * summon.moveSpeed * Time.deltaTime;
        }
        else if ( Vector3.Distance(transform.position, target.position) >= 1.0f)
        {
            // 랜덤한 방향의 Vector3 얻기
            Vector3 randomDirection = GetRandomDirection();
            transform.position += randomDirection.normalized * summon.moveSpeed * Time.deltaTime;
        }
    }

    Vector3 GetRandomDirection()
    {
        // 2D에서 랜덤한 방향을 얻기 위해 0도에서 360도 사이의 랜덤한 각도를 생성
        float angle = Random.Range(0f, 360f);
        
        // 각도를 라디안으로 변환
        float radians = angle * Mathf.Deg2Rad;
        
        // 코사인과 사인을 사용하여 방향 벡터 계산
        float x = Mathf.Cos(radians);
        float y = Mathf.Sin(radians);
        
        // Vector3로 반환 (z는 0으로 설정)
        return new Vector3(x, y, 0f);
    }

    protected void ProjectileShoot()
    {
        // 공격 대기 시간이 지났는지 확인
        if (Time.time > lastAttackTime + attackCooldown)
        {
            // 투사체 프리팹을 소환수의 위치에 인스턴스화
            GameObject projectile = Instantiate(attackPrefab, transform.position, Quaternion.identity);

            // 인스턴스화된 투사체의 EnemyProjectile 스크립트를 가져옴
            SummonProjectile summonProjectile = projectile.GetComponent<SummonProjectile>();

            // 투사체의 타겟을 설정
            summonProjectile.SetTargetwithDamage(target, summon.attackDamage); 
            // 마지막 공격 시간을 갱신합니다.
            lastAttackTime = Time.time;
        }
    }

    bool IsTargetInRange()
    {
        // 타겟이 공격 범위 안에 있는지 확인
        return Vector3.Distance(transform.position, target.position) <= summon.attackRange;
    }

    void AttackTarget()
    {
        // 공격 대기 시간이 지났는지 확인
        if (Time.time > lastAttackTime + attackCooldown)
        {
#if DEBUG_MODE
            // 여기서 타겟에게 데미지를 입히는 로직을 구현
            Debug.Log("Attacking target for " + summon.attackDamage + " damage.");
            Debug.Log("SummonSpeed: " + summon.moveSpeed);
#endif

            // 타겟에게 데미지를 입히는 방법은 타겟의 Enemy Script의 Takedamage()에 접근해서 체력을 줄이는 식으로 구현
            AttackEnemy enemy = target.GetComponent<AttackEnemy>();
            if (enemy != null)
            {
                enemy.TakeDamage(summon.attackDamage);
            }

            // 마지막 공격 시간을 갱신합니다.
            lastAttackTime = Time.time;
// #if DEBUG_MODE
//             Debug.Log("AttackTime: " + Time.time + " / Enemy HP: " + enemy.hp);
// #endif
        }
    }

    private void OnDrawGizmos()
    {
        Gizmos.color = Color.blue;
        if (summon == null)
        {
            Gizmos.DrawWireSphere(transform.position, attackRange);
        }
        else
        {
            Gizmos.DrawWireSphere(transform.position, summon.attackRange);
        }
    }

}
