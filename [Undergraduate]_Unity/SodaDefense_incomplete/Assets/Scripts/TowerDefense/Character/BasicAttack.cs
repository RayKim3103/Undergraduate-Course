using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BasicAttack : MonoBehaviour
{
    public string playerTag = "Player";
    protected GameObject player;
    protected Vector3 moveDirection;
    protected float moveSpeed;
    protected float damage;
    public float attackRange = 1.0f;
    public bool isLongRange = false;

    protected Vector3 initialPlayerPosition;
    protected Transform target;

    protected virtual void Start() 
    {
        player = GameObject.FindWithTag(playerTag);
        initialPlayerPosition = player.transform.position;
    }

    protected virtual void Update() 
    {
        MoveProjectile();
        if (IsTooFarFromInitialPosition())
        {
            Destroy(gameObject);
            return;
        }

        if (target != null && IsTargetInRange())
        {
            AttackTarget();
        }

        if (target == null)
        {
            FindTarget();
        }
    }

    public void Setup(Vector3 direction, float attackSpeed, float damage) 
    {
        moveDirection = direction;
        moveSpeed = attackSpeed;
        this.damage = damage;
    }

    protected void MoveProjectile()
    {
        transform.position += moveDirection * moveSpeed * Time.deltaTime;
    }

    protected bool IsTooFarFromInitialPosition()
    {
        return Vector3.Distance(initialPlayerPosition, transform.position) > player.GetComponent<BaseCharacter>().baseMaxDistance;
    }

    protected void FindTarget()
    {
        GameObject closestEnemy = null;
        float closestDistance = Mathf.Infinity;
        // 적을 찾는 로직을 구현, 태그가 "Enemy"인 모든 적을 찾음
        GameObject[] enemies = GameObject.FindGameObjectsWithTag("Enemy");
        
        foreach (GameObject enemy in enemies)
        {
            float distance = Vector3.Distance(transform.position, enemy.transform.position);
            if (distance < closestDistance)
            {
                closestDistance = distance;
                closestEnemy = enemy;
            }
        }

        if (closestEnemy != null)
        {
            target = closestEnemy.transform;
        }
    }

    protected bool IsTargetInRange()
    {
        return Vector3.Distance(transform.position, target.position) <= attackRange;
    }

    protected virtual void AttackTarget()
    {
#if DEBUG_MODE
        Debug.Log("Attacking target for " + damage + " damage.");
#endif
        AttackEnemy enemy = target.GetComponent<AttackEnemy>();
        if (enemy != null)
        {
            enemy.TakeDamage(damage);
            Debug.Log($"Basic Attack Hit!! {enemy.GetType().Name} Hp: {enemy.hp}");
            Destroy(gameObject);
        }
    }

    private void OnDrawGizmos()
    {
        if(player.GetComponent<BaseCharacter>() != null)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawWireSphere(player.transform.position, player.GetComponent<BaseCharacter>().baseMaxDistance);
            Gizmos.color = Color.blue;
            Gizmos.DrawWireSphere(transform.position, attackRange);
        }
    }
}


// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine;

// public class BasicAttack : MonoBehaviour
// {
//     /* Player 정보를 담을 변수 */
//     public string playerTag = "Player"; // Player 오브젝트의 태그
//     protected GameObject player;
//     protected  MageController mageController;
//     protected  WarriorController warriorController;
//     protected  ArcherController archerController;


//     // /* 적의 정보 */
//     // private AttackEnemy enemy;

//     /* BasicAttack 방향, 속도, 데미지 */
//     [SerializeField]
//     protected Vector3 moveDirection;
//     [SerializeField]
//     protected float moveSpeed;
//     [SerializeField]
//     protected float damage;

//     /* 발사체의 범위 변수 */
//     protected Vector3 initialPlayerPosition; // 발사 시점의 플레이어 위치

//     /* Warrior 참격이 활성화 될 시 (나중 증강 위한 변수) */
//     public bool isLongRange = false;

//     protected RaycastHit2D[] targets;
    

//     /* 공격 범위 */
//     public float attackRange = 1.0f;

//     public Transform target; // 발사체가 추적할 타겟

//     protected virtual void Awake() 
//     {
//         player = GameObject.FindWithTag(playerTag);
//         initialPlayerPosition = player.transform.position; // 발사 시점의 플레이어 위치 저장
//     }

//     protected virtual void Update() 
//     {
//         // RotateToDirection(moveDirection);
//         transform.position += moveDirection * moveSpeed * Time.deltaTime;
//         FindTarget(); // 타겟을 찾습니다.
//         target = GetNearestTarget();
//         if (target != null)
//         {
//             if (IsTargetInRange())
//             {
//                 AttackTarget(); // 타겟이 범위 안에 있으면 공격합니다.
//             }
//         }
//         else
//         {
//             // target이 없으면
//             Destroy(gameObject); // 발사체 파괴
//         }
        
//         if (Vector3.Distance(initialPlayerPosition, transform.position) > player.GetComponent<BaseCharacter>().baseMaxDistance)
//         {
//             Destroy(gameObject); // 발사체 파괴
//         }
//     }

//     public void Setup(Vector3 direction, float attackSpeed, float damage) // SetDirectionwithSpeedwithDamage
//     {
//         moveDirection = direction;
//         moveSpeed = attackSpeed;
//         this.damage = damage;
//         // this.target = target;
//     }

    
//     private Transform GetNearestTarget()
//     {
//         Transform nearest = null;
//         float minDistance = float.MaxValue;

//         foreach (RaycastHit2D hit in targets)
//         {
//             float distance = Vector3.Distance(transform.position, hit.transform.position);
//             if (distance < minDistance)
//             {
//                 minDistance = distance;
//                 nearest = hit.transform;
//             }
//         }

//         return nearest;
//     }

//     protected void FindTarget()
//     {    
//         // 적을 찾는 로직을 구현, 태그가 "Enemy"인 모든 적을 찾음
//         GameObject[] enemies = GameObject.FindGameObjectsWithTag("Enemy");
//         float closestDistance = Mathf.Infinity; // Distance 무한으로 초기화
//         GameObject closestEnemy = null;         // 가까운 적은 없음으로 초기화

//         foreach (GameObject enemy in enemies)
//         {
//             // Distance: 자기 자신과 target의 거리 반환
//             float distance = Vector3.Distance(transform.position, enemy.transform.position);
            
//             // foreach를 이용하여 가장 가까운 거리의 적을 저장
//             if (distance < closestDistance)
//             {
//                 closestDistance = distance;
//                 closestEnemy = enemy;
//             }
//         }

//         // foreach문이 끝나고 closestEnemy가 null이 아니면, target 결정
//         if (closestEnemy != null)
//         {
//             target = closestEnemy.transform;
//         }
//     }

//     protected void MoveTowardsTarget()
//     {
//         // 타겟을 향해 이동하는 로직
//         Vector3 direction = (target.position - transform.position).normalized; // 속도를 일정하게 하기 위하여 정규화
//         transform.position += direction * moveSpeed * Time.deltaTime;
//     }

//     protected bool IsTargetInRange()
//     {
//         // 타겟이 공격 범위 안에 있는지 확인
//         return Vector3.Distance(transform.position, target.position) <= attackRange;
//     }

//     protected void AttackTarget()
//     {
// #if DEBUG_MODE
//         // 여기서 타겟에게 데미지를 입히는 로직을 구현
//         Debug.Log("Attacking target for " + playerController.damage + " damage.");
// #endif
//         HandleEnemy(target.gameObject);
//     }


//     /* 추후 인터페이스에 대한 공부 후 => 인터페이스 이용하여 간단하게 해보기 */
//     protected virtual void HandleEnemy(GameObject target)
//     {
//         // // Pink, Mint, Purple, RedOctopus 또는 AttackEnemy 컴포넌트를 찾아서 가져옴
//         // var enemy = target.GetComponent<LongRange>() ?? target.GetComponent<FastSpeed>() ?? target.GetComponent<PurpleOctopus>() 
//         // ?? target.GetComponent<RedOctopus>() ?? target.GetComponent<AttackEnemy>();

//         AttackEnemy enemy = target.GetComponent<AttackEnemy>();

//         // enemy가 null이 아니면 즉, 해당 컴포넌트가 존재하면
//         if (enemy != null)
//         {
//             // 적의 타입을 로그에 출력
//             Debug.Log($"{enemy.GetType().Name} 컴포넌트가 존재합니다.");

//             // 적에게 데미지를 주고, 로그에 남김
//             enemy.TakeDamage(damage);
//             Debug.Log($"Basic Attack Hit!! {enemy.GetType().Name} Hp: {enemy.hp}");

//             /***********************************/

//             // 공격이 끝났으므로 발사체 오브젝트를 파괴
//             Destroy(gameObject);
//         }
//     }

//     /* 디버깅용: player의 기본 공격범위를 표시 */
//     private void OnDrawGizmos()
//     {
//         Gizmos.color = Color.red;
//         Gizmos.DrawWireSphere(player.transform.position, player.GetComponent<BaseCharacter>().maxDistance);
//         Gizmos.color = Color.blue;
//         Gizmos.DrawWireSphere(transform.position, attackRange);
//     }
// }
