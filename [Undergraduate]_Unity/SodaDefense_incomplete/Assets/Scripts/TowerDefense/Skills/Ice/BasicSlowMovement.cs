using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/* Ice의 움직임 */
public class BasicSlowMovement : MonoBehaviour
{
    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private BaseCharacter playerController;
    private IceAttack iceAttack;

    /* Ice 의 정보 */
    private Transform target; // ICE가 추적할 타겟
    private float timeCount = 0.0f; // 생성되었을 때 시간 기록, 지속시간을 위해

    [HideInInspector]
    // public bool isIce = false;
    private CapsuleCollider2D capsuleCollider2D;

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        playerController = player.GetComponent<BaseCharacter>();
        iceAttack = player.GetComponentInChildren<IceAttack>();
    }

    private void Start()
    {
        // 생성되었을 때 시간 기록, 지속시간을 위해
        timeCount = Time.time;

        FindTarget(); // 타겟을 찾습니다.
        if (target != null)
        {
            IcePosition(); // 타겟의 발 위치에 생성
            IceAttackTarget(); // 타겟이 범위 안에 있으면 공격합니다.   
        }
    }

    void Update()
    {
        // Player의 능력치에 비례하여 iceAttack의 damage 설정
        iceAttack.iceDamage = playerController.damage * iceAttack.iceDamageDecreaseAmount;

        // Enemy가 IceAttackTarget에 의해 Destroy 되었을 때, if BasicAttack에 의해 죽었을 때 이 스크립트가 실행되면, target을 찾을 수 없기에 에러 O
        // BasicAttack에 의해 죽었을 때는 다른 코드에서 이 스크립트가 실행되지 않도록 방지해야 함 (ex. enemy.hp > 0)
        if (target == null)
        {
            // isIce = false;
            Destroy(gameObject);
        }    

        // ICE 지속시간 끝났을 때, isSpeedReduced도 참조해야 함(얼어있으면 추가로 더 얼지는 않게 하였기에)
        else if(Time.time > iceAttack.maintainanceTime + timeCount || target.GetComponent<EnemyMovement>().isSpeedReduced == false)
        {
            // isIce = false;
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
#if DEBUG_MODE
            Debug.Log("Ice Target is: " + target);
#endif
        }
    }

    void IcePosition()
    {
        // CapsuleCollider2D에서 bounds를 가져와서 발 위치 계산
        capsuleCollider2D = target.GetComponent<CapsuleCollider2D>();
        Bounds bounds = capsuleCollider2D.bounds;
        Vector2 footPosition = new Vector2(bounds.center.x, bounds.min.y);
        transform.position = footPosition;
    }

    
    void IceAttackTarget()
    {
//#if DEBUG_MODE
        // 여기서 타겟에게 데미지를 입히는 로직을 구현
        Debug.Log("Attacking target for " + iceAttack.iceDamage + " damage.");
//#endif
        // 타겟에게 데미지를 입히는 방법은 타겟의 Enemy Script의 Takedamage()에 접근해서 체력을 줄이는 식으로 구현
        AttackEnemy enemy = target.GetComponent<AttackEnemy>();
        if (enemy != null)
        {
            enemy.TakeDamage(iceAttack.iceDamage);
        }
#if DEBUG_MODE
        Debug.Log("AttackTime: " + Time.time + " / Enemy HP: " + enemy.hp);
#endif
    }

}
