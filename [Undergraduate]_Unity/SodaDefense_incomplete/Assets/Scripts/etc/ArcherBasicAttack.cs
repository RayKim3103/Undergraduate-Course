using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ArcherBasicAttack : MonoBehaviour
{
    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private ArcherController playerController;

    /* BasicAttack 방향 */
    private Vector3 moveDirection;

    /* 공격 범위 */
    [SerializeField]
    private float attackRange = 1.0f;

    private Transform target; // 발사체가 추적할 타겟

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        playerController = player.GetComponent<ArcherController>();
    }

    public void Setup(Vector3 direction)
    {
        moveDirection = direction;
    }
    
    private void Update() 
    {
        transform.position += moveDirection * playerController.attackSpeed * Time.deltaTime;
        
        // 입력된 방향이 있을 때만 회전 변경
        if (moveDirection != Vector3.zero)
        {
            // 방향 벡터의 각도를 계산 [arctan]하여 라디안을 도로 변환, 올바른 각도로 설정
            float angle = Mathf.Atan2(moveDirection.y, moveDirection.x) * Mathf.Rad2Deg;
            // 각도 조정
            angle += 180f;
            // transform의 rotation 조정
            transform.rotation = Quaternion.Euler(new Vector3(0f, 0f, angle));
        }

        FindTarget(); // 타겟을 찾습니다.
        if (target != null)
        {
            if (IsTargetInRange())
            {
                AttackTarget(); // 타겟이 범위 안에 있으면 공격합니다.
            }
        }
    }

    void FindTarget()
    {
        /* 공격범위에서만 적을 찾는 로직 추가 해야함 if문 이용 */
        
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

    bool IsTargetInRange()
    {
        // 타겟이 공격 범위 안에 있는지 확인
        return Vector3.Distance(transform.position, target.position) <= attackRange;
    }

    void AttackTarget()
    {
#if DEBUG_MODE
        // 여기서 타겟에게 데미지를 입히는 로직을 구현
        Debug.Log("Attacking target for " + playerController.damage + " damage.");
#endif
        HandleEnemy(target.gameObject);
    }

    protected void HandleEnemy(GameObject target)
    {
        // Pink, Mint, Purple, RedOctopus 또는 AttackEnemy 컴포넌트를 찾아서 가져옴
        var enemy = target.GetComponent<LongRange>() ?? target.GetComponent<FastSpeed>() ?? target.GetComponent<PurpleOctopus>() 
        ?? target.GetComponent<RedOctopus>() ?? target.GetComponent<AttackEnemy>();

        // enemy가 null이 아니면 즉, 해당 컴포넌트가 존재하면
        if (enemy != null)
        {
            // 적의 타입을 로그에 출력
            Debug.Log($"{enemy.GetType().Name} 컴포넌트가 존재합니다.");
            
            // Player에서 MageBasicSkillEffect 컴포넌트를 가져옴
            // 부모가 자식의 함수를 참조하는게 되네?? -> 한번 궁금해서 코드 작성해봄
            // mageBasicSkillEffect = gameObject.GetComponent<MageBasicSkillEffect>();

            // 적에게 데미지를 주고, 로그에 남김
            enemy.TakeDamage(playerController.damage);
            Debug.Log($"Basic Attack Hit!! {enemy.GetType().Name} Hp: {enemy.hp}");

            // MageBasicSkillEffect가 null이 아닐 경우 BasicSkill 메서드를 호출
            // mageBasicSkillEffect.BasicSkill();

            // 공격이 끝났으므로 발사체 오브젝트를 파괴
            Destroy(gameObject);
        }
    }
}
