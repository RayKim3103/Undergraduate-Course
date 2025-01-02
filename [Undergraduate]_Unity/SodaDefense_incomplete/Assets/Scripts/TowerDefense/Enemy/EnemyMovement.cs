#define DEBUG_MODE
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnemyMovement : MonoBehaviour
{
    /* Enemy 탐지 범위 (Player 탐지) */
    public float detectionRange = 5.0f;
    public float originalSpeed = 1.0f; // 적의 원래 속도 (느려지기 전)
    public float speed = 1.0f; // 적의 속도

    /* Slow를 위한 변수들 */
    public bool isSpeedReduced = false;
    private float reducedSpeedMultiplier = 0f;
    public float slowMaintainanceTime = 3.0f; // BasicSlow 지속시간

    /* 쫓아갈 대상을 위한 변수 */
    public string nexusTag = "Nexus"; // Nexus 오브젝트의 태그
    public string playerTag = "Player"; // Player 오브젝트의 태그
    public Transform nexusTransform;
    public Transform playerTransform;
    public Transform target;
    private AttackEnemy attackEnemy;
    private Rigidbody2D rb;
    //private Vector3 direction;
    private Vector3 originalDirection;
    private Vector3 moveDirection;
    private bool isChangingDirection = false;
    private int wallCollide = 0;
    public int maxWallCollide = 2;

    private void Awake()
    {
        // speed = gameObject.GetComponent<AttackEnemy>().speed;

        rb = GetComponent<Rigidbody2D>(); // Rigidbody2D 컴포넌트를 가져옴
        GameObject nexusObject = GameObject.FindWithTag(nexusTag);
        nexusTransform = nexusObject.transform;
        GameObject playerGameObject = GameObject.FindWithTag(playerTag);
        playerTransform = playerGameObject.transform;
        attackEnemy = gameObject.GetComponent<AttackEnemy>();
        target = nexusTransform; // 초기 타겟은 NEXUS
    }

    void Update()
    {
        float distanceToPlayer = Vector3.Distance(transform.position, playerTransform.position);

        // Mage의 confuse가 target을 바꿔버릴 수 있기에 제한 조건을 달음
        if(target == playerTransform || target == nexusTransform)
        {
            // distanceToPlayer가 detectionRange 이하일 경우 target을 player로 설정, 그렇지 않으면 target을 nexus로 설정
            target = (distanceToPlayer <= detectionRange) ? playerTransform : nexusTransform;
        }

        // Player가 죽었을 때, Player의 거리를 무한대로 해서 Player가 Target이 되지 않도록
        if(playerTransform.GetComponent<BaseCharacter>().isDied)
        {
            target = nexusTransform;
            MoveTowardsTarget();
            // // Debug.Log("Enemy moves to Nexus, since Player Died");
            // distanceToPlayer = Mathf.Infinity;
        }
        // target이 존재하면 적 이동
        else if(target != null)
        {
            MoveTowardsTarget();
        }
        else
        { 
            // 플레이어가 살아있을 때, target이 존재하지 않으면, null연산자 사용, Confuse 효과 다시 적용
            target = playerTransform.GetComponentInChildren<Confuse>()?.FindTarget(gameObject.transform);   
        }
    }

    private void OnTriggerExit2D(Collider2D other) // OnTriggerEnter2D
    {
        if (other.gameObject.CompareTag("Wall") && !isChangingDirection) //  && wallCollide < maxWallCollide
        {
            StartCoroutine(ChangeDirectionTemporarily());
            // wallCollide += 1;
#if DEBUG_MODE            
            Debug.Log("Enemy & Wall collide");
#endif
        }
        else if (other.gameObject.CompareTag("Wall") && !isChangingDirection)
        {
            ChangeRandomDirectionTemporarily();
            wallCollide = -2; // 3번의 랜덤 방향바꾸기 기회 줌
#if DEBUG_MODE            
            Debug.Log("Enemy & Wall collide / Moving Randomly");
#endif
        }
    }

    private IEnumerator ChangeDirectionTemporarily()
    {
        isChangingDirection = true;

        // 충돌 시 반대 방향으로 이동
        moveDirection = -originalDirection;

        // 0.5초 동안 대기
        yield return new WaitForSeconds(0.5f);

        // 원래 방향으로 복귀
        moveDirection = originalDirection;
        isChangingDirection = false;
    }

    private IEnumerator ChangeRandomDirectionTemporarily()
    {
        isChangingDirection = true;

        // 충돌 시 반대 방향으로 이동
        moveDirection = GetRandomDirection();

        // 0.5초 동안 대기
        yield return new WaitForSeconds(0.5f);

        // 원래 방향으로 복귀
        moveDirection = originalDirection;
        isChangingDirection = false;
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

    void MoveTowardsTarget()
    {
        if(isChangingDirection == false)
        {
            moveDirection = (target.position - transform.position).normalized;
            originalDirection = moveDirection;
        }
        // target에 너무 가까이 가지 않도록 설정
        if(Vector3.Distance(transform.position, target.position) >= attackEnemy.attackRange * 0.8f)
            transform.position += moveDirection * speed * Time.deltaTime;
    }

    public void ReduceSpeedTemporarily()
    {
        if (!isSpeedReduced)
        {
            StartCoroutine(ReduceSpeedCoroutine());
        }
    }

    private IEnumerator ReduceSpeedCoroutine()
    {
        isSpeedReduced = true;
        // 속도를 감소시킴
        speed = originalSpeed * reducedSpeedMultiplier;
        if(reducedSpeedMultiplier == 0f)
        {
            gameObject.GetComponent<Rigidbody2D>().constraints 
                = RigidbodyConstraints2D.FreezePositionX | RigidbodyConstraints2D.FreezePositionY | RigidbodyConstraints2D.FreezeRotation;
        }

        // 3초 동안 유지
        yield return new WaitForSeconds(slowMaintainanceTime);

        // 원래 속도로 복귀
        speed = originalSpeed;
        gameObject.GetComponent<Rigidbody2D>().constraints =  RigidbodyConstraints2D.FreezeRotation;
        isSpeedReduced = false;
    }

    /* Tower의  ReduceSpeed에 사용*/
    public void ReduceSpeedTemporarily(float slowFactor, float duration)
    {
        if (!isSpeedReduced)
        {
            StartCoroutine(ReduceSpeedCoroutine(slowFactor, duration));
        }
    }

    private IEnumerator ReduceSpeedCoroutine(float slowFactor, float duration)
    {
        // 속도를 감소시킴
        isSpeedReduced = true;
        speed = originalSpeed * slowFactor;
        if(slowFactor == 0f)
        {
            gameObject.GetComponent<Rigidbody2D>().constraints 
                = RigidbodyConstraints2D.FreezePositionX | RigidbodyConstraints2D.FreezePositionY | RigidbodyConstraints2D.FreezeRotation;
        }

        // 주어진 지속 시간 동안 유지
        yield return new WaitForSeconds(duration);

        // 원래 속도로 복귀
        speed = originalSpeed;
        gameObject.GetComponent<Rigidbody2D>().constraints =  RigidbodyConstraints2D.FreezeRotation;
        isSpeedReduced = false;
    }
}
