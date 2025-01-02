#define DEBUG_MODE

using System.Collections;
using UnityEngine;
using UnityEngine.InputSystem;

public class BaseCharacter : MonoBehaviour
{
    /* Player 상태 변수 */
    [HideInInspector] public float statPoints = 0f;
    public bool isDied = false;
    public float hp = 10f;
    public float maxHp = 10f;
    public float moveSpeed = 2.0f;
    public float baseMoveSpeed = 2.0f;
    public int coin = 0;
    public int diamonds = 0;

    [SerializeField]
    private Vector3 moveDirection;
    public Vector3 lastMoveDirection = Vector3.left;
    public LayerMask targetLayer;
    public int targetLayerIndex = 6;

    /* Player 공격 변수 */
    public float damage = 1f;
    public float attackSpeed = 3.0f;
    public float fireCooldown = 1.0f;
    public float maxDistance = 5.0f;
    public float baseMaxDistance = 5.0f;
    public GameObject attackPrefab;
    protected float lastFireTime;
    
    /* Player 움직임, 적 scan */
    protected SpriteRenderer spriteRenderer;
    protected GameObject clone;
    protected RaycastHit2D[] targets;
    protected Transform nearestTarget;

    [SerializeField]
    protected Vector2 inputVec;

    // level업을 하기 위한 어려운 정도
    public float levelHardness = 1.0f;

    protected virtual void Awake()
    {
        gameObject.transform.localScale = Vector3.zero;
        spriteRenderer = GetComponent<SpriteRenderer>();

        // 레이어 이름을 인덱스로 변환, RayCast 시 레이어 사용
        // 게임 오브젝트의 레이어를 설정 / 2의 6승: 6번째 index
        targetLayer = 1 << targetLayerIndex;
    }

    protected virtual void Update()
    {
        if (isDied)
        {
            HandleMovement(); // 죽어도 맵을 보게 하려고 추가
            return;
        } 

        HandleMovement();
        HandleAttack();
        InGameUIManager.instance.ShowStats();
    }

    protected virtual void HandleMovement()
    {
        moveDirection = new Vector3(inputVec.x, inputVec.y, 0);
        if (moveDirection != Vector3.zero)
        {
            lastMoveDirection = moveDirection.normalized;
            spriteRenderer.flipX = inputVec.x > 0;
        }
        transform.position += moveDirection.normalized * moveSpeed * Time.deltaTime;
    }

    private void OnTriggerExit2D(Collider2D other) // OnTriggerEnter2D
    {
        if (other.gameObject != null && other.gameObject.activeInHierarchy)
        {
            if(other.gameObject.CompareTag("Wall"))
            {
                if (GameObject.Find("Boss") != null)
                {
                    transform.position -= 30.0f * moveDirection.normalized * moveSpeed * Time.deltaTime;
                    Debug.Log("벽과 플레이어 충돌!!");
                }
                else{return;}
            }
        }
    }

    protected virtual void HandleAttack()
    {
        targets = Physics2D.CircleCastAll(transform.position, maxDistance, Vector2.zero, 0, targetLayer);
        nearestTarget = GetNearestTarget();

        if (nearestTarget && Time.time >= lastFireTime + fireCooldown)
        {
            Vector3 direction = (nearestTarget.position - transform.position).normalized;
            clone = Instantiate(attackPrefab, transform.position, Quaternion.Euler(new Vector3(0f, 0f, RotateToDirection(direction))));
            clone.name = "BasicAttack";
            clone.GetComponent<BasicAttack>().Setup(direction, attackSpeed, damage);
            // Debug.Log("Nearest Target: " + nearestTarget +" Direction: " + direction);
            lastFireTime = Time.time;
        }
    }

    protected virtual Transform GetNearestTarget()
    {
        Transform nearest = null;
        float minDistance = Mathf.Infinity;

        foreach (RaycastHit2D hit in targets)
        {
            float distance = Vector3.Distance(transform.position, hit.transform.position);
            if (distance < minDistance)
            {
                minDistance = distance;
                nearest = hit.transform;
            }
        }

        return nearest;
    }

    public float RotateToDirection(Vector3 direction)
    {
        if (direction == Vector3.zero) return 0f;
        float angle = Mathf.Atan2(direction.y, direction.x) * Mathf.Rad2Deg + 180f;
        return angle;
    }

    public virtual void TakeDamage(float damage)
    {
        if (isDied) return;

        if (gameObject.GetComponentInChildren<ManaShield>().manaShieldActive)
        {
            damage *= gameObject.GetComponentInChildren<ManaShield>().damageReduceAmount;
        }
        if (gameObject.GetComponentInChildren<CounterAttack>().counterAttackActive)
        {
            gameObject.GetComponentInChildren<CounterAttack>().accumulatedDamage += damage;
        }

        hp -= damage;
        Debug.Log("Player Current Hp: " + hp);

        if (hp <= 0)
        {
            Die();
        }
    }

    protected virtual void Die()
    {
        Debug.Log("Player died / gameObject: " + gameObject);
        StartCoroutine(HandleDeath(10f));
    }

    protected virtual IEnumerator HandleDeath(float delay)
    {
        isDied = true;
        InGameUIManager.instance.ShowStats(); // Hp가 0이하로 떨어진 것 보여줘야 함
        StartCoroutine(DisableMovementCoroutine(2.0f)); // 2초간은 못 움직임
        SetChildrenActive(false);
        Vector3 originalPosition = transform.position;
        transform.localScale = Vector3.zero;

        Debug.Log($"Waiting for {delay} seconds...");
        yield return new WaitForSeconds(delay);

        isDied = false;
        SetChildrenActive(true);
        transform.localScale = Vector3.one;
        transform.position = originalPosition;
        hp = maxHp;
    }

    protected virtual void SetChildrenActive(bool isActive)
    {
        Transform[] children = GetComponentsInChildren<Transform>(true);
        foreach (Transform child in children)
        {
            if (child == children[2]) // Ensuring we don't modify the parent object itself
            {
                child.gameObject.SetActive(isActive);
            }
        }
    }

    private IEnumerator DisableMovementCoroutine(float seconds)
    {
        float originalMoveSpeed = moveSpeed;
        moveSpeed = 0f;
        yield return new WaitForSeconds(seconds);
        moveSpeed = originalMoveSpeed;
    }

    void OnMove(InputValue value)
    {
        inputVec = value.Get<Vector2>();
    }
}


// // #define DEBUG_MODE

// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine;
// using UnityEngine.InputSystem;

// public class BaseCharacter : MonoBehaviour
// {
//     /* 캐릭터 level, HP, 공격력, EXP, StatPoints */
//     [HideInInspector]
//     // public float level = 1f;
//     public float statPoints = 0f;
//     public float damage = 1f;
//     public float attackSpeed = 3.0f;
//     public bool isDied = false;
//     public float maxDistance = 5.0f;
//     public float baseMaxDistance = 5.0f;

//     [HideInInspector]
//     public float hp = 10f;
    
//     [HideInInspector]
//     public float maxHp = 10f;

//     [HideInInspector]
//     public float coin = 0.0f;

//     public GameObject attackPrefab;
//     protected GameObject clone;

//     [SerializeField]
//     // protected KeyCode keyCodeFire = KeyCode.Space; // space 누르면 Attack
//     public float fireCooldown = 1.0f; // 발사 쿨타임
//     protected float lastFireTime; // 마지막 발사 시간
//     public Vector3 lastMoveDirection = Vector3.left; // Basic Attack 방향 초기값

//     /* Enemy 검색을 위한 변수들 */
//     public RaycastHit2D[] targets;
//     public Transform nearestTarget;
//     // 레이어를 이용하여 적을 검색 할 예정
//     public LayerMask targetLayer;

//     // level업을 하기 위한 어려운 정도
//     public float levelHardness = 1.0f;
    
//     /* 캐릭터 이동속도, 방향 설정 */
//     public float moveSpeed = 2.0f;
//     public Vector2 inputVec;


//     /* 캐릭터 sprite 이미지 방향 설정*/
//     protected SpriteRenderer spriteRenderer;

//     protected virtual void Awake()
//     {
//         spriteRenderer = GetComponent<SpriteRenderer>();
//     }

//     protected virtual void Update()
//     {
//         UIManager.instance.ShowStats();
//         /******************** Player 움직임 제어 ********************/
//         float moveHorizontal = inputVec.x; // Input.GetAxisRaw("Horizontal");
//         float moveVertical = inputVec.y; // Input.GetAxisRaw("Vertical");
        
//         // Character의 움직임 제어 & Character의 속도를 일정하게 하기 위한 if문
//         transform.position += (moveHorizontal != 0 && moveVertical != 0)
//         ? new Vector3(moveHorizontal * 0.7f, moveVertical * 0.7f, 0) * moveSpeed * Time.deltaTime
//         : new Vector3(moveHorizontal, moveVertical, 0) * moveSpeed * Time.deltaTime;

//         // Character의 sprite 이미지 방향 제어
//         if (moveHorizontal != 0)
//         {
//             spriteRenderer.flipX = (moveHorizontal > 0) ? true : false;
//         }
//         /************************************************************/
        
//         if(moveHorizontal != 0 || moveVertical != 0)
//         {
//             lastMoveDirection = (moveHorizontal != 0 && moveVertical != 0) 
//                 ? new Vector3(moveHorizontal * 0.7f, moveVertical * 0.7f, 0) 
//                 : new Vector3(moveHorizontal, moveVertical, 0);
//         }

//         if(isDied == false)
//         {
//             // 원형의 RayCast를 실시하는 코드: CircleCastAll
//             // 파라미터: 1. 캐스팅 시작 위치, 2. 원의 Radius, 3. 캐스팅 방향, 4. 캐스팅 길이, 5. 대상 레이어
//             targets = Physics2D.CircleCastAll(transform.position, maxDistance, Vector2.zero, 0, targetLayer);
//             nearestTarget = GetNearest();
//             Fire();
//         }
//     }

//     // InputSystem의 함수
//     void OnMove(InputValue value)
//     {
//         inputVec = value.Get<Vector2>();
//     }

//     protected virtual Transform GetNearest()
//     {
//         Transform result = null;
//         float diff = 100; // 적당히 큰 거리

//         foreach (RaycastHit2D target in targets) 
//         {
//             Vector3 myPos = transform.position;
//             Vector3 targetPos = target.transform.position;
//             float curDiff = Vector3.Distance(myPos, targetPos);

//             if (curDiff < diff) 
//             {
//                 diff = curDiff;
//                 result = target.transform;
//             }
//         }

//         return result;
//     }

//     protected virtual void Fire()
//     {
//         if (!nearestTarget)
//             return;

//         // 적이 존재하는 위치
//         Vector3 targetPos = nearestTarget.position;
//         // 공격이 날라가는 방향 (normalize 필요)
//         Vector3 dir = targetPos - transform.position;
//         dir = dir.normalized;

//         if (Time.time >= lastFireTime + fireCooldown )
//         {
//             clone = Instantiate(attackPrefab, transform.position, Quaternion.Euler(new Vector3(0f, 0f, RotateToDirection(dir))));

//             clone.name = "BasicAttack";

//             // Attack의 방향 설정
//             clone.GetComponent<BasicAttack>().Setup(dir, attackSpeed, damage); // clone.GetComponent<MageBasicAttack>().Setup(lastMoveDirection);

//             // 마지막 발사 시간을 현재 시간으로 업데이트
//             lastFireTime = Time.time;
//         }
//     }

//     /****************** Basic Attack sprite의 방향 (Controller 코드에서 쓸 예정) ******************/
//     public virtual float RotateToDirection(Vector3 direction)
//     {
//         float angle = 0f;

//         // 입력된 방향이 있을 때만 회전 변경
//         if (direction != Vector3.zero)
//         {
//             // 방향 벡터의 각도를 계산 [arctan]하여 라디안을 도로 변환, 올바른 각도로 설정
//             angle = Mathf.Atan2(direction.y, direction.x) * Mathf.Rad2Deg;
//             // 각도 조정
//             angle += 180f;
//         }

//         return angle;
//     }
//     /*********************************************************************************************/

//     // Player가 데미지를 입음, (ManaShield의 피해감소를 적용)
//     public virtual void TakeDamage(float damage)
//     {
//         if(isDied == false)
//         {
//             if (gameObject.GetComponentInChildren<ManaShield>().manaShieldActive)
//             {
//                 damage *= gameObject.GetComponentInChildren<ManaShield>().damageReduceAmount;
//             }
//         }
        
//         hp -= damage;
//         Debug.Log("Player Current Hp: " + hp);
//         if (hp <= 0)
//         {
//             Die();
//         }
//     }
    
//     // 죽었을 때의 로직
//     protected virtual void Die()
//     {
//         Debug.Log("Player died / gameObject: " + gameObject);
//         StartCoroutine(DiedandWaitForSeconds(10f));
//     }

//     IEnumerator DiedandWaitForSeconds(float seconds)
//     {
//         isDied = true;

//         SetChildrenActive(false);
//         Vector3 locationDied = gameObject.transform.position;
//         gameObject.transform.localScale = new Vector3(0, 0);
        
//         // n초 동안 대기
//         Debug.Log("Waiting for " + seconds + " seconds...");
//         yield return new WaitForSeconds(seconds);

//         isDied = false;
//         SetChildrenActive(true);
//         gameObject.transform.localScale = new Vector3(1, 1);
//         gameObject.transform.position = locationDied;
//         hp = maxHp;
//     }

//     void SetChildrenActive(bool isActive)
//     {
//         // 자식 게임 오브젝트들의 활성 상태 변경
//         Transform[] children = GetComponentsInChildren<Transform>(true);
//         foreach (Transform child in children)
//         {
//             Debug.Log("Player Died / child: " + child);
//             if(child != children[2])
//             {
//                 continue;
//             }
//             child.gameObject.SetActive(isActive);
//         }
//     }

// }
