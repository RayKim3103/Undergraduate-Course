using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

public class AttackEnemy : MonoBehaviour
{
    /* Enemy들의 anim 또는 sprite */
    public RuntimeAnimatorController[] animCon;
    // public Sprite[] sprites;
    private int offSet = 0;

    /* Enemy 공격력, 범위 */
    [SerializeField]
    protected float damage = 1.0f;

    public float attackRange = 1.0f;
    public float nexusAttackRange = 5.0f; // 넥서스 크기로 인해 nexusAttackRange 변수 생성

    [SerializeField]
    protected float attackCooldown = 1.0f; // 공격 쿨타임
    protected float lastattackTime; // 마지막 공격 시간
    
    /* 원거리 Enemy 변수 */
    public bool isLongRange = false;
    protected float multiplier = 2.0f;
    protected float decreaseMultiplier = 0.5f;

    [SerializeField]
    protected GameObject projectilePrefab;

    /* Enemy 체력 */
    public float hp = 1.0f;
    protected float maxHp = 1.0f;

    /* Enemy가 공격할 대상 */
    protected GameObject target;

    /* Character 경험치 및 LevelUp 관련 변수 */
    // public string basicAttackTag = "BasicAttack"; // BasicAttack 오브젝트의 태그
    protected int coinIncrease = 1;
    public float randomNumberDiamonds = 0.05f; // 5%

    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    protected float hardness;
    protected GameObject player;
    protected BaseCharacter playerController;

    // /* BasicSkill 담을 변수 */
    // private MageBasicSkillEffect basicSkillEffect;

    protected virtual void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        playerController = player.GetComponent<BaseCharacter>();
        hardness = playerController.levelHardness;
        // basicSkillEffect = GetComponent<MageBasicSkillEffect>();
    }
    
    private void Start() 
    {
        target = gameObject.GetComponent<EnemyMovement>().target.gameObject;

        // SpawnData에서 관리하는 것으로 변경
        // if(isLongRange == true)
        // {
        //     // 원거리이므로, 사거리 증가
        //     attackRange = attackRange * multiplier; 
        // }
    }

    // Update is called once per frame
    protected virtual void Update()
    {
        Target();
        if (target == gameObject.GetComponent<EnemyMovement>().nexusTransform.gameObject)
        {
            if (Vector3.Distance(transform.position, target.transform.position) < nexusAttackRange && Time.time >= lastattackTime + attackCooldown)
            {
                AttackTarget();
                lastattackTime = Time.time;
            }
        }
        else if(target != null)
        {
            if (Vector3.Distance(transform.position, target.transform.position) < attackRange && Time.time >= lastattackTime + attackCooldown)
            {
                AttackTarget();
                lastattackTime = Time.time;
            }
        }
    }

    protected virtual void Target()
    {
        if(gameObject.GetComponent<EnemyMovement>().target != null)
        {
            target = gameObject.GetComponent<EnemyMovement>().target.gameObject;
        }
    }

    protected virtual void AttackTarget()
    {
        if(isLongRange == false)
        {
            // // Player가 죽었을 때, Nexus로 타겟 지정
            // if(playerController.isDied)
            // {
            //     target = gameObject.GetComponent<EnemyMovement>().nexusTransform.gameObject;
            //     target.GetComponent<NexusHealth>().TakeDamage(damage);
            //     Debug.Log("Enemy Target NEXUS / PlayerDied / Nexus HP: " + target.GetComponent<NexusHealth>().health);
            // }
            if (target == player)
            {
                player.GetComponent<BaseCharacter>().TakeDamage(damage);
            }
            else if (target == gameObject.GetComponent<EnemyMovement>().nexusTransform.gameObject)
            {
                target.GetComponent<NexusHealth>().TakeDamage(damage);
                // Debug.Log("Enemy Target NEXUS / Nexus HP: " + target.GetComponent<NexusHealth>().health);
            }
            // Mage의 Confuse 스킬 등의 경우
            else
            {
                if(target.GetComponent<AttackEnemy>() != null)
                {
                    target.GetComponent<AttackEnemy>().TakeDamage(damage);
                    Debug.Log("Confused!! Attack Other Enemy!! / Damage: " + damage);
                }
            }
        }
        // 원거리 일 때
        else
        {
            ProjectileShoot();
        }
    }
        

    public virtual void TakeDamage(float damage)
    {   
        hp -= damage;
        if (hp <= 0)
        {
            playerController.coin += coinIncrease + (int)Random.Range(-1,1);

            // 적은 확률로 diamonds 드롭
            if (Random.value <= randomNumberDiamonds)
            {
                playerController.diamonds += (int)Random.Range(1, 10);
                Debug.Log("Get Diamonds NOW diamonds: " + playerController.diamonds);
            }
            
#if DEBUG_MODE
            // Debug 용 출력문
            Debug.Log("Player Coin Up!! / Current Coin: " + playerController.coin);
#endif
            Die();
        }
    }

    // 적이 공격을 발사할 때 호출되는 메소드
    // override한 Update에서 Target()을 하였기에 base.Update()를 안해도 target설정 됨
    protected void ProjectileShoot()
    {
        // 투사체 프리팹을 적의 위치에 인스턴스화
        GameObject projectile = Instantiate(projectilePrefab, transform.position, Quaternion.identity);

        // 인스턴스화된 투사체의 EnemyProjectile 스크립트를 가져옴
        EnemyProjectile pinckOctopusProjectile = projectile.GetComponent<EnemyProjectile>();

        // 투사체의 타겟을 설정
        pinckOctopusProjectile.SetTargetwithDamage(target, damage); 
    }

    public void Die()
    {
        // 적이 죽었을 때의 로직을 구현합니다.
        Debug.Log("Enemy died.");
        // basicSkillEffect.IceBreak();

        Destroy(gameObject);
    }

    public void Init(SpawnData data, int roundLevel, int index)
    {
        // 애니메이터 컨트롤러를 할당 -> 애니메이션 만들어지면 나중에 처리해야 함
        // anim.runtimeAnimatorController = animCon[data.spriteType];
        gameObject.GetComponent<SpriteRenderer>().sprite = data.sprites[index];
        gameObject.GetComponent<EnemyMovement>().speed = data.speed[index];
        gameObject.GetComponent<EnemyMovement>().originalSpeed = data.speed[index];
        hp = data.hp[index];
        maxHp = data.hp[index];
        damage = data.damage[index];
        attackRange = data.attackRange[index];
        isLongRange = data.isLong[index];
        coinIncrease = data.coinIncrease[index];

        // 캡슐 콜라이더 설정
        CapsuleCollider2D capsuleCollider = gameObject.GetComponent<CapsuleCollider2D>();
        if (capsuleCollider != null)
        {
            // 스프라이트의 크기에 맞게 캡슐 콜라이더의 크기와 오프셋을 설정
            Vector2 spriteSize = gameObject.GetComponent<SpriteRenderer>().sprite.bounds.size;
            capsuleCollider.size = spriteSize;
            capsuleCollider.offset = Vector2.zero;
        }
    }

    /* 디버깅용: 적의 기본 공격범위를 표시 */
    protected virtual void OnDrawGizmos()
    {
        Gizmos.color = Color.red;
        Gizmos.DrawWireSphere(transform.position, attackRange);
        Gizmos.color = new Color(0, 1, 0, 1);
        Gizmos.DrawWireSphere(transform.position, nexusAttackRange);
    }
}

    

    
