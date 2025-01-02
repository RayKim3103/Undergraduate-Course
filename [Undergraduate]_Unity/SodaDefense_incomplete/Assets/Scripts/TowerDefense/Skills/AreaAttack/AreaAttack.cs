using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AreaAttack : MonoBehaviour, ISkill
{
    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private BaseCharacter baseCharacter;
    private SkillController skillController;

    /* Area Attack 변수 */
    public GameObject attackAreaPrefab;
    private GameObject attackArea;
    public bool isPrefabSpawned = false;

    [SerializeField]
    private float damagePerSecond; // 초당 입히는 데미지 (20%)
    private float decreaseAmount = 0.2f;

    [SerializeField]
    private float attackRadius = 3f; // 공격 반경
    [SerializeField]
    private float attackDuration = 5f; // 공격 지속 시간
    [SerializeField]
    private float attackInterval = 1f; // 공격 간격

    public bool isAttacking = false; // 공격 중인지 여부를 나타내는 플래그

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        baseCharacter = player.GetComponent<BaseCharacter>();
        skillController = GetComponentInParent<SkillController>();
        // Debug.Log("AreaAttack Start / baseCharacter: " + baseCharacter);
        damagePerSecond = decreaseAmount * baseCharacter.damage;
    }

    public void Activate()
    {
        // 공격 중이 아닌 경우, 이 조건 없으면 1초에 수십번 공격함
        if (!isAttacking)
        {
            StartCoroutine(ActivateAttack()); // 지역 공격 시작
        }
    }
    public void SkillEffectActivate(Transform target)
    {
        // 인터페이스를 위한 공란 함수 정의
    }


    public IEnumerator ActivateAttack()
    {
        isAttacking = true; // 공격 중 플래그 설정
        // Debug.Log("Enemy Area Attack is True!!");

        // 프리팹을 생성하고 attackRadius에 맞춰 크기를 조절
        if(isPrefabSpawned == false)
        {
            isPrefabSpawned = true;
            attackArea = Instantiate(attackAreaPrefab, transform.position, Quaternion.identity);
            // 프리팹 크기를 설정
            attackArea.transform.localScale = new Vector3(attackRadius, attackRadius, 0);
        }

        // 공격 반경 내의 적을 모두 감지
        Collider2D[] hitEnemies = Physics2D.OverlapCircleAll(transform.position, attackRadius);

        // 감지된 각 적에게 데미지를 입힘
        foreach (Collider2D enemy in hitEnemies)
        {
            if (enemy.GetComponent<AttackEnemy>() != null)
            {
                enemy.GetComponent<AttackEnemy>().TakeDamage(damagePerSecond);
                Debug.Log("Enemy Area Attack Damage: " + damagePerSecond);
            }
        }

        yield return new WaitForSeconds(attackInterval); // 공격 간격 대기        

        isAttacking = false; // 공격 중 플래그 해제
    }

    public void LevelUp(float damageIncreaseAmount)
    {
        // 0렙때, 0.2f -> 1렙때, 0.4f
        this.decreaseAmount += damageIncreaseAmount;
        damagePerSecond = decreaseAmount * baseCharacter.damage;
    }

    // Gizmos를 통해 공격 반경을 시각적으로 표시
    private void OnDrawGizmos()
    {
        if(skillController != null) // 게임종료시 에러메세지 뜨는 거 보기 안 좋아서 if문 넣음, 사실상 필요X
        {
            if(skillController.ArrayContainsValue(skillController.skills, (int)SkillSelection.SkillType.AreaAttack))
            {
                Gizmos.color = new Color(1, 0, 1);
                Gizmos.DrawWireSphere(transform.position, attackRadius);
            }
        }
    }
}
