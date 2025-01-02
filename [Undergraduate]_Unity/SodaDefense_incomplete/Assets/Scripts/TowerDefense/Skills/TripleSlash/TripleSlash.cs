using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TripleSlash : MonoBehaviour, ISkill
{
    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private BaseCharacter baseCharacter;

    /* Slash 정보를 담을 변수들 */
    public float damage; // 참격 데미지
    public float attackSpeed;

    /* Slash의 쿨타임 변수들 */
    public float fireCooldown = 3.0f; // 발사 쿨타임
    public float baseFireCooldown = 3.0f; // 발사 쿨타임
    protected float lastFireTime; // 마지막 발사 시간

    
    [SerializeField]
    private GameObject slashPrefab; // 참격 프리팹

    /* preFab 생성 위치 변수 */
    private CapsuleCollider2D capsuleCollider2D;

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        baseCharacter = player.GetComponent<BaseCharacter>();
        damage = baseCharacter.damage;
        attackSpeed = baseCharacter.attackSpeed;
        // Debug.Log("Triple Slash Damage: " + damage);
    }

    public void Activate()
    {
        if (Time.time >= lastFireTime + fireCooldown)
        {
            // CapsuleCollider2D에서 bounds를 가져와서 중앙 위치 계산
            capsuleCollider2D = GetComponentInParent<CapsuleCollider2D>();
            Bounds bounds = capsuleCollider2D.bounds;
            Vector3 centerPosition = bounds.center;

            // player의 현재 방향을 가져옴
            bool isFacingRight = player.transform.localScale.x > 0;

            // 각도 설정 (시계 방향으로 10시, 12시, 2시)
            float[] angles = isFacingRight ? new float[] { -30f, 0f, 30f } : new float[] { -150f, 180f, 150f };

            // 각 방향으로 참격 프리팹 인스턴스화 및 이동 구현
            foreach (float angle in angles)
            {
                Quaternion extraRotation = Quaternion.Euler(0, 0, angle);
                Vector3 direction = extraRotation * baseCharacter.lastMoveDirection; // gameObject.GetComponentInParent<BaseCharacter>()
                // Debug.Log("TripleSlash Direction: " + direction + "/ last MoveDirection: " + baseCharacter.lastMoveDirection);
                float rotation = baseCharacter.RotateToDirection(direction);
                GameObject clone = Instantiate(slashPrefab, centerPosition, Quaternion.Euler(new Vector3(0, 0, rotation)));
                // Attack의 방향 설정
                clone.GetComponent<BasicAttack>().isLongRange = true;
                // clone.GetComponent<BasicAttack>().attackRange *= 2; // 기본 attackRange의 2배
                clone.GetComponent<BasicAttack>().Setup(direction, attackSpeed, damage);
                
            }
            Debug.Log("Slash in three directions with " + damage + " damage each.");

            lastFireTime = Time.time; // 마지막 공격 시간을 현재 시간으로 업데이트
        }
    }
    public void SkillEffectActivate(Transform target)
    {
        // 인터페이스를 위한 공란 함수 정의
    }

    public void LevelUp(float increaseAmount)
    {
        this.fireCooldown = baseFireCooldown - baseFireCooldown * increaseAmount;
    }   
}
