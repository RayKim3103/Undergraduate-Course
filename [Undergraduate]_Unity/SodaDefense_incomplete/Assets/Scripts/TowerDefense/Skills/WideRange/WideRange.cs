// #define DEBUG_MODE
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WideRange : MonoBehaviour, ISkill
{
    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private BaseCharacter playerController;

    /* WideRange 정보 */
    private float extraExplosionDamage; // 범위 데미지

    [SerializeField]
    private float damageDecreaseAmount = 0.5f;

    [SerializeField]
    private float explosionRadius = 3f; // 범위 데미지 반경
    private float baseExplosionRadius = 3f; // original 범위 데미지 반경

    [SerializeField]
    public GameObject explosionPrefab; // 범위 Prefab

    // private int debugCount = 1;

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        playerController = player.GetComponent<BaseCharacter>();
    }
    
    private void Start() 
    {
        // Player의 능력치에 비례하여 Explosion의 damage 설정
        extraExplosionDamage = playerController.damage * damageDecreaseAmount;    
    }
    public void Activate()
    {
        // 인터페이스를 위한 공란 함수 정의
    }
    
    public void SkillEffectActivate(Transform target)
    {
        Vector3 explosionCenter = target.position;

        // OverlapSphere: 지정된 위치와 반경 내의 모든 콜라이더를 찾는 함수
        Collider2D[] hitColliders = Physics2D.OverlapCircleAll(explosionCenter, explosionRadius);
#if DEBUG_MODE
        Debug.Log(" ************* Explosion at: " + explosionCenter + ", found: " + hitColliders.Length + " colliders." + " Radius: " + explosionRadius + " ************* ");
#endif
        foreach (Collider2D hitCollider in hitColliders)
        {
            // 모든 콜라이더에서 foreach문을 사용해 각각에게 데미지를 줌
            if (hitCollider.CompareTag("Enemy"))
            {
                // Debug.Log(" Hit Collider" + debugCount + ": " + hitCollider.CompareTag("Enemy"));
                AttackEnemy enemy = hitCollider.GetComponent<AttackEnemy>();
                if (enemy != null)
                {
                    enemy.TakeDamage(extraExplosionDamage);
                    // debugCount++;
#if DEBUG_MODE
                    Debug.Log("WideRange Damaging enemy: " + enemy.gameObject.name + " / Damaged, Enemy HP: " + enemy.hp);
#endif
                }
            }
        }
        Instantiate(explosionPrefab, target.position, target.rotation);
    }

    public void LevelUp(float increaseAmount)
    {
        // 0렙때, 0.2f -> 1렙때, 0.4f
        this.damageDecreaseAmount += increaseAmount;
        this.explosionRadius = baseExplosionRadius + baseExplosionRadius * increaseAmount;
    }
}
