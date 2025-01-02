using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CounterAttack : MonoBehaviour, ISkill
{
    public GameObject damageAreaPrefab; // 데미지 영역을 나타낼 프리팹
    private bool isPrefabSpawned = false;
    public float damageAccumulationTime = 5f; // 데미지 축적 시간
    public float damageRadius = 2f; // 데미지를 줄 범위
    public float baseDamageRadius = 2f; // 데미지를 줄 범위
    public float accumulatedDamage = 0f; // 축적된 데미지

    [SerializeField]
    private float timer = 0f; // 시간 타이머

    /* CounterAttack 정보를 담을 변수 */
    public bool counterAttackActive = false;

    public void Activate()
    {
        if (counterAttackActive == false)
        {
            // baseCharacter 코드에서 사용할 변수, baseCharacter의 TakeDamage()에서 사용
            counterAttackActive = true;
        }

        // 타이머 업데이트
        timer += Time.deltaTime;

        // 타이머가 damageAccumulationTime을 초과하면 데미지를 주고 타이머 리셋
        if (timer >= damageAccumulationTime)
        {
            DealDamageToEnemies();
            timer = 0f; // 타이머 초기화
            accumulatedDamage = 0f; // 축적된 데미지 초기화
        }
    
    }

    public void SkillEffectActivate(Transform target)
    {
        // 인터페이스를 위한 공란 함수 정의
    }

    private void DealDamageToEnemies()
    {
        // 지정된 반경 내의 모든 적에게 축적된 데미지 적용
        Collider2D[] hitEnemies = Physics2D.OverlapCircleAll(transform.position, damageRadius);
        
        foreach (Collider2D enemy in hitEnemies)
        {
            if (enemy.CompareTag("Enemy"))
            {
                if (accumulatedDamage != 0f)
                {
                    // 적에게 데미지 주기 (적 스크립트의 TakeDamage 함수 호출)
                    enemy.GetComponent<AttackEnemy>().TakeDamage(accumulatedDamage);
                    Debug.Log("Enemy: " + enemy + "CounterAttack Damge: " + accumulatedDamage);

                    // 프리팹을 생성하고 damageRadius에 맞춰 크기를 조절
                    if(isPrefabSpawned == false)
                    {
                        isPrefabSpawned = true;
                        GameObject damageArea = Instantiate(damageAreaPrefab, transform.position, Quaternion.identity);
                        damageArea.transform.localScale = new Vector3(damageRadius, damageRadius, 0);
                        Destroy(damageArea, 0.3f); // 바꿀일 없을 것 같아서 하드코딩함 
                    }
                    // damageArea.transform.parent = transform; // 프리팹을 SKillControl 오브젝트에 자식으로 설정
                }
            }
        }
        isPrefabSpawned = false;
    }


    public void LevelUp(float increaseAmount)
    {
        damageRadius = baseDamageRadius + increaseAmount;
    }

    private void OnDrawGizmosSelected()
    {
        // 데미지 범위 시각화
        Gizmos.color = new Color(1, 1, 1);;
        Gizmos.DrawWireSphere(transform.position, damageRadius);
    }
}
