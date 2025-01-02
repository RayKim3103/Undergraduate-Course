using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Confuse : MonoBehaviour, ISkill
{
    /* Confuse 변수 */
    private Transform originalTarget;
    private Transform target;
    
    [SerializeField]
    private float duration = 3.0f;
    private float baseDuration = 3.0f;

    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    protected GameObject player;

    protected virtual void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
    }

    public void Activate()
    {
        // 인터페이스를 위한 공란 함수 정의
    }

    public void SkillEffectActivate(Transform target)
    {
        // 코루틴 시작
        StartCoroutine(ConfuseForDuration(target, duration));
    }

    private IEnumerator ConfuseForDuration(Transform target, float duration)
    {
        EnemyMovement enemyMovement = target.GetComponent<EnemyMovement>();

        // target에 EnemyMovement 컴포넌트가 없으면, 함수 종료
        if (enemyMovement == null)
        {
            yield break;
        }

        // 기존 타겟 저장
        originalTarget = enemyMovement.target;

        // 새 타겟 설정
        enemyMovement.target = FindTarget(target);

        yield return new WaitForSeconds(duration);

        // 원래 타겟 복원
        enemyMovement.target = originalTarget;
    }

    public Transform FindTarget(Transform target)
    {
        // 적을 찾는 로직을 구현, 태그가 "Enemy"인 모든 적을 찾음
        GameObject[] enemies = GameObject.FindGameObjectsWithTag("Enemy");
        float closestDistance = Mathf.Infinity; // Distance 무한으로 초기화
        GameObject closestEnemy = null;         // 가까운 적은 없음으로 초기화

        foreach (GameObject enemy in enemies)
        {
            // Distance: 자기 자신과 target의 거리 반환
            float distance = Vector3.Distance(target.transform.position, enemy.transform.position);
            
            // foreach를 이용하여 자기자신 제외 가장 가까운 거리의 적을 저장
            if (distance < closestDistance && enemy != target.gameObject) // && distance != 0
            {
                Debug.Log("Confuse Target Array: " + enemy);
                closestDistance = distance;
                closestEnemy = enemy;
            }
        }

        // foreach문이 끝나고 closestEnemy가 null이 아니면, target 결정
        if (closestEnemy != null)
        {
            Debug.Log("Confuse Target: " + closestEnemy.transform);
            return closestEnemy.transform;
        }

        Debug.Log("Confuse Target: " + player.transform);
        return player.transform;
    }

    public void LevelUp(float increaseAmount)
    {
        // 0렙때, 0.2f -> 1렙때, 0.4f
        this.duration = baseDuration + baseDuration * increaseAmount;
        Debug.Log("Confuse Duration: " + duration);
    }
}
