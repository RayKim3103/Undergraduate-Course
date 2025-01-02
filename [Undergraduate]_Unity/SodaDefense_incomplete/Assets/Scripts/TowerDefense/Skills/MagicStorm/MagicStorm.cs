using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MagicStorm : MonoBehaviour
{
    private float lastMagicStormTime; // 마지막 설치 시간
    public float damageInterval = 1f; // 데미지 간격 (초)

    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player; // Player 오브젝트를 참조할 변수
    private BaseCharacter playerController; // Player의 MageController 스크립트를 참조할 변수

    private List<GameObject> enemiesInRange = new List<GameObject>(); // 범위 내의 적 오브젝트들을 담을 리스트

    private InstantiateMagicStorm instantiateMagicStorm;

    private void Awake() 
    {
        // 태그를 이용해 Player 오브젝트를 찾고, MageController 스크립트를 가져옴
        player = GameObject.FindWithTag(playerTag);
        playerController = player.GetComponent<BaseCharacter>(); 
        instantiateMagicStorm = player.GetComponentInChildren<InstantiateMagicStorm>();
    }

    void Start()
    {
        lastMagicStormTime = Time.time;
        // DamageEnemies 코루틴 시작
        StartCoroutine(DamageEnemies());
    }

    private void Update() 
    {
        // Player의 공격력에 감소 비율을 곱하여 attackDamage 설정
        instantiateMagicStorm.attackDamage = playerController.damage * instantiateMagicStorm.attackDamageDecrease;

        // 지속시간 초과시 Destroy
        if(Time.time >= lastMagicStormTime + instantiateMagicStorm.magicStormMaintainanceTime)
        {
            Destroy(gameObject);
        }
    }

    void OnTriggerEnter2D(Collider2D other)
    {
        // 트리거 범위에 들어온 오브젝트가 "Enemy" 태그를 가진 경우 리스트에 추가
        if (other.CompareTag("Enemy"))
        {
            enemiesInRange.Add(other.gameObject);
        }
    }

    void OnTriggerExit2D(Collider2D other)
    {
        // 트리거 범위를 벗어난 오브젝트가 "Enemy" 태그를 가진 경우 리스트에서 제거
        if (other.CompareTag("Enemy"))
        {
            enemiesInRange.Remove(other.gameObject);
        }
    }

    IEnumerator DamageEnemies()
    {
        while (true)
        {
            // damageInterval 만큼 대기
            yield return new WaitForSeconds(damageInterval);

            // 범위 내의 적 리스트의 복사본을 만듦
            // 복사본을 안 만들시, List가 열거(enumerate)되고 있는 동안 해당 List가 수정(추가 또는 제거)되기에 에러 발생
            List<GameObject> enemiesToDamage = new List<GameObject>(enemiesInRange);

            // 복사본을 사용하여 데미지 줌
            foreach (GameObject enemy in enemiesToDamage)
            {
                if (enemy != null)
                {
                    // 적의 AttackEnemy 스크립트의 TakeDamage 메소드를 호출하여 데미지 줌
                    enemy.GetComponent<AttackEnemy>().TakeDamage(instantiateMagicStorm.attackDamage);
#if DEBUG_MODE
                    Debug.Log("MagicStorm Damged Enemy, Enemy HP: " + enemy.GetComponent<AttackEnemy>().hp);
#endif
                }
            }
        }
    }
}
