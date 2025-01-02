using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnemyProjectile : MonoBehaviour
{
    // 투사체의 이동 속도
    public float speed = 10f;

    // 적의 공격렬
    private float projectileDamage;

    // 목표 적의 Transform
    private Transform targetTransform;
    private Vector3 direction;

    private void Start() 
    {
        // 1. 목표를 향한 방향 벡터 계산 (목표 위치 - 현재 위치), 논타겟팅으로 설정됨 (피하기 가능)
        direction = (targetTransform.position - transform.position).normalized;
    }

    // 매 프레임마다 호출되는 Unity 메소드
    private void Update()
    {
        // 목표가 없으면 투사체를 제거
        if (targetTransform == null)
        {
            Debug.Log("projectile Destroyed!");
            Destroy(gameObject);
            return;
        }

        // // 2. 목표를 향한 방향 벡터 계산 (목표 위치 - 현재 위치), 타겟팅으로 설정됨 (유도탄 공격)
        // direction = (targetTransform.position - transform.position).normalized;
        
        // 적을향하도록 회전
        float angle = Mathf.Atan2(direction.y, direction.x) * Mathf.Rad2Deg;
        transform.rotation = Quaternion.Euler(new Vector3(0, 0, angle));

        // 투사체를 목표 방향으로 이동
        transform.position += direction * speed * Time.deltaTime;

        // 목표와의 거리가 일정 이하로 가까워지면 목표에 도달했다고 간주
        if (Vector3.Distance(transform.position, targetTransform.position) < 0.5f)
        {
            OnHitTarget(); // 목표에 도달하면 OnHitTarget 메소드 호출
        }
    }

    // 목표를 설정하는 메소드
    /* 코딩공부: 참조 타입 vs 값 타입 */
/*
    참조 타입: 변수가 데이터의 참조(주소)를 저장. 동일한 객체를 여러 변수가 참조할 수 있으며, 변경이 반영됨.  ex) Transform    사용예: target.transform
    값 타입: 변수가 데이터를 직접 저장. 데이터를 다른 변수에 할당하면 복사본이 생성되며, 변경이 반영되지 않음. ex) Position
*/
    public void SetTargetwithDamage(GameObject target, float projectileDamage)
    {
        // 입력된 GameObject의 Transform을 타겟으로 설정
        targetTransform = target.transform;
#if DEBUG_MODE
        Debug.Log("SetTarget 결과: " + target + " 의 Transform");
#endif
        this.projectileDamage = projectileDamage;
#if DEBUG_MODE
        Debug.Log("SetTarget 결과: " + this.projectileDamage + " 의 공격력");
#endif
    }

    // 목표에 도달했을 때 호출되는 메소드
    void OnHitTarget()
    {
        // 목표 적의 BaseCharacter 컴포넌트를 가져옴
        BaseCharacter player = targetTransform.GetComponent<BaseCharacter>();
        AttackEnemy enemy = targetTransform.GetComponent<AttackEnemy>();
#if DEBUG_MODE
        Debug.Log("OnHitTarget의 player 선정: " + player);
#endif

        // 목표 적이 BaseCharacter 컴포넌트를 가지고 있다면
        // if(player.isDied)
        // {
        //     targetTransform.GetComponent<NexusHealth>().TakeDamage(projectileDamage);
        // }
        if (player != null)
        {
            // 적에게 데미지를 입힘
            player.TakeDamage(projectileDamage);
        }
        else if (enemy != null)
        {
            enemy.GetComponent<AttackEnemy>().TakeDamage(projectileDamage);
        }
        // 아닐 경우 목표가 Nexus임
        else
        {
            targetTransform.GetComponent<NexusHealth>().TakeDamage(projectileDamage);
        }

        // 투사체 제거
        Destroy(gameObject);
    }
}
