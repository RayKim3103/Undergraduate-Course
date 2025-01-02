using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SummonProjectile : MonoBehaviour
{
    // 투사체의 이동 속도
    public float speed = 10f;

    // 투사체가 적에게 입힐 피해량
    private float damage;

    // 목표 적의 Transform
    protected Transform target;

    // 매 프레임마다 호출되는 Unity 메소드
    void Update()
    {
        // 목표가 없으면 투사체를 제거
        if (target == null)
        {
            Destroy(gameObject);
            return;
        }

        // 목표를 향한 방향 벡터 계산 (목표 위치 - 현재 위치)
        Vector3 direction = (target.position - transform.position).normalized;
        
        // 적을 향하도록 회전
        float angle = Mathf.Atan2(direction.y, direction.x) * Mathf.Rad2Deg;
        transform.rotation = Quaternion.Euler(new Vector3(0, 0, angle-40));

        // 투사체를 목표 방향으로 이동
        transform.position += direction * speed * Time.deltaTime;

        // 목표와의 거리가 일정 이하로 가까워지면 목표에 도달했다고 간주
        if (Vector3.Distance(transform.position, target.position) < 0.5f)
        {
            OnHitTarget(); // 목표에 도달하면 OnHitTarget 메소드 호출
        }
    }

    // 목표 적을 설정하는 메소드
    public void SetTargetwithDamage(Transform target, float damage)
    {
        // 입력된 GameObject의 Transform을 타겟으로 설정
        this.target = target.transform;
        this.damage = damage;
    }

    // 목표에 도달했을 때 호출되는 메소드
    protected virtual void OnHitTarget()
    {
        // 목표 적의 AttackEnemy 컴포넌트를 가져옴
        AttackEnemy enemy = target.GetComponent<AttackEnemy>();

        // 목표 적이 AttackEnemy 컴포넌트를 가지고 있다면
        if (enemy != null)
        {
            // 적에게 데미지를 입힘
            enemy.TakeDamage(damage);
        }

        // 투사체 제거
        Destroy(gameObject);
    }
}
