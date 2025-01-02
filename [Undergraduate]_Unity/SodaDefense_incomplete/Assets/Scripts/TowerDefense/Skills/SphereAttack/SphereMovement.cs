using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SphereMovement : MonoBehaviour
{
    public GameObject player;  // 플레이어를 참조
    private BaseCharacter baseCharacter;
    private SphereAttack sphereAttack;
    // private float orbitRadius = 2.0f;  // 회전 반지름
    // private float orbitSpeed = 100.0f;  // 회전 속도
    private float angle;  // 현재 각도

    private void Awake() 
    {
        player = GameObject.FindGameObjectWithTag("Player");
        baseCharacter = player.GetComponent<BaseCharacter>();
        sphereAttack = player.GetComponentInChildren<SphereAttack>();
        // Debug.Log("Player: " + player + " / BaseCharacter: " + baseCharacter + " / shpereAttack: " + sphereAttack);
    }

    void Update()
    {
        if (baseCharacter.isDied == false && sphereAttack.isSphereActivate == true)
        {
            sphereAttack.damage = baseCharacter.damage * sphereAttack.damageDecreaseAmount;

            // 각도를 증가시켜 구체가 회전하도록 함
            angle += sphereAttack.orbitSpeed * Time.deltaTime;

            // 라디안으로 변환
            float rad = angle * Mathf.Deg2Rad;

            // 새로운 위치 계산
            float x = player.transform.position.x + Mathf.Cos(rad) * sphereAttack.orbitRadius;
            float y = player.transform.position.y + Mathf.Sin(rad) * sphereAttack.orbitRadius;

            // 구체의 위치를 업데이트
            transform.position = new Vector2(x, y);
        }
        else
        {
            sphereAttack.isSphereActivate = false;
            Destroy(gameObject);
        }

        // 아래 주석된 부분은 필요 X, Inactive일 때는, Update함수 작동 X, but, 참조하는 방법으로 Inactive일 때도 함수들 사용가능
        // if (baseCharacter.isDied == true)
        // {
        //     // Player가 죽었을 때는 Sphere가 이미 Destroy되어 있으므로
        //     sphereAttack.isLevelUp = false;
        // }
        if (sphereAttack.isLevelUp == true)
        {
            // Player가 살아 있을 때, Skill이 LevelUp을 했다는 뜻, Sphere을 Destroy하고 다시 생성해야 하기에, Sphere Destroy
            Destroy(gameObject);
            sphereAttack.isLevelUp = false;
        }
    }

    void OnTriggerEnter2D(Collider2D other)
    {
        // 충돌한 객체의 태그가 "Enemy"인지 확인
        if (other.CompareTag("Enemy"))
        {
            // Enemy 스크립트를 가져와 데미지를 줌
            AttackEnemy enemy = other.GetComponent<AttackEnemy>();
            if (enemy != null)
            {
                enemy.TakeDamage(sphereAttack.damage);
                Debug.Log("SphereAttack Hit Enemy!! / Damage: " + sphereAttack.damage);
            }
        }
    }

    public void AngleSetUp(float angle)
    {
        this.angle = angle;
    }
}
