// #define DEBUG_MODE

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/* PinkOctopus: 사거리 증가, 발사체 투사 */
public class LongRange : AttackEnemy
{
//     private float multiplier = 2.0f;
//     private float decreaseMultiplier = 0.5f;

//     [SerializeField]
//     private GameObject projectilePrefab;

//     protected override void Awake() 
//     {
//         base.Awake();

//         // 원거리이므로, 근거리보다 체력은 낮게 설정
//         hp = hp * decreaseMultiplier;
//         attackRange = attackRange * multiplier;
//     }

//     // Update is called once per frame
//     protected override void Update()
//     {
//         base.Update();
//     }

//     // Taget 또한 원본의 target의 참조 타입을 다루므로, 
//     // 매 프레임마다 값을 Update 하지 않아도 원본의 변화를 따라감
//     protected override void AttackTarget()
//     {
//         //Target();
//         ProjectileShoot();
//     }

//     public override void TakeDamage(float damage)
//     {
//         hp -= damage;
//         if (hp <= 0)
//         {
//             // 부모와 다른 부분, exp 주는 양 증가
//             playerController.exp += expIncrease;
// #if DEBUG_MODE
//             // Debug 용 출력문
//             Debug.Log("Player EXP Up!! / Current EXP: " + playerController.exp);
// #endif
//             Die();
//         }
//     }

    // // 적이 공격을 발사할 때 호출되는 메소드
    // // override한 Update에서 Target()을 하였기에 base.Update()를 안해도 target설정 됨
    // protected void ProjectileShoot()
    // {
    //     // 투사체 프리팹을 적의 위치에 인스턴스화
    //     GameObject projectile = Instantiate(projectilePrefab, transform.position, Quaternion.identity);

    //     // 인스턴스화된 투사체의 EnemyProjectile 스크립트를 가져옴
    //     EnemyProjectile pinckOctopusProjectile = projectile.GetComponent<EnemyProjectile>();

    //     // 투사체의 타겟을 설정
    //     pinckOctopusProjectile.SetTargetwithDamage(target, damage); 
    // }

    // public void LongRangeInit(SpawnData data)
    // {
    //     int index = Random.Range(0, data.spriteType.Length);
    //     // 애니메이터 컨트롤러를 할당 -> 애니메이션 만들어지면 나중에 처리해야 함
    //     // anim.runtimeAnimatorController = animCon[data.spriteType];
    //     gameObject.GetComponent<SpriteRenderer>().sprite = sprites[data.spriteType[index]];
    //     gameObject.GetComponent<EnemyMovement>().speed = data.speed[index];
    //     gameObject.GetComponent<EnemyMovement>().originalSpeed = data.speed[index];
    //     hp = data.hp[index];
    //     maxHp = data.hp[index];
    //     damage = data.damage[index];
    //     expIncrease = data.expIncrease;
    // }
}
