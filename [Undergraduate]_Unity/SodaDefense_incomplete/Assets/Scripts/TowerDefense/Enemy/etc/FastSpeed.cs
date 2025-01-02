// #define DEBUG_MODE
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/* PurpleOctopus: 이동속도 증가 */
public class FastSpeed : AttackEnemy
{
//     private float multiplier = 2.0f;
//     private float coolDownMultiplier = 0.5f;
//     protected override void Awake() 
//     {
//         base.Awake();
        
//         // attackCooldown = attackCooldown * coolDownMultiplier;
//         // attackRange = attackRange * multiplier;
//         // damage = damage * multiplier;
//         // hp = hp * multiplier;
//         gameObject.GetComponent<EnemyMovement>().speed = gameObject.GetComponent<EnemyMovement>().speed * multiplier;
//         gameObject.GetComponent<EnemyMovement>().originalSpeed = gameObject.GetComponent<EnemyMovement>().originalSpeed * multiplier;
//         expIncrease = expIncrease * multiplier;
//     }

//     // Update is called once per frame
//     protected override void Update()
//     {
//         base.Update();
//     }

//     public override void TakeDamage(float damage)
//     {
//         hp -= damage;
//         if (hp <= 0)
//         {
//             playerController.exp += expIncrease;
// #if DEBUG_MODE
//             // Debug 용 출력문
//             Debug.Log("Player EXP Up!! / Current EXP: " + playerController.exp);
// #endif
//             Die();
//         }
//     }
}
