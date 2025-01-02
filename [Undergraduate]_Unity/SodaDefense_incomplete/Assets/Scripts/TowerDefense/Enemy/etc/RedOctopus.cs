using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/* RedOctopus: HP 증가 */
public class RedOctopus : AttackEnemy
{
//     private float multiplier = 2.0f;
//     private float coolDownMultiplier = 0.5f;

//     protected override void Awake() 
//     {
//         base.Awake();
//         // damage = damage * multiplier;
//         // speed = speed * multiplier;
//         // attackCooldown = attackCooldown * coolDownMultiplier;
//         hp = hp * multiplier;
//         // attackRange = attackRange * multiplier;
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
