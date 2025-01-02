// #define DEBUG_MODE
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;

/* MageBasicAttack한테 붙일 스크립트로 변경함 */
/* 부모: MageBasicAttack */
public class MageBasicSkillEffect : BasicAttack
{
//     private MageController playerController;
    
//     /* 발사체의 범위 변수 */

//     [SerializeField]
//     private float maxDistance = 5.0f;
//     private Vector3 initialPlayerPosition; // 발사 시점의 플레이어 위치

//     /* WideRange 공격 변수 */
//     private WideRange wideRange;

//     /* Curse 공격 변수 */
//     private Curse curse;

//     /* Ice 공격 변수 */
//     private IceAttack iceAttack;

//     /* InstantiateMagicStorm 스킬 변수 */
//     private InstantiateMagicStorm instantiateMagicStorm;

//     /* Confuse 스킬 변수 */
//     private Confuse confuse;

//     protected override void Awake() 
//     {
//         base.Awake();
//         initialPlayerPosition = player.transform.position; // 발사 시점의 플레이어 위치 저장
//         // player = GameObject.FindWithTag(playerTag);
//         playerController = player.GetComponent<MageController>();
//         wideRange = player.GetComponent<WideRange>();
//         curse = player.GetComponent<Curse>();
//         iceAttack = player.GetComponent<IceAttack>();
//         instantiateMagicStorm = player.GetComponent<InstantiateMagicStorm>();
//         confuse = player.GetComponent<Confuse>();
//     }

//     protected override void Update() 
//     {
//         base.Update();

//         // 발사 시점 player 위치로 부터 maxDistance 이상 거리가 멀어지면 발사체 Destroy
//         if (Vector3.Distance(initialPlayerPosition, transform.position) > maxDistance)
//         {
//             Destroy(gameObject); // 발사체 파괴
//         }
//         // MageBasicAttack mageBasicAttack = gameObject.GetComponent<MageBasicAttack>();
//         // if (mageBasicAttack != null && mageBasicAttack.target != null)
//         // {
//         //     attackTarget = mageBasicAttack.target.gameObject;  
//         //     Debug.Log("MageBasicSkillEffect / attackTarget: " + attackTarget);
//         // }   
//         // else
//         // {
//         //     Debug.Log("MageBasicSkillEffect / No attackTarget!!!: " + attackTarget);
//         // }
//     }

//     public void BasicSkill()
//     {
//             /********************** 스킬의 효과 조건문 ************************/
//             if (playerController.ArrayContainsValue(playerController.skills, (int)SkillSelection.SkillType.WideRangeAttack))
//             {
// #if DEBUG_MODE
//                 Debug.Log("PlayerController has WideRange!");
// #endif
//                 wideRange.ApplyExplosionDamage(target.position);
//                 // 폭발 효과 생성
//                 Instantiate(wideRange.explosionPrefab, target.transform.position, target.transform.rotation);
//             }
//             if (playerController.ArrayContainsValue(playerController.skills, (int)SkillSelection.SkillType.CurseAttack))
//             {
// #if DEBUG_MODE
//                 Debug.Log("PlayerController has CurseAttack!");
// #endif
//                 // curse.BasicCurse();
//                 // 저주 효과 생성
//                 Instantiate(curse.cursePrefab, target.transform.position, target.transform.rotation);
//             }
//             if (playerController.ArrayContainsValue(playerController.skills, (int)SkillSelection.SkillType.IceAttack) && target.GetComponent<AttackEnemy>().hp > 0)
//             {
// #if DEBUG_MODE
//                 Debug.Log("PlayerController has IceAttack!");
// #endif
//                 // 얼음 효과 생성
//                 iceAttack.MakeIce(target);
//             }
//             if (playerController.ArrayContainsValue(playerController.skills, (int)SkillSelection.SkillType.InstantiateMagicStorm))
//             {
//                 // 마력폭풍 효과 생성
//                 instantiateMagicStorm.MakeMagicStorm(target);
//             }
//             if (playerController.ArrayContainsValue(playerController.skills, (int)SkillSelection.SkillType.Confuse))
//             {
//                 // 혼란 효과 생성
//                 confuse.ConfuseAttack(target);
//             }
//             /*******************************************************************/
//     }

//     protected override void HandleEnemy(GameObject target)
//     {
//         base.HandleEnemy(target);
//         BasicSkill();
//     }

//     /* 디버깅용: player의 기본 공격범위를 표시 */
//     private void OnDrawGizmos()
//     {
//         Gizmos.color = Color.red;
//         Gizmos.DrawWireSphere(player.transform.position, maxDistance);
//         Gizmos.color = Color.blue;
//         Gizmos.DrawWireSphere(transform.position, attackRange);
//     }

//     // // prefab을 설정하는 메서드
//     // public void SetPrefab(GameObject prefabToAssign)
//     // {
//     //     footPrefab = prefabToAssign;
//     // }
}
