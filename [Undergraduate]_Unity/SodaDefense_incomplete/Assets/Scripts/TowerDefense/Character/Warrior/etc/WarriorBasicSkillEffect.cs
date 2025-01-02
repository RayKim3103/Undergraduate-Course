using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WarriorBasicSkillEffect : BasicAttack
{
    // private WarriorController playerController;

    // /* 발사체의 범위 변수 */
    // [SerializeField]
    // private float maxDistance = 5f; // 기본: 매우 짧은 범위 -> 참격 증강 선택 시 참격이 날라가는 범위로써 사용 가능
    // private Vector3 initialPlayerPosition; // 발사 시점의 플레이어 위치
    // public float increaseDetection = 2.0f;

    // /* 참격이 활성화 될 시 (나중 증강 위한 변수) */
    // public bool isLongRange = false;

    // /* TripleSlash 스킬 변수 */
    // private TripleSlash tripleSlash;

    // /* 흡혈 스킬 변수 */
    // private LifeStealSkill lifeStealSkill;
    

    // protected override void Awake() 
    // {
    //     base.Awake();
    //     initialPlayerPosition = player.transform.position; // 발사 시점의 플레이어 위치 저장
    //     attackRange = attackRange * increaseDetection; // 발사체가 적을 찾는 범위 증가 (전사는 짧은 리치지만, 더 넓은 범위에서 적 1명을 공격)
    //     playerController = player.GetComponent<WarriorController>();
    //     lifeStealSkill = player.GetComponent<LifeStealSkill>();
    //     tripleSlash = player.GetComponent<TripleSlash>();
    // }

    // protected override void Update() 
    // {
    //     if(isLongRange == false)
    //     {
    //         FindTarget(); // 타겟을 찾습니다.
    //         if (target != null)
    //         {
    //             if (IsTargetInRange())
    //             {
    //                 AttackTarget(); // 타겟이 범위 안에 있으면 공격합니다.
    //             }
    //             else
    //             {
    //                 Destroy(gameObject); // prefab 파괴
    //             }
    //         }
    //     }
        
    //     else
    //     {
    //         base.Update();
    //         // 발사 시점 player 위치로 부터 maxDistance 이상 거리가 멀어지면 발사체 Destroy
    //         if (Vector3.Distance(initialPlayerPosition, transform.position) > maxDistance)
    //         {
    //             Destroy(gameObject); // 발사체 파괴
    //         }
    //     }
    // }

    // public void BasicSkill(GameObject target)
    // {
    //     if (playerController.ArrayContainsValue(playerController.skills, 1))
    //     {
    //         tripleSlash.Slash();
    //     }
    //     if (playerController.ArrayContainsValue(playerController.skills, 4))
    //     {
    //         lifeStealSkill.ApplyLifeSteal();
    //     }
    // }

    // protected override void HandleEnemy(GameObject target)
    // {        
    //     base.HandleEnemy(target);
    //     BasicSkill(target); // target을 전달하여 BasicSkill 호출
    // }
}
