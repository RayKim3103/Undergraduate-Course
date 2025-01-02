using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ArcherBasicSkillEffect : BasicAttack
{
    // private ArcherController playerController;

    // protected override void Awake() 
    // {
    //     base.Awake();
    //     initialPlayerPosition = player.transform.position; // 발사 시점의 플레이어 위치 저장
    //     // player = GameObject.FindWithTag(playerTag);
    //     playerController = player.GetComponent<ArcherController>();

    // }

    // protected override void Update() 
    // {
    //     base.Update();

    //     // 발사 시점 player 위치로 부터 maxDistance 이상 거리가 멀어지면 발사체 Destroy
    //     if (Vector3.Distance(initialPlayerPosition, transform.position) > maxDistance)
    //     {
    //         Destroy(gameObject); // 발사체 파괴
    //     }
    // }

    // public void BasicSkill()
    // {
    //         /********************** 스킬의 효과 조건문 ************************/

    //         /*******************************************************************/
    // }

    // protected override void HandleEnemy(GameObject target)
    // {
    //     base.HandleEnemy(target);
    //     BasicSkill();
    // }

    // /* 디버깅용: player의 기본 공격범위를 표시 */
    // private void OnDrawGizmos()
    // {
    //     Gizmos.color = Color.red;
    //     Gizmos.DrawWireSphere(player.transform.position, maxDistance);
    //     Gizmos.color = Color.blue;
    //     Gizmos.DrawWireSphere(transform.position, attackRange);
    // }

    // // prefab을 설정하는 메서드
    // public void SetPrefab(GameObject prefabToAssign)
    // {
    //     footPrefab = prefabToAssign;
    // }
}
