#define DEBUG_MODE
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;

/* MageBasicAttack한테 붙일 스크립트로 변경함 */
/* 부모: MageBasicAttack */
public class BasicSkillEffect : BasicAttack
{
    private SkillController skillController;
    
    public float increaseDetection = 3.0f; // 근접 캐릭의 경우 때릴 수 있는 거리 3배 -> 기본사거리보다 길어야

    // 스킬 컴포넌트 변수
    private Dictionary<SkillSelection.SkillType, ISkill> skillComponents = new Dictionary<SkillSelection.SkillType, ISkill>();

    protected override void Start() 
    {
        base.Start();
        
        // player = GameObject.FindWithTag(playerTag);
        skillController = player.GetComponentInChildren<SkillController>();
        skillComponents.Add(SkillSelection.SkillType.WideRangeAttack, player.GetComponentInChildren<WideRange>());
        skillComponents.Add(SkillSelection.SkillType.CurseAttack, player.GetComponentInChildren<Curse>());
        skillComponents.Add(SkillSelection.SkillType.IceAttack, player.GetComponentInChildren<IceAttack>());
        skillComponents.Add(SkillSelection.SkillType.InstantiateMagicStorm, player.GetComponentInChildren<InstantiateMagicStorm>());
        skillComponents.Add(SkillSelection.SkillType.Confuse, player.GetComponentInChildren<Confuse>());
        skillComponents.Add(SkillSelection.SkillType.TripleSlash, player.GetComponentInChildren<TripleSlash>());
        skillComponents.Add(SkillSelection.SkillType.LifeSteal, player.GetComponentInChildren<LifeStealSkill>());
        
        if (player.GetComponent<WarriorController>() != null)
            attackRange = attackRange * increaseDetection; // 발사체가 적을 찾는 범위 증가 (전사는 짧은 리치지만, 더 넓은 범위에서 적 1명을 공격)
    }

    protected override void Update() 
    {
        if (player.GetComponent<WarriorController>() != null)
        {
            if (IsTooFarFromInitialPosition())
            {
                Destroy(gameObject);
                return;
            }

            if (target != null && IsTargetInRange())
            {
                AttackTarget();
            }

            if (target == null)
            {
                FindTarget();
            }
        }
        else
        {
            base.Update();
        }
    }

    public void BasicSkill()
    {
            /********************** 스킬의 효과 조건문 ************************/
            // skillController.skills 배열은 스킬을 획득하면 0이 아닌 값을 가짐, 현재 Dictionary에는 key가 0인 값 없음
            foreach (int skillType in skillController.skills)
            {
                if (skillComponents.TryGetValue((SkillSelection.SkillType)skillType, out ISkill skill))
                {
    #if DEBUG_MODE
                    Debug.Log($"SkillController has {skillType}! / skill: " + skill);
    #endif
                    if(player.GetComponent<BaseCharacter>().isDied == false)
                        skill.SkillEffectActivate(target);
                }
            }
            /*******************************************************************/
    }

    protected override void AttackTarget()
    {
        base.AttackTarget();
        BasicSkill();
    }

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
