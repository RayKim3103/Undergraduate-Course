using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;  

public class SkillController : MonoBehaviour
{  
    // 캐릭터 및 스킬 관련 변수
    private BaseCharacter baseCharacter;
    private SkillSelection.SkillType selectedSkill;

    public int[] skills;
    public int skillIndex = 0;

    // 스킬 컴포넌트 변수
    private Dictionary<SkillSelection.SkillType, ISkill> skillComponents = new Dictionary<SkillSelection.SkillType, ISkill>();
    public SkillData[] skillData;
    public int[] skillLevel;

    protected void Awake() 
    {
        baseCharacter = GetComponentInParent<BaseCharacter>();

        /* 3. skillComponents에 스킬 추가 (ISkill 실수하면 여기서 에러 나옴) */ /* 3번 했으면, SkillSlot 스크립트로 이동 */
        /********************************* 모든 스킬 컴포넌트를 Dictionary에 저장 *********************************/
        skillComponents.Add(SkillSelection.SkillType.SummonAttack, GetComponentInChildren<Summon>());
        skillComponents.Add(SkillSelection.SkillType.WideRangeAttack, GetComponentInChildren<WideRange>());
        skillComponents.Add(SkillSelection.SkillType.CurseAttack, GetComponentInChildren<Curse>());
        skillComponents.Add(SkillSelection.SkillType.IceAttack, GetComponentInChildren<IceAttack>());
        skillComponents.Add(SkillSelection.SkillType.ManaShield, GetComponentInChildren<ManaShield>());
        skillComponents.Add(SkillSelection.SkillType.Heal, GetComponentInChildren<Heal>());
        skillComponents.Add(SkillSelection.SkillType.Teleport, GetComponentInChildren<Teleport>());
        skillComponents.Add(SkillSelection.SkillType.InstantiateMagicStorm, GetComponentInChildren<InstantiateMagicStorm>());
        skillComponents.Add(SkillSelection.SkillType.Confuse, GetComponentInChildren<Confuse>());
        skillComponents.Add(SkillSelection.SkillType.TripleSlash, GetComponentInChildren<TripleSlash>());
        skillComponents.Add(SkillSelection.SkillType.AreaAttack, GetComponentInChildren<AreaAttack>());
        skillComponents.Add(SkillSelection.SkillType.LifeSteal, GetComponentInChildren<LifeStealSkill>());
        skillComponents.Add(SkillSelection.SkillType.Boomerang, GetComponentInChildren<Boomerang>());
        skillComponents.Add(SkillSelection.SkillType.SphereAttack, GetComponentInChildren<SphereAttack>());
        skillComponents.Add(SkillSelection.SkillType.CounterAttack, GetComponentInChildren<CounterAttack>());
    }

    protected void Update()
    {
        // SkillSelection을 처음에 Player의 자식으로 안 뒀기에 에러 나올수 있음
        if (baseCharacter == null)
        {
            baseCharacter = GetComponentInParent<BaseCharacter>();
        }
        if(baseCharacter.isDied == false)
            UseSkill();
    }

    public void SetSelectedSkill(SkillSelection.SkillType skill)
    {
        selectedSkill = skill;
        int index = (int)selectedSkill;

        if (ArrayContainsValue(skills, index))
        {
            UpgradeSkill(selectedSkill, index);
        }
        else
        {
            AssignSkill(index);
        }

        // Time.timeScale = 1f; // 게임 재개
    }

    private void AssignSkill(int index)
    {
        if (skillIndex < skills.Length)
        {
            skills[skillIndex] = index;
            skillIndex++;
        }
    }

    private void UpgradeSkill(SkillSelection.SkillType skillType, int index)
    {
        // 키를 사용하여 값 가져오기: skillComponents 딕셔너리에서 skillType 키에 해당하는 값을 가져옴
        // 여기서 skillType은 특정 스킬 타입을 나타내며, skillComponents[skillType]는 해당 스킬 타입에 해당하는 ISkill 객체를 반환
        // 반환된 값은 ISkill 인터페이스 타입으로 캐스팅: ISkill 인터페이스에서 정의된 Activate와 LevelUp 메서드를 사용할 수 있게 됨
        ISkill skillComponent = skillComponents[skillType];
        if (skillComponent != null)
        {
            skillComponent.LevelUp(skillData[index].increaseAmount[skillLevel[index]]);
            Debug.Log($"{skillType} LevelUp called!!!");

            if (skillLevel[index] < skillData[index].increaseAmount.Length - 1)
                skillLevel[index]++;
        }
    }

    public void UseSkill()
    {
        if (!baseCharacter.isDied)
        {
            // skills 배열은 스킬을 획득하면 0이 아닌 값을 가짐, 현재 Dictionary에는 key가 0인 값 없음
            foreach (int skill in skills)
            {
                SkillSelection.SkillType skillType = (SkillSelection.SkillType)skill;
                if (skillComponents.ContainsKey(skillType))
                {
                    skillComponents[skillType].Activate();
                }
                else
                {
#if DEBUG_MODE
                    Debug.Log("Unknown skill number: " + skill);
#endif
                }
            }
        }
    }

    public bool ArrayContainsValue(int[] array, int value)
    {
        foreach (int element in array)
        {
            if (element == value)
            {
                return true;
            }
        }
        return false;
    }
}

// 스킬 인터페이스 정의
public interface ISkill
{
    void Activate();
    void SkillEffectActivate(Transform target);
    void LevelUp(float increaseAmount);
}

// 1개의 스크립트 내에 여러개의 클래스 선언 가능
// 직렬화: 개체를 저장 or 전송하기 위해 변환
[System.Serializable]
public class SkillData
{
    public SkillSelection.SkillType skillType;
    public Sprite icon;
    public float[] increaseAmount;
    public int needCoin;
}

// using System.Collections;
// using System.Collections.Generic;
// using TMPro;
// using Unity.VisualScripting;
// using UnityEngine;  

// public class SkillController : MonoBehaviour
// {  
//     /* 캐릭터 set */
//     // public string playerTag = "Player"; // Player 오브젝트의 태그
//     // private GameObject player;
//     private BaseCharacter baseCharacter;
//     private SkillSelection.SkillType selectedSkill;
    
//     /* 캐릭터 스킬 set */
//     public int[] skills;
//     public int skillIndex = 0;

//     /**************  스킬들의 변수들을 선언함 **************/

//     public SkillData[] skillData;
//     public int[] skillLevel;

//     /* Summoner 스킬 변수 */
//     private Summon summon;

//     /* ManaShield 스킬 변수 */
//     private ManaShield manaShield;

//     /* Heal 스킬 변수 */
//     private Heal heal;

//     /* Teleport 스킬 변수 */
//     private Teleport teleport;

//     /* IncreaseManaPower 스킬 변수 */
//     private IncreaseManaPower increaseManaPower;

//     /* IncreaseAttackSpeed 스킬 변수 */
//     private IncreaseAttackSpeed increaseAttackSpeed;

//     /* InstantiateMagicStorm 스킬 변수 */
//     private InstantiateMagicStorm instantiateMagicStorm;

//     /* TripleSlash 스킬 변수 */
//     private TripleSlash tripleSlash;

//     /* AreaAttack 스킬 변수 */
//     private AreaAttack areaAttack;
//         /* WideRange 공격 변수 */
//     private WideRange wideRange;

//     /* Curse 공격 변수 */
//     private Curse curse;

//     /* Ice 공격 변수 */
//     private IceAttack iceAttack;

//     /* Confuse 스킬 변수 */
//     private Confuse confuse;

//     /* 흡혈 스킬 변수 */
//     private LifeStealSkill lifeStealSkill;

//     /***********************************************************/

//     // private System.Action[] skillLevelUpFunctions;

//     protected void Awake() 
//     {   
//         //player = GameObject.FindWithTag(playerTag);
//         baseCharacter = GetComponentInParent<BaseCharacter>();

//         /*************  스킬들의 컴포넌트들을 가져옴 *************/

//         // Summon 컴포넌트 가져옴 
//         summon = GetComponentInChildren<Summon>();

//         // WideRange 컴포넌트 가져옴 
//         wideRange = GetComponentInChildren<WideRange>();

//         // Curse 컴포넌트 가져옴 
//         curse = GetComponentInChildren<Curse>();

//         // IceAttack 컴포넌트 가져옴 
//         iceAttack = GetComponentInChildren<IceAttack>();

//         // ManaShield 컴포넌트 가져옴 
//         manaShield = GetComponentInChildren<ManaShield>();

//         // Heal 컴포넌트 가져옴
//         heal = GetComponentInChildren<Heal>();

//         // Teleport 컴포넌트 가져옴;
//         teleport = GetComponentInChildren<Teleport>();

//         // IncreaseManaPower 컴포넌트 가져옴;
//         increaseManaPower = GetComponentInChildren<IncreaseManaPower>();

//         // IncreaseAttackSpeed 컴포넌트 가져옴;
//         increaseAttackSpeed = GetComponentInChildren<IncreaseAttackSpeed>();

//         // InstantiateMagicStorm 컴포넌트 가져옴;
//         instantiateMagicStorm = GetComponentInChildren<InstantiateMagicStorm>();

//         // Confuse 컴포넌트 가져옴
//         confuse = GetComponentInChildren<Confuse>();

//         // TripleSlash 컴포넌트 가져옴
//         tripleSlash = GetComponentInChildren<TripleSlash>();

//         // AreaAttack 컴포넌트 가져옴
//         areaAttack = GetComponentInChildren<AreaAttack>();

//         // LifeStealSkill 컴포넌트 가져옴
//         lifeStealSkill = GetComponentInChildren<LifeStealSkill>();
//         /***********************************************************/
//     }

//     protected void Update()
//     {
//         // 할당된 스킬을 사용가능하게 해 줌
//         UseSkill();
//     }

//     // public void SkillLevelUp(int index)
//     // {
//     //     if(!ArrayContainsValue(skills, index))
//     //     {
//     //         skills[this.skillIndex] = index;
//     //     } 
//     //     else
//     //     {
//     //         summon.SummonLevelUp(skillData[index].increaseAmount[skillLevel[index]]);
//     //         Debug.Log("Summon LevelUp called!!!");

//     //         if(skillLevel[index] < skillData[index].increaseAmount.Length)
//     //             skillLevel[index]++;
//     //     }   
//     // }

//     /********** Skill 선택한 것을 받는 함수, 배열에 저장 **********/
//     public void SetSelectedSkill(SkillSelection.SkillType skill)
//     {
//         selectedSkill = skill;
//         int index;

//         switch (selectedSkill)
//         {
//             // skills[skillIndex] = 0 은 스킬이 없는 상태로 지정함
//             // 따라서, 1, 2, 3, 4, ... 로 할당
//             case SkillSelection.SkillType.SummonAttack:
//                 // 소환 스킬 처리 -> SkillLevelUp(int index) 의 로직은 아래 코드와 같음
//                 index = (int)SkillSelection.SkillType.SummonAttack;
//                 if(!ArrayContainsValue(skills, index))
//                 {
//                     skills[skillIndex] = index;
//                     skillIndex += 1;
//                 } 
//                 else
//                 {
//                     summon.SummonLevelUp(skillData[index].increaseAmount[skillLevel[index]]);
//                     Debug.Log("Summon LevelUp called!!!");

//                     if(skillLevel[index] < skillData[index].increaseAmount.Length - 1)
//                         skillLevel[index]++;
//                 }   
//                 break;
//             case SkillSelection.SkillType.WideRangeAttack:
//                 // 광역 스킬 처리
//                 index = (int)SkillSelection.SkillType.WideRangeAttack;
//                 if(!ArrayContainsValue(skills, index))
//                 {
//                     skills[skillIndex] = index;
//                     skillIndex += 1;
//                 } 
//                 else
//                 {
//                     wideRange.WideRangeLevelUp(skillData[index].increaseAmount[skillLevel[index]]);
//                     Debug.Log("WideRange LevelUp called!!!");

//                     if(skillLevel[index] < skillData[index].increaseAmount.Length - 1)
//                         skillLevel[index]++;
//                 }  
//                 break;
//             case SkillSelection.SkillType.CurseAttack:
//                 // 저주 스킬 처리
//                 index = (int)SkillSelection.SkillType.CurseAttack;
//                 if(!ArrayContainsValue(skills, index))
//                 {
//                     skills[skillIndex] = index;
//                     skillIndex += 1;
//                 } 
//                 else
//                 {
//                     curse.CurseLevelUp(skillData[index].increaseAmount[skillLevel[index]]);
//                     Debug.Log("Curse LevelUp called!!!");

//                     if(skillLevel[index] < skillData[index].increaseAmount.Length - 1)
//                         skillLevel[index]++;
//                 }   
//                 break;
//             case SkillSelection.SkillType.IceAttack:
//                 // 빙결 스킬 처리
//                 index = (int)SkillSelection.SkillType.IceAttack;
//                 if(!ArrayContainsValue(skills, index))
//                 {
//                     skills[skillIndex] = index;
//                     skillIndex += 1;
//                 } 
//                 else
//                 {
//                     iceAttack.IceLevelUp(skillData[index].increaseAmount[skillLevel[index]]);
//                     Debug.Log("IceAttack LevelUp called!!!");

//                     if(skillLevel[index] < skillData[index].increaseAmount.Length - 1)
//                         skillLevel[index]++;
//                 }   
//                 break;
//             case SkillSelection.SkillType.ManaShield:
//                 // 쉴드 스킬 처리
//                 index = (int)SkillSelection.SkillType.ManaShield;
//                 if(!ArrayContainsValue(skills, index))
//                 {
//                     skills[skillIndex] = index;
//                     skillIndex += 1;
//                 } 
//                 else
//                 {
//                     manaShield.ManaShieldLevelUp(skillData[index].increaseAmount[skillLevel[index]]);
//                     Debug.Log("ManaShield LevelUp called!!!");

//                     if(skillLevel[index] < skillData[index].increaseAmount.Length - 1)
//                         skillLevel[index]++;
//                 }   
//                 break;
//             case SkillSelection.SkillType.Heal:
//                 // 회복 스킬 처리
//                 index = (int)SkillSelection.SkillType.Heal;
//                 if(!ArrayContainsValue(skills, index))
//                 {
//                     skills[skillIndex] = index;
//                     skillIndex += 1;
//                 } 
//                 else
//                 {
//                     heal.HealLevelUp(skillData[index].increaseAmount[skillLevel[index]]);
//                     Debug.Log("Heal LevelUp called!!!");

//                     if(skillLevel[index] < skillData[index].increaseAmount.Length - 1)
//                         skillLevel[index]++;
//                 }   
//                 break;
//             case SkillSelection.SkillType.Teleport:
//                 // 순간이동 스킬 처리
//                 index = (int)SkillSelection.SkillType.Teleport;
//                 if(!ArrayContainsValue(skills, index))
//                 {
//                     skills[skillIndex] = index;
//                     skillIndex += 1;
//                 } 
//                 else
//                 {
//                     teleport.TeleportLevelUp(skillData[index].increaseAmount[skillLevel[index]]);
//                     Debug.Log("Teleport LevelUp called!!!");

//                     if(skillLevel[index] < skillData[index].increaseAmount.Length - 1)
//                         skillLevel[index]++;
//                 }   
//                 break;
//             case SkillSelection.SkillType.InstantiateMagicStorm:
//                 // 마력폭풍 스킬 처리
//                 index = (int)SkillSelection.SkillType.InstantiateMagicStorm;
//                 if(!ArrayContainsValue(skills, index))
//                 {
//                     skills[skillIndex] = index;
//                     skillIndex += 1;
//                 } 
//                 else
//                 {
//                     instantiateMagicStorm.MagicStormLevelUp(skillData[index].increaseAmount[skillLevel[index]]);
//                     Debug.Log("MagicStorm LevelUp called!!!");

//                     if(skillLevel[index] < skillData[index].increaseAmount.Length - 1)
//                         skillLevel[index]++;
//                 }   
//                 break;
//             case SkillSelection.SkillType.Confuse:
//                 // Confuse 스킬 처리
//                 index = (int)SkillSelection.SkillType.Confuse;
//                 if(!ArrayContainsValue(skills, index))
//                 {
//                     skills[skillIndex] = index;
//                     skillIndex += 1;
//                 } 
//                 else
//                 {
//                     confuse.ConfuseLevelUp(skillData[index].increaseAmount[skillLevel[index]]);
//                     Debug.Log("Confuse LevelUp called!!!");

//                     if(skillLevel[index] < skillData[index].increaseAmount.Length - 1)
//                         skillLevel[index]++;
//                 }   
//                 break;
//             case SkillSelection.SkillType.TripleSlash:
//                 // TripleSlash 스킬 처리
//                 index = (int)SkillSelection.SkillType.TripleSlash;
//                 if(!ArrayContainsValue(skills, index))
//                 {
//                     skills[skillIndex] = index;
//                     skillIndex += 1;
//                 } 
//                 else
//                 {
//                     tripleSlash.TripleSlashLevelUp(skillData[index].increaseAmount[skillLevel[index]]);
//                     Debug.Log("TripleSlash LevelUp called!!!");

//                     if(skillLevel[index] < skillData[index].increaseAmount.Length - 1)
//                         skillLevel[index]++;
//                 }   
//                 break;
//             case SkillSelection.SkillType.AreaAttack:
//                 index = (int)SkillSelection.SkillType.AreaAttack;
//                 if(!ArrayContainsValue(skills, index))
//                 {
//                     skills[skillIndex] = index;
//                     skillIndex += 1;
//                 } 
//                 else
//                 {
//                     areaAttack.AreaAttackLevelUp(skillData[index].increaseAmount[skillLevel[index]]);
//                     Debug.Log("AreaAttack LevelUp called!!!");

//                     if(skillLevel[index] < skillData[index].increaseAmount.Length - 1)
//                         skillLevel[index]++;
//                 }   
//                 break;
//             case SkillSelection.SkillType.LifeSteal:
//                 index = (int)SkillSelection.SkillType.LifeSteal;
//                 if(!ArrayContainsValue(skills, index))
//                 {
//                     skills[skillIndex] = index;
//                     skillIndex += 1;
//                 } 
//                 else
//                 {
//                     lifeStealSkill.LifeStealLevelUp(skillData[index].increaseAmount[skillLevel[index]]);
//                     Debug.Log("LifeSteal LevelUp called!!!");

//                     if(skillLevel[index] < skillData[index].increaseAmount.Length - 1)
//                         skillLevel[index]++;
//                 }   
//                 break;
//             case SkillSelection.SkillType.Null_1:
//                 index = (int)SkillSelection.SkillType.Null_1;
//                 if(!ArrayContainsValue(skills, index))
//                 {
//                     skills[skillIndex] = index;
//                     skillIndex += 1;
//                 } 
//                 else
//                 {
//                     if(skillLevel[index] < skillData[index].increaseAmount.Length - 1)
//                         skillLevel[index]++;
//                 }   
//                 break;
//             case SkillSelection.SkillType.Null_2:
//                 index = (int)SkillSelection.SkillType.Null_2;
//                 if(!ArrayContainsValue(skills, index))
//                 {
//                     skills[skillIndex] = index;
//                     skillIndex += 1;
//                 } 
//                 else
//                 {
//                     if(skillLevel[index] < skillData[index].increaseAmount.Length - 1)
//                         skillLevel[index]++;
//                 }   
//                 break;
//             case SkillSelection.SkillType.Null_3:
//                 index = (int)SkillSelection.SkillType.Null_3;
//                 if(!ArrayContainsValue(skills, index))
//                 {
//                     skills[skillIndex] = index;
//                     skillIndex += 1;
//                 } 
//                 else
//                 {
//                     if(skillLevel[index] < skillData[index].increaseAmount.Length - 1)
//                         skillLevel[index]++;
//                 }   
//                 break;
//             default:
//                 // 예외 처리 등
//                 break;
//         }

//         // // Debug 용 출력문
//         // Debug.Log("SkillIndex: " + skillIndex + " SelectedSkill: " + skills[skillIndex]);
        
//         // // skills 배열의 초기화는 SkillSelection에서 하고 있음
//         // if(skillIndex < skills.Length-1)
//         // {
//         //     // skillIndex 증가 및 범위가 안 넘어가도록 조절
//         //     skillIndex += 1;
//         // }

//         Time.timeScale = 1f; // 게임 재개
//         // UIManager.instance.CloseSkillSelectPanel();
//     }
//     /***************************************************************/


//     /*************  Mage 스킬들의 잠금 해제 or 잠금 여부 *************/
//     public void UseSkill()
//     {
//         if(baseCharacter.isDied == false)
//         {
//             // skills 배열에 있는 각 스킬 번호에 대해 반복
//             foreach (int skill in skills)
//             {
//                 // 스킬 번호에 따라 해당 스킬을 실행
//                 switch (skill)
//                 {
//                     case (int)SkillSelection.SkillType.SummonAttack:
//                         // 1번 스킬: Summon Skill 사용
//                         summon.BasicSummon();
//                         break;
//                     case (int)SkillSelection.SkillType.ManaShield:
//                         // 5번 스킬: Mana Shield 사용
//                         manaShield.ActivateManaShield();
//                         break;
//                     case (int)SkillSelection.SkillType.Heal:
//                         // 6번 스킬: Heal 사용
//                         heal.Healling();
//                         break;
//                     case (int)SkillSelection.SkillType.Teleport:
//                         // 7번 스킬: Teleport 사용
//                         teleport.ActivateTeleport();
//                         break;
//                     case (int)SkillSelection.SkillType.TripleSlash:
//                         // 10번 스킬: Teleport 사용
//                         tripleSlash.Slash();
//                         break;
//                     case (int)SkillSelection.SkillType.AreaAttack:
//                         // 11번 스킬: AreaAttack 사용
//                         areaAttack.ActivateAreaAttack();
//                         break;
//                     default:
// #if DEBUG_MODE
//                         // 정의되지 않은 스킬 번호가 있을 경우 디버그 메시지 출력
//                         Debug.Log("Unknown skill number: " + skill);
// #endif
//                         break;
//                 }
//             }
//         }
//     }
//     /***************************************************************/
                
//     /******** 배열의 해당 값이 있는지 찾는 함수 : 건드릴 필요 X *******/
//     public bool ArrayContainsValue(int[] array, int value)
//     {
//         foreach (int element in array)
//         {
//             if (element == value)
//             {
//                 return true;  // 값이 배열에 존재할 경우 true 반환
//             }
//         }
//         return false;         // 값이 배열에 존재하지 않을 경우 false 반환
//     }
//     /***************************************************************/
// }

