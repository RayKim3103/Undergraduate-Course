using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class SkillSlot : MonoBehaviour
{
    [SerializeField]
    private GameObject skillPrefab;
    public Text skillName;
    public Text description;
    public Button button;

    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private SkillController skillController;

    /* 4. SkillNames에 스킬 추가 */
    /* -> 이제 유니티로 이동해서 SkillControl 게임오브젝트 -> SkillController -> SkillData를 수정 -> SkillSet에 스크립트 넣으면 끝 */
    private static readonly Dictionary<SkillSelection.SkillType, string> SkillNames = new()
    {
        { SkillSelection.SkillType.SummonAttack, "Summon" },
        { SkillSelection.SkillType.WideRangeAttack, "WideRange" },
        { SkillSelection.SkillType.CurseAttack, "Curse" },
        { SkillSelection.SkillType.IceAttack, "Ice" },
        { SkillSelection.SkillType.ManaShield, "ManaShield" },
        { SkillSelection.SkillType.Heal, "Heal" },
        { SkillSelection.SkillType.Teleport, "Teleport" },
        { SkillSelection.SkillType.InstantiateMagicStorm, "InstantiateMagicStorm" },
        { SkillSelection.SkillType.Confuse, "Confuse" },
        { SkillSelection.SkillType.TripleSlash, "TripleSlash" },
        { SkillSelection.SkillType.AreaAttack, "AreaAttack" },
        { SkillSelection.SkillType.LifeSteal, "LifeSteal" },
        { SkillSelection.SkillType.Boomerang, "Boomerang" },
        { SkillSelection.SkillType.SphereAttack, "SphereAttack" },
        { SkillSelection.SkillType.CounterAttack, "CounterAttack" }
    };

    private void Awake()
    {
        // Awake는 주석 처리되었으므로 사용하지 않음
        // player = GameObject.FindWithTag(playerTag);
        // skillController = player.GetComponent<SkillController>();
    }

    public void Setup(SkillSelection.SkillType newSkill)
    {
        player = GameObject.FindWithTag(playerTag);
        skillController = player.GetComponentInChildren<SkillController>();
        Debug.Log("SkillSlot skillController: " + skillController);

        if (skillController == null)
        {
            Destroy(gameObject);
            return;
        }

        // 스킬 타입에 따라 설정
        if (SkillNames.TryGetValue(newSkill, out var skillName))
        {
            int skillIndex = (int)newSkill;
            if (skillController.ArrayContainsValue(skillController.skills, skillIndex) == false)
            {
                SetupSkill(skillIndex, skillName);
            }
            else
            {
                Destroy(gameObject);
            }
        }
        else
        {
            Debug.LogError("Unknown skill type!");
            Destroy(gameObject);
        }
    }

    private void SetupSkill(int skillIndex, string name)
    {
        Sprite skillIcon = skillController.skillData[skillIndex].icon;
        GameObject child = Instantiate(skillPrefab);

        // 자식 오브젝트 설정
        child.transform.SetParent(transform);
        child.transform.localPosition = Vector3.zero;
        child.GetComponent<Image>().sprite = skillIcon;
        child.name = name;
    }
}


// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine;
// using UnityEngine.UI;

// public class SkillSlot : MonoBehaviour
// {
//     // public Sprite[] icons;
//     private Sprite image;

//     [SerializeField]
//     private GameObject skillPrefab;
//     public Text skillName;
//     public Text description;
//     public Button button;
//     // public SkillSelection.SkillType mageSkill;

//     GameObject child;

//     public string playerTag = "Player"; // Player 오브젝트의 태그
//     private GameObject player;
//     private SkillController skillController;

//     /* PreFab이라서 게임 시작시 Awake 호출 안됨 */
//     // private void Awake() 
//     // {
//     //     player = GameObject.FindWithTag(playerTag);
//     //     skillController = player.GetComponent<SkillController>();
//     //     Debug.Log("SkillSlot sillController: " + skillController);
//     // }

//     public void Setup(SkillSelection.SkillType newSkill)
//     {
//         player = GameObject.FindWithTag(playerTag);
//         skillController = player.GetComponentInChildren<SkillController>();
//         Debug.Log("SkillSlot sillController: " + skillController);
//         if(skillController != null)
//         {
//             switch (newSkill)
//             {
//                 case SkillSelection.SkillType.SummonAttack:
//                     if(skillController.ArrayContainsValue(skillController.skills, (int)SkillSelection.SkillType.SummonAttack) == false)
//                     {
//                         image = skillController.skillData[(int)SkillSelection.SkillType.SummonAttack].icon;
//                         child = Instantiate(skillPrefab);
//                         child.name = "Summon";
//                     }
//                     break;
//                 case SkillSelection.SkillType.WideRangeAttack:
//                     if(skillController.ArrayContainsValue(skillController.skills, (int)SkillSelection.SkillType.WideRangeAttack) == false)
//                     {
//                         image = skillController.skillData[(int)SkillSelection.SkillType.WideRangeAttack].icon;
//                         child = Instantiate(skillPrefab);
//                         child.name = "WideRange";
//                     }
//                     break;
//                 case SkillSelection.SkillType.CurseAttack:
//                     if(skillController.ArrayContainsValue(skillController.skills, (int)SkillSelection.SkillType.CurseAttack) == false)
//                     {
//                         image = skillController.skillData[(int)SkillSelection.SkillType.CurseAttack].icon;
//                         child = Instantiate(skillPrefab);
//                         child.name = "Curse";
//                     }
//                     break;
//                 case SkillSelection.SkillType.IceAttack:
//                     if(skillController.ArrayContainsValue(skillController.skills, (int)SkillSelection.SkillType.IceAttack) == false)
//                     {
//                         image = skillController.skillData[(int)SkillSelection.SkillType.IceAttack].icon;
//                         child = Instantiate(skillPrefab);
//                         child.name = "Ice";
//                     }
//                     break;
//                 case SkillSelection.SkillType.ManaShield:
//                     if(skillController.ArrayContainsValue(skillController.skills, (int)SkillSelection.SkillType.ManaShield) == false)
//                     {
//                         image = skillController.skillData[(int)SkillSelection.SkillType.ManaShield].icon;
//                         child = Instantiate(skillPrefab);
//                         child.name = "ManaShield";
//                     }
//                     break;
//                 case SkillSelection.SkillType.Heal:
//                     if(skillController.ArrayContainsValue(skillController.skills, (int)SkillSelection.SkillType.Heal) == false)
//                     {
//                         image = skillController.skillData[(int)SkillSelection.SkillType.Heal].icon;
//                         child = Instantiate(skillPrefab);
//                         child.name = "Heal";
//                     }
//                     break;
//                 case SkillSelection.SkillType.Teleport:
//                     if(skillController.ArrayContainsValue(skillController.skills, (int)SkillSelection.SkillType.Teleport) == false)
//                     {
//                         image = skillController.skillData[(int)SkillSelection.SkillType.Teleport].icon;
//                         child = Instantiate(skillPrefab);
//                         child.name = "Teleport";
//                     }
//                     break;
//                 case SkillSelection.SkillType.InstantiateMagicStorm:
//                     if(skillController.ArrayContainsValue(skillController.skills, (int)SkillSelection.SkillType.InstantiateMagicStorm) == false)
//                     {
//                         image = skillController.skillData[(int)SkillSelection.SkillType.InstantiateMagicStorm].icon;
//                         child = Instantiate(skillPrefab);
//                         child.name = "InstantiateMagicStorm";
//                     }
//                     break;
//                 case SkillSelection.SkillType.Confuse:
//                     if(skillController.ArrayContainsValue(skillController.skills, (int)SkillSelection.SkillType.Confuse) == false)
//                     {
//                         image = skillController.skillData[(int)SkillSelection.SkillType.Confuse].icon;
//                         child = Instantiate(skillPrefab);
//                         child.name = "Confuse";
//                     }
//                     break;
//                 case SkillSelection.SkillType.TripleSlash:
//                     if(skillController.ArrayContainsValue(skillController.skills, (int)SkillSelection.SkillType.TripleSlash) == false)
//                     {
//                         image = skillController.skillData[(int)SkillSelection.SkillType.TripleSlash].icon;
//                         child = Instantiate(skillPrefab);
//                         child.name = "TripleSlash";
//                     }
//                     break;
//                 case SkillSelection.SkillType.AreaAttack:
//                     if(skillController.ArrayContainsValue(skillController.skills, (int)SkillSelection.SkillType.AreaAttack) == false)
//                     {
//                         image = skillController.skillData[(int)SkillSelection.SkillType.AreaAttack].icon;
//                         child = Instantiate(skillPrefab);
//                         child.name = "AreaAttack";
//                     }
//                     break;
//                 case SkillSelection.SkillType.LifeSteal:
//                     if(skillController.ArrayContainsValue(skillController.skills, (int)SkillSelection.SkillType.LifeSteal) == false)
//                     {
//                         image = skillController.skillData[(int)SkillSelection.SkillType.LifeSteal].icon;
//                         child = Instantiate(skillPrefab);
//                         child.name = "LifeSteal";
//                     }
//                     break;
//                 default:
//                     Debug.LogError("Unknown skill type!");
//                     break;
//             }

//             if (child != null)
//             {
//                 // 자식 게임 오브젝트의 부모를 현재 게임 오브젝트로 설정
//                 child.transform.SetParent(transform);

//                 // 자식 게임 오브젝트의 로컬 포지션을 (0,0,0)으로 설정
//                 child.transform.localPosition = Vector3.zero;

//                 // 자식 게임 오브젝트의 이미지 설정
//                 child.GetComponent<Image>().sprite = image;
//             }
//             else
//             {
//                 Destroy(gameObject);
//             }
            
//             // Time.timeScale = 1f; // 게임 재개
//             // UIManager.instance.CloseSkillSelectPanel();
//         }
//     }
// }