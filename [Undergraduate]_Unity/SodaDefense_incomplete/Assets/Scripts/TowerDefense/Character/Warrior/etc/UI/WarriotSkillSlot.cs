using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class WarriorSkillSlot : MonoBehaviour
{
    // public Sprite[] icons;
    // private Sprite image;

    // [SerializeField]
    // private GameObject skillPrefab;
    // public Text skillName;
    // public Text description;
    // public Button button;
    // public WarriorSkillSelection.SkillType warriorSkill;

    // GameObject child;

    // // public Vector2 childSize = new Vector2(100, 100); // 자식 UI 오브젝트의 크기

    // public void Setup(WarriorSkillSelection.SkillType newSkill)
    // {
    //     warriorSkill = newSkill;

    //     switch (warriorSkill)
    //     {
    //         case WarriorSkillSelection.SkillType.TripleSlash:
    //             image = icons[0];
    //             child = Instantiate(skillPrefab);
    //             child.name = "TripleSlash";
    //             break;
    //         case WarriorSkillSelection.SkillType.Shield:
    //             image = icons[1];
    //             child = Instantiate(skillPrefab);
    //             child.name = "Shield";
    //             break;
    //         case WarriorSkillSelection.SkillType.AreaAttack:
    //             image = icons[2];
    //             child = Instantiate(skillPrefab);
    //             child.name = "AreaAttack";
    //             break;
    //         case WarriorSkillSelection.SkillType.LifeSteal:
    //             image = icons[3];
    //             child = Instantiate(skillPrefab);
    //             child.name = "LifeSteal";
    //             break;
    //         default:
    //             Debug.LogError("Unknown skill type!");
    //             break;
    //     }

    //     // 자식 게임 오브젝트의 부모를 현재 게임 오브젝트로 설정
    //     child.transform.SetParent(transform);

    //     // 자식 게임 오브젝트의 로컬 포지션을 (0,0,0)으로 설정
    //     child.transform.localPosition = Vector3.zero;

    //     // 자식 게임 오브젝트의 이미지 설정
    //     child.GetComponent<Image>().sprite = image;

    //     // Time.timeScale = 1f; // 게임 재개
    //     UIManager.instance.CloseSkillSelectPanel();
    // }
}