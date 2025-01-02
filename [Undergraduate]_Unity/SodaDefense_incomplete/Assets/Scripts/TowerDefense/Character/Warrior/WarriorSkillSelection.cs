// #define DEBUG_MODE

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class WarriorSkillSelection : MonoBehaviour
{
//     // 열거형 변수 선언
//     public enum SkillType
//     {
//         Basic,
//         TripleSlash,
//         // MoveSpeedEnhance,
//         Shield,
//         // AttackBoost,
//         AreaAttack,
//         // WarriorIncreaseAttackSpeed,
//         LifeSteal,
//         Null_4,
//         Null_5,
//         Null_6,
//         Null_7,
//         Null_8,
//         Null_9,
//         Null_10,
//         Null_11
//     }

//     /* Skill UI */
//     public Transform skillPanel;
//     public GameObject skillSlotPrefab;
//     public List<SkillType> skills = new List<SkillType>();

//     [SerializeField]
//     private TextMeshProUGUI[] skillTexts; // UI에 표시할 Skills

//     private List<SkillType> selectableSkills = new List<SkillType>(); // 선택 가능한 스킬 리스트
//     public SkillType selectedSkill = SkillType.Basic; // 기본값으로 WideRangeAttack 선택

//     //public string playerTag = "Player"; // Player 오브젝트의 태그
//     // private GameObject player;
//     private WarriorController skillReceiver;

//     private void Awake() 
//     {
//         //player = GameObject.FindWithTag(playerTag);

//         skillReceiver = gameObject.GetComponent<WarriorController>();
//     }

//     void Start()
//     {
//         UIManager.instance.ShowSkillSelectPanel();
//         UIManager.instance.buttons[0].GetComponent<Button>().onClick.RemoveAllListeners();
//         UIManager.instance.buttons[0].GetComponent<Button>().onClick.AddListener(OnClickButton0);
//         UIManager.instance.buttons[1].GetComponent<Button>().onClick.RemoveAllListeners();
//         UIManager.instance.buttons[1].GetComponent<Button>().onClick.AddListener(OnClickButton1);
//         UIManager.instance.buttons[2].GetComponent<Button>().onClick.RemoveAllListeners();
//         UIManager.instance.buttons[2].GetComponent<Button>().onClick.AddListener(OnClickButton2);
//         UIManager.instance.CloseSkillSelectPanel();

//         // 모든 스킬 종류를 선택 가능한 스킬 리스트에 추가
//         selectableSkills.Add(SkillType.TripleSlash);
//         selectableSkills.Add(SkillType.Shield);
//         // selectableSkills.Add(SkillType.MoveSpeedEnhance);
//         // selectableSkills.Add(SkillType.AttackBoost);
//         selectableSkills.Add(SkillType.AreaAttack);
//         // selectableSkills.Add(SkillType.WarriorIncreaseAttackSpeed);
//         selectableSkills.Add(SkillType.LifeSteal);
//         selectableSkills.Add(SkillType.Null_4);
//         selectableSkills.Add(SkillType.Null_5);
//         selectableSkills.Add(SkillType.Null_6);
//         selectableSkills.Add(SkillType.Null_7);
//         selectableSkills.Add(SkillType.Null_8);
//         selectableSkills.Add(SkillType.Null_9);
//         selectableSkills.Add(SkillType.Null_10);
//         selectableSkills.Add(SkillType.Null_11);

//         // 랜덤으로 스킬 섞기
//         ShuffleSkills();

//         // 디버깅용 코드
//         for(int i = 0; i < selectableSkills.Count; i++)
//         {   
//             Debug.Log(i + " 번째 스킬: " + selectableSkills[i]);
//         }
//     }

//     private void Update() 
//     {
//         ShowSkills();    
//     }

//     public void ShowSkills()
//     {
//         for(int i = 0; i < 5; i++)
//         {
//             if(0 <= skillReceiver.level - skillReceiver.GetSkillLevel[i] && skillReceiver.level - skillReceiver.GetSkillLevel[i] < 5)
//             {
//                 // UI에 text를 적용
//                 skillTexts[0].SetText(selectableSkills[i*3].ToString());
//                 skillTexts[1].SetText(selectableSkills[i*3 + 1].ToString());
//                 skillTexts[2].SetText(selectableSkills[i*3 + 2].ToString());
//             }
//         }
//     }


//     private void ShuffleSkills()
//     {
//         // 리스트를 섞기 위해 Fisher-Yates 알고리즘 사용, 랜덤성을 위한 코드
//         for (int i = selectableSkills.Count - 1; i > 0; i--)
//         {
//             int randomIndex = Random.Range(0, i + 1);
//             SkillType temp = selectableSkills[randomIndex];
//             selectableSkills[randomIndex] = selectableSkills[i];
//             selectableSkills[i] = temp;
//         }
//     }

//     /******************** Player에 있는 skills 배열에 값을 주는 함수를 실행 **********************/
//     public void SelectSkill(int skillIndex)
//     {
//         selectedSkill = selectableSkills[skillIndex];
//         // skillReceiver.SetSelectedSkill(selectedSkill);
//         AddSkill(selectedSkill);
//         // Debug 용 출력문
//         Debug.Log("Selected skill: " + selectedSkill);
//     }
//     /******************************************************************************************/

//     public void AddSkill(SkillType newSkill)
//     {
//         skills.Add(newSkill);
//         GameObject newSlot = Instantiate(skillSlotPrefab, skillPanel);
//         newSlot.GetComponent<WarriorSkillSlot>().Setup(newSkill);
//     }

//     /* GetSkillLevel이 5렙마다 call되고, 25렙까지 스킬을 얻으므로 skill 종류가 15개 이상 있어야 ERROR 발생 X */
//     /* 현재 스킬 종류: 10 개 */

//     // 예시에서는 각 Button에 연결될 Method
//     // 실제로는 UI의 Button 클릭 이벤트 설정에서 이 Method들을 연결해야 함
//     public void OnClickButton0()
//     {
//         for(int i = 0; i < 5; i++)
//         {
// #if DEBUG_MODE
//             Debug.Log("1. Player LEVEL: " + skillReceiver.level);
// #endif
//             if(0 <= skillReceiver.level - skillReceiver.GetSkillLevel[i] && skillReceiver.level - skillReceiver.GetSkillLevel[i] < 5)
//             {
// #if DEBUG_MODE
//                 Debug.Log("2. Player LEVEL: " + skillReceiver.level);
// #endif
//                 SelectSkill(i*3);
// #if DEBUG_MODE
//             Debug.Log("3. Player LEVEL: " + skillReceiver.level);
// #endif
//             }
//         }
//     }

//     public void OnClickButton1()
//     {
//         for(int i = 0; i < 5; i++)
//         {
//             if(0 <= skillReceiver.level - skillReceiver.GetSkillLevel[i] && skillReceiver.level - skillReceiver.GetSkillLevel[i] < 5)
//             {
//                 SelectSkill(i*3 + 1);
//             }
//         }
//     }

//     public void OnClickButton2()
//     {
//         for(int i = 0; i < 5; i++)
//         {
//             if(0 <= skillReceiver.level - skillReceiver.GetSkillLevel[i] && skillReceiver.level - skillReceiver.GetSkillLevel[i] < 5)
//             {
//                 SelectSkill(i*3 + 2);
//             }
//         }
//     }

//     public void DebugHelloWorld()
//     {
//         Debug.Log("***** Button이 바뀌지 않아! Hello World *****");
//     }
}
