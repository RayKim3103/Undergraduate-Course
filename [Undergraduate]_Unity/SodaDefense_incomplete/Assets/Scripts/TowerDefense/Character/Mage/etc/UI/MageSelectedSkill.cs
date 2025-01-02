using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MageSelectedSkill : MonoBehaviour
{
//     /* Player 정보를 담을 변수 */
//     public string playerTag = "Player"; // Player 오브젝트의 태그
//     private GameObject player;
//     private MageController playerController;

//     public int skillIndex = 0;

//     /* Skill 이름들 */
//     /* MageSkillSlot 스크립트에서 정한 이름을 찾음 */
//     private string[] skillNames = 
//     {
//         "Summon", "WideRange", "Curse", "Ice", 
//         "ManaShield", "Heal", "Teleport", "InstantiateMagicStorm", "Confuse"
//     };

//     /* SkillSets이름에 따라 skill배열의 index다름 */
//     private Dictionary<string, int> skillSetIndices = new Dictionary<string, int>
//     {
//         { "SkillSets1", 0 },
//         { "SkillSets2", 1 },
//         { "SkillSets3", 2 },
//         { "SkillSets4", 3 },
//         { "SkillSets5", 4 }
//     };



//     private void Awake() 
//     {
//         player = GameObject.FindWithTag(playerTag);
//         playerController = player.GetComponent<MageController>();
//     }

//     // Start is called before the first frame update
//     void Start()
//     {
//         VerifySkillSets();
//     }

//     // Update is called once per frame
//     void Update()
//     {
//         if(playerController != null)
//         {
//             SelectSkill();

//             /*****************************************************************************/

//             if(!playerController.ArrayContainsValue(playerController.skills, 1))
//             {
//                 player.GetComponent<Summon>().ReleaseSkill();
//             }
//             if(!playerController.ArrayContainsValue(playerController.skills, 5))
//             {
//                 player.GetComponent<ManaShield>().ReleaseSkill();
//             }   
//         }
//     }

//     /* 특정 이름의 자식 게임 오브젝트가 존재하는지 확인 */
//     void SelectSkill()
//     {
//         for (int i = 0; i < skillNames.Length; i++)
//         {
//             if (transform.Find(skillNames[i]))
//             {
//                 playerController.skills[skillIndex] = i + 1;
// #if DEBUG_MODE
//                 Debug.Log($"{skillNames[i]} Skill Selected by Slot! / SkillIndex: {skillIndex}");
// #endif
//                 return;
//             }
//         }

//         playerController.skills[skillIndex] = 0;
//         // Debug.Log("SkillIndex: " + skillIndex + " / now it is 0");
//     }

//     /* 딕셔너리의 TryGetValue 메서드를 사용하여 gameObject.name에 해당하는 인덱스를 찾고, skillIndex에 할당 */
//     private void VerifySkillSets()
//     {
//         if (skillSetIndices.TryGetValue(gameObject.name, out skillIndex))
//         {
//             Debug.Log($"{gameObject.name} Index: {skillIndex}");
//         }
//         else
//         {
//             Debug.LogError("Invalid SkillSet name: " + gameObject.name);
//         }
//     }
}
