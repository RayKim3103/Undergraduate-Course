using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class SkillSelection : MonoBehaviour
{
    /* 1. 스킬 enum 추가 -> 2번으로 이동 */ /* Skill을 만들 때 ISkill 인터페이스를 상속받았는지 확인 -> 모르면 톡 해줘요 */
    public enum SkillType
    {
        Basic,
        SummonAttack,
        WideRangeAttack,
        CurseAttack,
        IceAttack,
        ManaShield,
        Heal,
        Teleport,
        InstantiateMagicStorm,
        Confuse,
        TripleSlash,
        AreaAttack,
        LifeSteal,
        Boomerang,
        SphereAttack,
        CounterAttack
    }

    /* Skill UI */
    public Transform skillPanel;
    public GameObject skillSlotPrefab;
    [SerializeField] private TextMeshProUGUI[] skillTexts; // UI에 표시할 Skills
    [SerializeField] private Text[] skillShopTexts;

    private List<SkillType> selectableSkills = new List<SkillType>(); // 선택 가능한 스킬 리스트
    public SkillType selectedSkill = SkillType.Basic; // 기본값으로 Basic 선택

    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private BaseCharacter baseCharacter;
    private SkillController skillController;

    /* Reroll 관련 변수 */
    public int rerollCost = 2;

    private void Awake()
    {
        skillController = gameObject.GetComponent<SkillController>();
        baseCharacter = GetComponentInParent<BaseCharacter>();
    }

    private void Start()
    {
        InitializeUI();
        InitializeSkills();
        ShuffleSkills();
        InitializeSkillController();
        ShowSkills();
    }

    private void Update() 
    {
        // SkillSelection을 처음에 Player의 자식으로 안 뒀기에 에러 나올수 있음
        if (baseCharacter == null)
        {
            baseCharacter = GetComponentInParent<BaseCharacter>();
        }
        ShowSkills();
    }

    private void InitializeUI()
    {
        InGameUIManager.instance.ShowSkillSelectPanel();

        for (int i = 0; i < 3; i++)
        {
            int index = i; // Capture the loop variable
            InGameUIManager.instance.buttons[i].GetComponent<Button>().onClick.RemoveAllListeners();
            // 람다 표현식을 사용하는 이유는 C#의 클로저(closure) 동작 방식 때문
            // 클로저는 익명 함수 또는 람다 표현식이 생성될 때 그 함수가 정의된 스코프에 있는 변수들을 캡처하는 기능을 제공
            InGameUIManager.instance.buttons[i].GetComponent<Button>().onClick.AddListener(() => OnClickButton(index));
        }

        InGameUIManager.instance.CloseSkillSelectPanel();

        for (int i = 3; i < 8; i++)
        {
            int index = i - 3; // Adjust for skill shop buttons
            InGameUIManager.instance.buttons[i].GetComponent<Button>().onClick.RemoveAllListeners();
            if (i == 7)
                InGameUIManager.instance.buttons[i].GetComponent<Button>().onClick.AddListener(OnClickReroll);
            else
                InGameUIManager.instance.buttons[i].GetComponent<Button>().onClick.AddListener(() => OnClickButton(index));
        }
    }

    /* 2. 스킬 추가 */ /* 2번 했으면, SkillController 스크립트로 이동 */
    private void InitializeSkills()
    {
        selectableSkills.AddRange(new SkillType[]
        {
            SkillType.SummonAttack,
            SkillType.WideRangeAttack,
            SkillType.CurseAttack,
            SkillType.IceAttack,
            SkillType.ManaShield,
            SkillType.Heal,
            SkillType.Teleport,
            SkillType.InstantiateMagicStorm,
            SkillType.Confuse,
            SkillType.TripleSlash,
            SkillType.AreaAttack,
            SkillType.LifeSteal,
            SkillType.Boomerang,
            SkillType.SphereAttack,
            SkillType.CounterAttack
        });
    }

    private void InitializeSkillController()
    {
        skillController.skills = new int[selectableSkills.Count + 1];
        skillController.skillLevel = new int[selectableSkills.Count + 1];
    }

    private void ShowSkills()
    {
        for (int i = 0; i < 3; i++)
        {
            skillTexts[i].SetText(selectableSkills[i].ToString());
        }

        for (int i = 0; i < 4; i++)
        {
            skillShopTexts[i].text = selectableSkills[i].ToString();
        }
    }

    private void ShuffleSkills()
    {
        // Fisher-Yates 알고리즘
        for (int i = selectableSkills.Count - 1; i > 0; i--)
        {
            int randomIndex = Random.Range(0, i + 1);
            SkillType temp = selectableSkills[randomIndex];
            selectableSkills[randomIndex] = selectableSkills[i];
            selectableSkills[i] = temp;
        }
    }

    public void SelectSkill(int skillIndex)
    {
        selectedSkill = selectableSkills[skillIndex];
        AddSkill(selectedSkill);
        skillController.SetSelectedSkill(selectedSkill);

        Debug.Log("Selected skill: " + selectedSkill);
    }

    public void AddSkill(SkillType newSkill)
    {
        GameObject newSlot = Instantiate(skillSlotPrefab, skillPanel);
        newSlot.GetComponent<SkillSlot>().Setup(newSkill);
    }

    public void OnClickReroll()
    {
        if (baseCharacter.coin >= rerollCost)
        {
            baseCharacter.coin -= rerollCost;
            ShuffleSkills();
        }
    }

    public void OnClickButton(int index)
    {
        if (baseCharacter.coin >= skillController.skillData[(int)selectableSkills[index]].needCoin)
        {
            baseCharacter.coin -= skillController.skillData[(int)selectableSkills[index]].needCoin;
            SelectSkill(index);
            ShuffleSkills();
        }
    }
}


// // #define DEBUG_MODE

// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine;
// using TMPro;
// using UnityEngine.UI;
// using System.Data.Common;

// public class SkillSelection : MonoBehaviour
// {
//     // 열거형 변수 선언
//     public enum SkillType
//     {
//         Basic,
//         SummonAttack,
//         WideRangeAttack,
//         CurseAttack,
//         IceAttack,
//         ManaShield,
//         Heal,
//         Teleport,
//         // IncreaseManapower,
//         // IncreaeAttackSpeed,
//         InstantiateMagicStorm,
//         Confuse,
//         TripleSlash,
//         // MoveSpeedEnhance,
//         AreaAttack,
//         LifeSteal,
//         Null_1,
//         Null_2,
//         Null_3
//     }

    
//     /* Skill UI */
//     public Transform skillPanel;
//     public GameObject skillSlotPrefab;
//     //public List<SkillType> skills = new List<SkillType>();

//     [SerializeField]
//     private TextMeshProUGUI[] skillTexts; // UI에 표시할 Skills
    
//     [SerializeField]
//     private Text[] skillShopTexts;

//     /*************************************************************/
    
//     private List<SkillType> selectableSkills = new List<SkillType>(); // 선택 가능한 스킬 리스트
//     public SkillType selectedSkill = SkillType.Basic; // 기본값으로 Basic 선택

//     /* Player 정보를 담을 변수 */
//     public string playerTag = "Player"; // Player 오브젝트의 태그
//     private GameObject player;
//     private BaseCharacter baseCharacter;
//     private SkillController skillReceiver;
    

//     /* Reroll 관련 변수 */
//     public float RerollCost = 2.0f;

//     private void Awake()
//     {
//         // player = GameObject.FindWithTag(playerTag);
//         skillReceiver = gameObject.GetComponent<SkillController>();
        
//         baseCharacter = GetComponentInParent<BaseCharacter>();
//     }

//     void Start()
//     {
//         /* Skill 증강 (초기구현한 것) 버튼 */
//         UIManager.instance.ShowSkillSelectPanel();
//         UIManager.instance.buttons[0].GetComponent<Button>().onClick.RemoveAllListeners();
//         UIManager.instance.buttons[0].GetComponent<Button>().onClick.AddListener(OnClickButton0);
//         UIManager.instance.buttons[1].GetComponent<Button>().onClick.RemoveAllListeners();
//         UIManager.instance.buttons[1].GetComponent<Button>().onClick.AddListener(OnClickButton1);
//         UIManager.instance.buttons[2].GetComponent<Button>().onClick.RemoveAllListeners();
//         UIManager.instance.buttons[2].GetComponent<Button>().onClick.AddListener(OnClickButton2);
//         UIManager.instance.buttons[3].GetComponent<Button>().onClick.RemoveAllListeners();
//         UIManager.instance.CloseSkillSelectPanel();

//         /* Skill Shop Button */
//         UIManager.instance.buttons[3].GetComponent<Button>().onClick.RemoveAllListeners();
//         UIManager.instance.buttons[3].GetComponent<Button>().onClick.AddListener(OnClickButton0);
//         UIManager.instance.buttons[4].GetComponent<Button>().onClick.RemoveAllListeners();
//         UIManager.instance.buttons[4].GetComponent<Button>().onClick.AddListener(OnClickButton1);
//         UIManager.instance.buttons[5].GetComponent<Button>().onClick.RemoveAllListeners();
//         UIManager.instance.buttons[5].GetComponent<Button>().onClick.AddListener(OnClickButton2);
//         UIManager.instance.buttons[6].GetComponent<Button>().onClick.RemoveAllListeners();
//         UIManager.instance.buttons[6].GetComponent<Button>().onClick.AddListener(OnClickButton3);
//         UIManager.instance.buttons[7].GetComponent<Button>().onClick.RemoveAllListeners();
//         UIManager.instance.buttons[7].GetComponent<Button>().onClick.AddListener(OnClickReroll);
        

//         // 모든 스킬 종류를 선택 가능한 스킬 리스트에 추가
//         selectableSkills.Add(SkillType.SummonAttack);
//         selectableSkills.Add(SkillType.WideRangeAttack);
//         selectableSkills.Add(SkillType.CurseAttack);
//         selectableSkills.Add(SkillType.IceAttack);
//         selectableSkills.Add(SkillType.ManaShield);
//         selectableSkills.Add(SkillType.Heal);
//         selectableSkills.Add(SkillType.Teleport);
//         // selectableSkills.Add(SkillType.IncreaseManapower);
//         // selectableSkills.Add(SkillType.IncreaeAttackSpeed);
//         selectableSkills.Add(SkillType.InstantiateMagicStorm);
//         selectableSkills.Add(SkillType.Confuse);
//         selectableSkills.Add(SkillType.TripleSlash);
//         selectableSkills.Add(SkillType.AreaAttack);
//         selectableSkills.Add(SkillType.LifeSteal);
//         selectableSkills.Add(SkillType.Null_1);
//         selectableSkills.Add(SkillType.Null_2);
//         selectableSkills.Add(SkillType.Null_3);

//         // 랜덤으로 스킬 섞기 (맨 처음 초기에 1번 랜덤으로 섞음)
//         ShuffleSkills();

//         // SkillController의 skills배열을 스킬의 개수의 배열을 가지도록 생성해 줌
//         skillReceiver.skills = new int [selectableSkills.Count + 1];
//         skillReceiver.skillLevel = new int [selectableSkills.Count + 1];

//         // // 디버깅용 코드
//         // for(int i = 0; i < selectableSkills.Count; i++)
//         // {
//         //     Debug.Log(i + " 번째 스킬: " + selectableSkills[i]);
//         // }
//     }

//     private void Update() 
//     {
//         ShowSkills();
//     }

//     public void ShowSkills()
//     {
//         /* Skill 증강 (초기구현한 것) 스킬 이름 TEXT */
//         skillTexts[0].SetText(selectableSkills[0].ToString());
//         skillTexts[1].SetText(selectableSkills[1].ToString());
//         skillTexts[2].SetText(selectableSkills[2].ToString());

//         /* SkillShop 스킬 이름 TEXT */
//         skillShopTexts[0].text= selectableSkills[0].ToString();
//         skillShopTexts[1].text= selectableSkills[1].ToString();
//         skillShopTexts[2].text= selectableSkills[2].ToString();
//         skillShopTexts[3].text= selectableSkills[3].ToString();
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
//         AddSkill(selectedSkill);
//         skillReceiver.SetSelectedSkill(selectedSkill);
        
//         // Debug 용 출력문
//         Debug.Log("Selected skill: " + selectedSkill);
//     }
//     /******************************************************************************************/


//     public void AddSkill(SkillType newSkill)
//     {
//         //skills.Add(newSkill);
//         GameObject newSlot = Instantiate(skillSlotPrefab, skillPanel);
//         newSlot.GetComponent<SkillSlot>().Setup(newSkill);
//     }

    
//     public void OnClickReroll()
//     {
//         if(baseCharacter.coin >= RerollCost)
//         {
//             baseCharacter.coin -= RerollCost;
//             ShuffleSkills();
//         }
//     }

//     // 예시에서는 각 Button에 연결될 Method
//     // 실제로는 UI의 Button 클릭 이벤트 설정에서 이 Method들을 연결해야 함
//     public void OnClickButton0()
//     {
//         if(baseCharacter.coin >= skillReceiver.skillData[(int)selectableSkills[0]].needCoin)
//         {
//             baseCharacter.coin -= skillReceiver.skillData[(int)selectableSkills[0]].needCoin;
//             SelectSkill(0);
//             ShuffleSkills();
//         }
//     }

//     public void OnClickButton1()
//     {
//         if(baseCharacter.coin >= skillReceiver.skillData[(int)selectableSkills[1]].needCoin)
//         {
//             baseCharacter.coin -= skillReceiver.skillData[(int)selectableSkills[1]].needCoin;
//             SelectSkill(1);
//             ShuffleSkills();
//         }
//     }

//     public void OnClickButton2()
//     {
//         if(baseCharacter.coin >= skillReceiver.skillData[(int)selectableSkills[2]].needCoin)
//         {
//             baseCharacter.coin -= skillReceiver.skillData[(int)selectableSkills[2]].needCoin;
//             SelectSkill(2);
//             ShuffleSkills();
//         }
//     }

//     public void OnClickButton3()
//     {
//         if(baseCharacter.coin >= skillReceiver.skillData[(int)selectableSkills[3]].needCoin)
//         {
//             baseCharacter.coin -= skillReceiver.skillData[(int)selectableSkills[3]].needCoin;
//             SelectSkill(3);
//             ShuffleSkills();
//         }
//     }
// }
