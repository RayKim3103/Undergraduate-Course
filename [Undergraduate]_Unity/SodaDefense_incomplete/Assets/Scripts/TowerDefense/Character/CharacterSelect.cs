using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class CharacterSelect : MonoBehaviour
{
    public Button[] buttons; // 버튼을 에디터에서 할당하거나, 코드에서 참조를 설정합니다.
    public Sprite[] sprites;
    private SpriteRenderer spriteRenderer;
    public GameObject[] prefabToWarrior; // 할당할 프리팹
    public GameObject[] prefabToMage; // 할당할 프리팹
    public GameObject[] prefabToArcher; // 할당할 프리팹
    // public  WarriorController warriorController;
    // public  MageController mageController;
    // public  ArcherController archerController;
    // public GameObject[] allObjects;
    // public string targetNameWarrior = "Warrior";
    // public string targetNameMage = "Mage";
    // public string targetNameArcher = "Archer";

    // private void Awake() 
    // {
    //     // 모든 GameObject를 가져옵니다.
    //     allObjects = FindObjectsOfType<GameObject>();    
    // }

    void Start()
    {
        // SpriteRenderer 컴포넌트 가져오기
        spriteRenderer = GetComponent<SpriteRenderer>();
        // 버튼 클릭 이벤트에 메서드 연결
        buttons[0].onClick.AddListener(OnButtonClickWarrior);
        buttons[1].onClick.AddListener(OnButtonClickMage);
        buttons[2].onClick.AddListener(OnButtonClickArcher);
    }

    void OnButtonClickWarrior()
    {
        //BaseCharacterScriptRemove();
        WarriorController warriorController = gameObject.AddComponent<WarriorController>();
        // index순서 중요
        warriorController.attackPrefab = prefabToWarrior[0];
        warriorController.attackInstancePrefab = prefabToWarrior[1];
        spriteRenderer.sprite = sprites[0];

        InGameUIManager.instance.CharacterSelect();
    }

    void OnButtonClickMage()
    {
        //BaseCharacterScriptRemove();
        MageController mageController = gameObject.AddComponent<MageController>();
        // index순서 중요
        mageController.attackPrefab = prefabToMage[0];
        spriteRenderer.sprite = sprites[1];

        InGameUIManager.instance.CharacterSelect();
    }
    void OnButtonClickArcher()
    {
        //BaseCharacterScriptRemove();
        ArcherController archerController = gameObject.AddComponent<ArcherController>();
        // index순서 중요
        archerController.attackPrefab = prefabToArcher[0];
        spriteRenderer.sprite = sprites[2];

        InGameUIManager.instance.CharacterSelect();
    }

    void BaseCharacterScriptRemove()
    {
        // 제거하려는 스크립트를 찾아서 제거합니다.
        BaseCharacter baseCharacter = GetComponent<BaseCharacter>();

        if (baseCharacter != null)
        {
            Destroy(baseCharacter);
        }
    }

    // void OnButtonClickArcher()
    // {
    //     foreach (GameObject obj in allObjects)
    //     {
    //         // 특정 이름과 일치하는 GameObject를 활성화
    //         if (obj.name == targetNameArcher)
    //         {
    //             obj.SetActive(true);
    //         }
    //         // 일치하지 않는 GameObject를 비활성화
    //         else if (obj.name == targetNameWarrior || obj.name == targetNameMage)
    //         {
    //             obj.SetActive(false);
    //         }
    //     }
    // }
}
