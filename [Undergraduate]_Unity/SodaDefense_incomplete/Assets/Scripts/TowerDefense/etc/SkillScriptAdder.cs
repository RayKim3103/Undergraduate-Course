using UnityEngine;

public class SkillScriptAdder : MonoBehaviour
{
    [SerializeField] 
    private GameObject prefabToAssign; // Inspector에서 할당할 prefab
    
    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private MageController playerController;
    private MageBasicSkillEffect mageBasicSkillEffect;

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);        
    }

    void Start()
    {   
        // 특정 컴포넌트가 있는지 확인
        if (player.GetComponent<MageController>() != null)
        {
            Debug.Log("MageController 컴포넌트가 존재합니다.");
            // MageScript를 게임 오브젝트에 추가
            MageBasicSkillEffect mageBasicSkillEffect = gameObject.AddComponent<MageBasicSkillEffect>();
            // prefab을 MageScript의 변수에 할당
            // mageBasicSkillEffect.SetPrefab(prefabToAssign); // prefab을 설정하는 메서드 필요

        }
        else if (player.GetComponent<ArcherController>() != null)
        {
            Debug.Log("ArcherController 컴포넌트가 존재합니다.");
        }
        else if (player.GetComponent<WarriorController>() != null)
        {
            Debug.Log("WarriorController 컴포넌트가 존재합니다.");
            // WarriorScript를 게임 오브젝트에 추가
            // WarriorBasicSkillEffect warriorBasicSkillEffect = gameObject.AddComponent<WarriorBasicSkillEffect>();

            // // prefab을 WarriorScript의 변수에 할당 -> 나중에 prefab을 Enemy한테 설정해야 할 일 있으면 주석풀고 코드 작성하세요 님들
            // warriorBasicSkillEffect.SetPrefab(prefabToAssign); // prefab을 설정하는 메서드 필요
        }
        else
        {
            Debug.Log("컴포넌트가 존재하지 않습니다.");
        }


        // 또는, 기존의 게임 오브젝트를 참조하여 추가할 수도 있습니다.
        // GameObject existingGameObject = GameObject.Find("ExistingGameObject");
        // existingGameObject.AddComponent<MyScript>();
    }
}