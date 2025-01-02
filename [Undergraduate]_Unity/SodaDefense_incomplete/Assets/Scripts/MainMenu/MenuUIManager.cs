using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

public class MenuUIManager : MonoBehaviour
{
    public static MenuUIManager instance = null;

    /* GameStart & GameOver 관련 변수 */
    [SerializeField]
    private GameObject getTowerPanel; // 게임 시작 패널

    [SerializeField]
    private GameObject towerCardSetPanel; // 게임 시작 패널

    [SerializeField]
    private GameObject settingPanel; // Setting 패널
    
    [SerializeField]
    private GameObject inGameTowerPanel; // Setting 패널

    public Text textDiamonds;

    public Transform availableTowersPanel;
    public GameObject towerSlotPrefab;
    public Transform inGameTowersPanel;
    public GameObject inGameTowerSlotPrefab;

    private void Awake() 
    {
        // 싱글톤 인스턴스 설정
        if (instance == null)    
        {
            // MenuUIManager를 instance에 넣어줌
            instance = this;
        }
    }

    private void Start() 
    {
        InitiateTowerSlots();
        ShowDiamonds();
    }

    private void Update() 
    {
        
    }

    public void ShowTowerSelectPanel()
    {
        ToggleInGameTowerSetPanel();
    }

    public void GameStart()
    {
        // Scene 이름을 사용하여 Scene 로드
        string sceneName = "InGame";
        SceneManager.LoadScene(sceneName);
        Time.timeScale = 1f;
    }
    
    public void GameRetry()
    {
        // Scene 이름을 사용하여 Scene 로드
        string sceneName = "Menu";
        SceneManager.LoadScene(sceneName);
        Time.timeScale = 1f;
    }

    public void GameQuit()
    {
        // 어플리케이션 종료하는 내장함수
        Application.Quit();
    }

    
    // GetTower 패널 표시 & 닫기 메서드
    public void ToggleGetTowerPanel()
    {
        if(getTowerPanel.activeSelf)
        {
            CloseGetTowerPanel();
        }
        else
        {
            getTowerPanel.SetActive(true);
        }        
    }
    private void CloseGetTowerPanel()
    {
        // getTowerPanel 게임 오브젝트의 Transform 컴포넌트를 가져옵니다.
        Transform getTowerPanelTransform = getTowerPanel.transform;

        // 자식 게임 오브젝트의 이름을 지정하여 찾습니다.
        Transform resultTransform = getTowerPanelTransform.Find("Result");
        Transform resultTowerTransform = resultTransform.Find("ResultTower");

        // 찾은 자식 게임 오브젝트가 null이 아닌 경우에만 작업을 수행합니다.
        if (resultTowerTransform != null)
        {
            // 찾은 자식 게임 오브젝트에 대한 작업을 수행합니다.
            resultTowerTransform.gameObject.SetActive(false);
        }

        getTowerPanel.SetActive(false);
    }

    // GetTower 패널 표시 & 닫기 메서드
    public void ToggleTowerCardSetPanel()
    {
        if(towerCardSetPanel.activeSelf)
        {
            towerCardSetPanel.SetActive(false);
        }
        else
        {
            towerCardSetPanel.SetActive(true);
        }
    }

    // InGameTower 패널 표시 & 닫기 메서드
    public void ToggleInGameTowerSetPanel()
    {
        if(inGameTowerPanel.activeSelf)
        {
            inGameTowerPanel.SetActive(false);
        }
        else
        {
            inGameTowerPanel.SetActive(true);
        }
    }

    // Setting 패널 표시 & 닫기 메서드
    public void ToggleSettingPanel()
    {
        if(settingPanel.activeSelf)
        {
            settingPanel.SetActive(false);
        }
        else
        {
            settingPanel.SetActive(true);
        }
    }

    public void ShowDiamonds()
    {
        textDiamonds.text = GameManager.Instance.diamonds.ToString();
    }

    public void InitiateTowerSlots()
    {
        int cnt = 0;
        foreach (GameManager.Tower tower in GameManager.Instance.towers)
        {
            // PlayerPrefs에서 타워의 개수를 가져와서 cnt에 추가합니다
            cnt += PlayerPrefs.GetInt(tower.ToString());

            // cnt만큼 타워 슬롯을 생성합니다
            for (int i = 0; i < cnt; i++)
            {
                CreateTowerSlots(tower);
            }

            Debug.Log("GetTowerButton cnt: " + cnt);

            // 다음 타워를 위한 cnt 초기화
            cnt = 0;
        }
    }

    // 타워 슬롯을 생성하고 설정하는 함수
    public void CreateTowerSlots(GameManager.Tower tower)
    {
        // Available Towers Panel에 타워 슬롯을 생성합니다.
        GameObject newSlot = Instantiate(towerSlotPrefab, availableTowersPanel);
        newSlot.GetComponent<Image>().sprite = GameManager.Instance.sprites[(int)tower];

        // In-Game Towers Panel에 타워 슬롯을 생성하고 설정합니다.
        GameObject inGameSlot = Instantiate(inGameTowerSlotPrefab, inGameTowersPanel);
        inGameSlot.GetComponent<TowerSlot>().Setup(tower, GameManager.Instance.sprites[(int)tower]);
    }
}
