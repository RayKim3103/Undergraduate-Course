using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
public class InGameUIManager : MonoBehaviour
{
    // singleton 디자인 패턴
    // 아래와 같이 구성
    // 다른 class에서 GameManager.instance.(어떤 동작) 과 같이 동작함
    public static InGameUIManager instance = null;

    [SerializeField]
    private TextMeshProUGUI[] statTexts; // UI에 표시할 Stats

    [SerializeField]
    private TextMeshProUGUI roundText; // 현재 round를 표시할 Text
    public float roundDuration = 5.0f;
    private WaitForSecondsRealtime waitReal;
    private WaitForSeconds wait;

    /* GameStart & GameOver 관련 변수 */
    [SerializeField]
    private GameObject gameStartPanel; // 게임 시작 패널

    [SerializeField]
    private GameObject characterSelectPanel; // 캐릭터 선택창

    [SerializeField]
    private GameObject gameOverPanel; // 게임 오버 패널
    
    [HideInInspector]
    public bool isGameOver = false;

    [SerializeField]
    private GameObject startObject; // Start 게임 UI

    [SerializeField]
    private GameObject inGameObject; // 인 게임 오브젝트

    /* Round 패널 띄우는 변수 */
    public GameObject round; 

    /* Stat 패널 띄우는 변수 */
    public GameObject stat; 

    /* CardSet 패널 띄우는 변수 */
    public GameObject skillCardSet; // 스킬 셋

    /* 버튼입력 안할 시 자동으로 선택하게 하는 변수 */
    public Button[] buttons; // 자동 클릭할 버튼 배열

    /** Player LEVEL이 일정수준 도달 시 Skill Selection 패널 띄우는 변수 **/
    public GameObject skillSelection; // 스킬 선택

    /* Tower를 선택할 수 있는 패널을 띄우는 변수 */
    public GameObject towerSelection; // 타워 선택

    public GameObject towerSelectButtons; // 타워 설치 위치의 버튼들을 띄우는 변수
    public Image[] towerImages;
    public Sprite[] towerSourceImages;
    public Text[] towerTexts;
    public Text[] coinAndDiamondsTexts;

    /*******************************************************************/

    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private BaseCharacter playerController;

    // Awake: start보다 더 빠르게 실행되는 함수(메소드)
    void Awake() 
    {
        // 싱글톤 인스턴스 설정
        if (instance == null)    
        {
            // GameManaager를 instance에 넣어줌
            instance = this;
        }
        player = GameObject.FindWithTag(playerTag);

        // Round 패널 띄울 때 사용
        waitReal = new WaitForSecondsRealtime(roundDuration);
        wait = new WaitForSeconds(roundDuration);
    }

    // Start is called before the first frame update
    void Start()
    {
        /* UI 다 숨김 */ //-> 사실 Active(false)로 시작할 것이라 상관 X
        startObject.transform.localScale = Vector3.zero;
        // startObject.SetActive(false);
        Time.timeScale = 0f; // 임시 코드
    }

    // Update is called once per frame
    void Update()
    {
        // playerController가 변동하기에 Update에 넣음 -> 임시 코드, playerController가 변동되는 시점 포착해서 거기에 넣을 것
        playerController = player.GetComponent<BaseCharacter>();
        if(playerController != null)
        {
            ShowCoinsAndDiamonds();
        }
        ShowSelectableTowers();
    }

    /* GameStart 와 GameEnd */

    public void GameStart()
    {
        gameStartPanel.SetActive(false);
        characterSelectPanel.SetActive(true);
    }

    public void CharacterSelect()
    {
        characterSelectPanel.SetActive(false);
        inGameObject.SetActive(true);
        // startObject.SetActive(true);
        startObject.transform.localScale = Vector3.one;
        player.transform.localScale = Vector3.one;
        
        /* 수정된 부분 */
        // 조이스틱 GamePad활성화 때문에 (AWAKE보다는 늦게 실행되야 함) 
        player.SetActive(false);
        player.SetActive(true);
        Time.timeScale = 1f;
    }

    public void GameOver()
    {
        isGameOver = true;
        // 'Enemy' 태그를 가진 모든 게임 오브젝트를 찾음 -> 각 라운드 Enemy 초기화 (게임 끝났으니)
        GameObject[] enemies = GameObject.FindGameObjectsWithTag("Enemy");

        // 찾은 모든 게임 오브젝트를 파괴
        foreach (GameObject enemy in enemies)
        {
            Destroy(enemy);
        }

        // 게임이 끝났으므로, 시작 시 고른 타워들 초기화
        GameManager.Instance.purchasedTowers = new List<GameManager.Tower>();
        GameManager.Instance.towerUsed = 0;
        // 얻은 다이아몬드들 게임 매니저에 저장
        PlayerPrefs.SetInt(GameManager.Instance.diamondName, GameManager.Instance.diamonds + playerController.diamonds);
        GameManager.Instance.diamonds = PlayerPrefs.GetInt(GameManager.Instance.diamondName);

        StartCoroutine(GameOverRoutine());
    }

    IEnumerator GameOverRoutine()
    {
        yield return new WaitForSeconds(0.5f);

        inGameObject.SetActive(false);
        startObject.SetActive(false);
        // startObject.transform.localScale = Vector3.zero;
        gameOverPanel.SetActive(true);
        Time.timeScale = 0f;
    }

    public void GameRetry()
    {
        isGameOver = false;
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

    /* Stats UI */

    public void ShowStats()
    {
        // UI에 text를 적용
        statTexts[0].SetText(playerController.coin.ToString());
        statTexts[1].SetText(playerController.hp.ToString());
        statTexts[2].SetText(playerController.damage.ToString());
        statTexts[3].SetText(playerController.attackSpeed.ToString());
        statTexts[4].SetText(playerController.moveSpeed.ToString());
        statTexts[5].SetText(playerController.statPoints.ToString());
    }

    public void RoundCntText(int roundCnt)
    {
        // UI에 Round 수 text를 적용
        roundText.SetText("Round " + roundCnt.ToString());
    }

    // Stat 패널 표시 & 닫기 메서드
    public void ShowStatPanel()
    {
        Time.timeScale = 0f;
        stat.SetActive(true);
    }
    public void CloseStatPanel()
    {
        stat.SetActive(false);
        if(!stat.activeSelf && !skillCardSet.activeSelf && !skillSelection.activeSelf && !towerSelection.activeSelf)
        {
            Time.timeScale = 1f;
        }
    }

    /* Round 패널 표시 & 닫기 메서드 */
    public void ShowRoundPanel()
    {
        StartCoroutine(NoticeRound());
    }
    
    IEnumerator NoticeRound()
    {
        round.SetActive(true);

        // Start 게임 오브젝트가 게임화면에 안 보일 때 Time.time에 영향 받음
        if(startObject.transform.localScale != Vector3.one)
            yield return wait;
        // Start 게임 오브젝트가 게임화면에 보일 때 Time.time에 영향 안받음
        else
            yield return waitReal;

        round.SetActive(false);
    }

    /* SkillCardSet 패널 표시 & 닫기 메서드 */
    public void ShowSkillCardSetPanel()
    {
        Time.timeScale = 0f;
        skillCardSet.SetActive(true);
    }
    public void CloseSkillCardSetPanel()
    {
        skillCardSet.SetActive(false);
        if(!stat.activeSelf && !skillCardSet.activeSelf && !skillSelection.activeSelf && !towerSelection.activeSelf)
        {
            Time.timeScale = 1f;
        }
    }


    // Skill Select 패널 표시 & 닫기 메서드
    public void ShowSkillSelectPanel()
    {
        Time.timeScale = 0f;
        skillSelection.SetActive(true);
    }
    public void CloseSkillSelectPanel()
    {
        skillSelection.SetActive(false);
        if(!stat.activeSelf && !skillCardSet.activeSelf && !skillSelection.activeSelf && !towerSelection.activeSelf)
        {
            Time.timeScale = 1f;
        }
    }

    /* Tower Select Panel 열기 */
    public void ShowTowerSelectPanel()
    {
        Time.timeScale = 0f;
        towerSelection.SetActive(true);
    }
    public void CloseTowerSelectPanel()
    {
        towerSelection.SetActive(false);
        if(!stat.activeSelf && !skillCardSet.activeSelf && !skillSelection.activeSelf && !towerSelection.activeSelf)
            Time.timeScale = 1f;       
    }

    // Tower를 설치할 수 있는 버튼들 표기하기, 여기서는 Time.time은 건들 필요 X
    public void ShowTowerSelectButtons()
    {
        towerSelectButtons.SetActive(true);
    }
    public void CloseTowerSelectButtons()
    {
        towerSelectButtons.SetActive(false);
    }

    // 선택할 수 있는 Tower 표시
    public void ShowSelectableTowers()
    {
        // towerImages 배열과 towerTexts 배열 중 더 작은 길이를 구합니다.
        int minLength = Mathf.Min(GameManager.Instance.purchasedTowers.Count, towerImages.Length, towerTexts.Length);

        for (int i = 0; i < minLength; i++)
        {
            // 각 towerImages 요소에 sprite 할당
            towerImages[i].sprite = towerSourceImages[(int)GameManager.Instance.purchasedTowers[i]];

            // 각 towerTexts 요소에 purchasedTowers 값을 문자열로 할당
            towerTexts[i].text = GameManager.Instance.purchasedTowers[i].ToString();
        }

        // 선택한 타워가 없을 때
        if (minLength == 0)
        {
            // Basic Tower를 기본제공
            towerImages[0].sprite = towerSourceImages[1];
            towerTexts[0].text = "Basic Tower";
        }
    }

    // 얻은 코인과 다이아몬드 표시
    public void ShowCoinsAndDiamonds()
    {
        coinAndDiamondsTexts[0].text = playerController.coin.ToString();
        coinAndDiamondsTexts[1].text = playerController.diamonds.ToString();

    }
}
