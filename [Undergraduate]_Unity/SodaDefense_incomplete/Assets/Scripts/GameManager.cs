using System.Collections.Generic;
using System;
using UnityEngine;
using UnityEngine.UI;

public class GameManager : Singleton<GameManager>
{
    // 타워들
    public enum Tower { AreaTower, BasicTower, IceTower, PoisonTower, RapidTower, SlowTower, SniperTower }
    public Tower[] towers;
    public Sprite[] sprites; // 랜덤으로 표시할 스프라이트 배열

    public List<Tower> purchasedTowers; // 구입한 아이템 리스트
    public int maxTower = 3;   // 인 게임에 3개의 타워까지만 들고 갈 수 있음
    public int towerUsed = 0;

    // 보석들 (타워를 살 수 있는 코인)
    public int diamonds = 0;
    public string diamondName = "diamonds";

    protected override void Awake() 
    {
        base.Awake();
        // Tower타입으로 변환하여 towers라는 instance안에 Tower의 정보 들어감
        towers = (Tower[])Enum.GetValues(typeof(Tower));

        // MyData라는 key가 저장되어 있지 않으면 실행
        // 게임을 다시 시작해도, MyData라는 key가 저장되어 있기에, Init 실행 더 이상X
        if (!PlayerPrefs.HasKey("MyData")) 
        {
            Init();
        }

        // 저장되어 있던 diamonds 불러옴
        diamonds = (int)PlayerPrefs.GetInt(diamondName);
        Debug.Log("Player Prefs Diamonds: " + PlayerPrefs.GetInt(diamondName));
        Debug.Log("Player Prefs Diamonds Next: " + PlayerPrefs.GetInt(diamondName));
        Debug.Log("GameManager Diamonds: " + diamonds);
    }

    void Init()
    {
        // PlayerPrefs: 간단한 저장 기능을 제공하는 유니티 제공 클래스
        // 아래코드는 MyData라는 key값에 1이 저장되는 예시 코드
        PlayerPrefs.SetInt("MyData", 1);

        // tower에 따라 저장, 0으로 초기화 작업
        foreach (Tower tower in towers) 
        {
            PlayerPrefs.SetInt(tower.ToString(), 0);
        }

        // 다이아몬드도 0으로 초기화
        PlayerPrefs.SetInt(diamondName, 0);
    }

    private void Start()
    {
        purchasedTowers = new List<Tower>();
        // InitiateTowerSlots();
    }

    // 아이템을 구입할 때 호출되는 메서드
    public void PurchaseTower(Tower tower)
    {
        if (towerUsed < maxTower) // !purchasedTowers.Contains(tower) && 
        {
            purchasedTowers.Add(tower);
            towerUsed += 1;
        }
    }

    // 아이템을 확인할 때 호출되는 메서드
    public bool HasTower(Tower tower)
    {
        return purchasedTowers.Contains(tower);
    }

    // 아이템을 판매할 때 호출되는 메서드
    public void SellTower(Tower tower)
    {
        if (purchasedTowers.Contains(tower))
        {
            purchasedTowers.Remove(tower);
            towerUsed -= 1;
            Debug.Log($"Sold tower: {tower}. Remaining slots: {maxTower-towerUsed}");
        }
        else
        {
            Debug.Log($"Cannot sell tower: {tower}. Not owned.");
        }
    }

    
    // public void InitiateTowerSlots()
    // {
    //     int cnt = 0;
    //     foreach (Tower tower in towers)
    //     {
    //         // PlayerPrefs에서 타워의 개수를 가져와서 cnt에 추가합니다
    //         cnt += PlayerPrefs.GetInt(tower.ToString());

    //         // cnt만큼 타워 슬롯을 생성합니다
    //         for (int i = 0; i < cnt; i++)
    //         {
    //             CreateTowerSlots(tower);
    //         }

    //         Debug.Log("GetTowerButton cnt: " + cnt);

    //         // 다음 타워를 위한 cnt 초기화
    //         cnt = 0;
    //     }
    // }

    // // 타워 슬롯을 생성하고 설정하는 함수
    // public void CreateTowerSlots(Tower tower)
    // {
    //     // Available Towers Panel에 타워 슬롯을 생성합니다.
    //     GameObject newSlot = Instantiate(towerSlotPrefab, availableTowersPanel);
    //     newSlot.GetComponent<Image>().sprite = sprites[(int)tower];

    //     // In-Game Towers Panel에 타워 슬롯을 생성하고 설정합니다.
    //     GameObject inGameSlot = Instantiate(inGameTowerSlotPrefab, inGameTowersPanel);
    //     inGameSlot.GetComponent<TowerSlot>().Setup(tower, sprites[(int)tower]);
    // }
}
