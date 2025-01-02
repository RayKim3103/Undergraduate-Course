using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System;
using System.Collections.Generic;


public class GetTowerButton : MonoBehaviour
{
    public Button getButton; // 버튼 컴포넌트
    public Image towerImage;   // 이미지 컴포넌트
    public RectTransform towerImagePanel;  // 크기 조절할 패널의 RectTransform을 할당합니다.

    public float duration = 1.0f; // 애니메이션이 진행될 시간 (1초)
    private Vector2 originalSize;  // 패널의 원래 크기
    private bool gettingTower = false;

    void Start()
    {
        getButton.onClick.AddListener(OnButtonClick);
        towerImage.gameObject.SetActive(false); // 시작할 때 이미지를 비활성화합니다.
        
        // 패널의 원래 크기를 저장해 둡니다.
        originalSize = towerImagePanel.sizeDelta;
        
        // 패널 크기를 0으로 초기화합니다.
        towerImagePanel.sizeDelta = Vector2.zero;
    }

    void OnButtonClick()
    {
        // 이미지 활성화
        if (gettingTower == false)
        {
            gettingTower = true;

            // 랜덤 스프라이트 선택
            int randomIndex = UnityEngine.Random.Range(0, GameManager.Instance.sprites.Length);
            towerImage.sprite = GameManager.Instance.sprites[randomIndex];
            GameManager.Tower tower = GameManager.Instance.towers[randomIndex];

            TowerLevelUp(tower);
            MenuUIManager.instance.CreateTowerSlots(tower);

            towerImage.gameObject.SetActive(true);
            StartCoroutine(ScalePanel());
        }

    }

    void TowerLevelUp(GameManager.Tower tower)
    {
        switch (PlayerPrefs.GetInt(tower.ToString()))
        {
            case 0:
                PlayerPrefs.SetInt(tower.ToString(), 1);
                Debug.Log(tower + " is LV.1");
                break;
            case 1:
                PlayerPrefs.SetInt(tower.ToString(), 2);
                Debug.Log(tower + " is LV.2");
                break;
            case 2:
                PlayerPrefs.SetInt(tower.ToString(), 3);
                Debug.Log(tower + " is LV.3");
                break;
            default:
                // 예외 처리 등
                break;
        }

    }

    private IEnumerator ScalePanel()
    {
        float elapsedTime = 0f;

        while (elapsedTime < duration)
        {
            // 진행 비율 계산 (0에서 1까지)
            float t = elapsedTime / duration;
            
            // 크기 보간
            towerImagePanel.sizeDelta = Vector2.Lerp(Vector2.zero, originalSize, t);

            // 경과 시간 업데이트
            elapsedTime += Time.deltaTime;

            yield return null;
        }

        // 마지막으로 정확하게 원래 크기로 설정
        towerImagePanel.sizeDelta = originalSize;

        // 타워 획득이 끝났으므로 false로 바꿔줌, 코루틴이므로 false를 OnButtonClick()이 아닌 IEnumerator안에 적어야 함
        gettingTower = false;
        // Debug.Log("gettingTower: false");
    }
}
