using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class TowerSlot : MonoBehaviour
{
    Sprite sprite;
    GameManager.Tower tower;
    public string selectedTowersPanelName = "SelectedTower";
    private GameObject selectedTowersPanel;
    public GameObject selectedTowerPrefab;
    GameObject selectedTowerSlot;
    private bool toggle = false;

    private void Start() 
    {
        gameObject.GetComponent<Button>().onClick.AddListener(OnButtonClick);
        gameObject.GetComponent<Image>().sprite = sprite;
    }

    public void OnButtonClick()
    {
        if(GameManager.Instance.towerUsed <= GameManager.Instance.maxTower)
        {
            if( toggle == false && GameManager.Instance.towerUsed < GameManager.Instance.maxTower)
            {
                toggle = true;
                GameManager.Instance.PurchaseTower(tower);
                // Selected Tower Panel에 타워 슬롯을 생성합니다.
                selectedTowersPanel = GameObject.Find(selectedTowersPanelName);
                selectedTowerSlot = Instantiate(selectedTowerPrefab, selectedTowersPanel.transform);
                selectedTowerSlot.GetComponent<Image>().sprite = sprite;
            }
            else if(selectedTowerSlot != null)
            {
                toggle = false;
                GameManager.Instance.SellTower(tower);
                // 생성되었던 towerSlot Destroy
                Destroy(selectedTowerSlot);
            }
        }
    }

    public void Setup(GameManager.Tower tower, Sprite sprite)
    {
        this.sprite = sprite;
        this.tower = tower;
    }

}