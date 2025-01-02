using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro; // TextMeshPro를 사용할 경우 필요

public class SoundButton : MonoBehaviour
{
    public Button soundButton; // 버튼 참조
    public Text soundButtonText; // 텍스트 참조 (Legacy Text)

    private bool isSoundOn = true;

    void Start()
    {
        soundButton.onClick.AddListener(ToggleSound);
    }

    void ToggleSound()
    {
        isSoundOn = !isSoundOn;
        if (isSoundOn)
        {
            soundButtonText.text = "Sound ON";
        }
        else
        {
            soundButtonText.text = "Sound OFF";
        }
    }
}
