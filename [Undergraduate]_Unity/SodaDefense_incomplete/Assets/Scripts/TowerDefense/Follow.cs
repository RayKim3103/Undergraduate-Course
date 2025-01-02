using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class Follow : MonoBehaviour
{
    // /* Player 정보를 담을 변수 */
    // public string playerTag = "Player"; // Player 오브젝트의 태그
    // private GameObject player;

    private HUD hud;

    // UI의 경우 RectTransform을 따로 정의해 주어야 함
    RectTransform rect;

    void Awake()
    {
        // player = GameObject.FindWithTag(playerTag);
        rect = GetComponent<RectTransform>();
        hud = GetComponentInChildren<HUD>();
    }

    void FixedUpdate()
    {
        if (hud.player != null && hud.nexus != null)
        {
            switch (hud.type) 
            {
                // case InfoType.Exp:
                //     float curExp = GameManager.instance.exp;
                //     float maxExp = GameManager.instance.nextExp[Mathf.Min(GameManager.instance.level, GameManager.instance.nextExp.Length - 1)];
                //     // Slider의 게이지 바 값을 설정해줌
                //     mySlider.value = curExp / maxExp;
                //     break;
                // case InfoType.Level:
                //     // string.Format (문자열로 바꾸는 방법 중 하나)
                //     // Lv.{0:F0} / 0 => index 순번, F0 => 소수점 0자리 필요
                //     myText.text = string.Format("Lv.{0:F0}", GameManager.instance.level);
                //     break;
                // case InfoType.Kill:
                //     myText.text = string.Format("{0:F0}", GameManager.instance.kill);
                //     break;
                // case InfoType.Time:
                //     float remainTime = GameManager.instance.maxGameTime - GameManager.instance.gameTime;
                //     int min = Mathf.FloorToInt(remainTime / 60);
                //     int sec = Mathf.FloorToInt(remainTime % 60);
                //     // {0:D2}:{1:D2} / Decimal 2자리만 표현
                //     myText.text = string.Format("{0:D2}:{1:D2}", min, sec);
                //     break;
                case HUD.InfoType.PlayerHealth:
                    // RectTransform을 조절
                    // 주의! : rect.position = GameManager.instance.player.transform.position;
                    // 위와 같이 하면 에러 발생! / UI(스크린)의 좌표계와 Player(월드)의 좌표계는 다르기에
                    rect.position = Camera.main.WorldToScreenPoint(hud.player.transform.position);
                    break;
                case HUD.InfoType.NexusHealth:
                    rect.position = Camera.main.WorldToScreenPoint(hud.nexus.transform.position);
                    break;
            }
        }
    }
}

