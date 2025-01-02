using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class HUD : MonoBehaviour
{
    public enum InfoType { Exp, Level, Kill, Time, PlayerHealth, NexusHealth }
    public InfoType type;

    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    public string nexusTag = "Nexus"; // Nexus 오브젝트의 태그
    public GameObject player;
    public GameObject nexus;
    private BaseCharacter baseCharacter;

    Text myText;
    Slider mySlider;

    void Awake()
    {
        player = GameObject.FindWithTag(playerTag);

        myText = GetComponent<Text>();
        mySlider = GetComponent<Slider>();
    }

    void LateUpdate()
    {
        baseCharacter = player.GetComponent<BaseCharacter>();
        nexus = GameObject.FindWithTag(nexusTag);
        if(baseCharacter != null && nexus != null)
        {
            switch (type) {
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
                case InfoType.PlayerHealth:
                    float curPlayerHealth = baseCharacter.hp;
                    float maxPlayerHealth = baseCharacter.maxHp;
                    mySlider.value = curPlayerHealth / maxPlayerHealth;
                    break;
                case InfoType.NexusHealth:
                    float curNexusHealth = nexus.GetComponent<NexusHealth>().hp;
                    float maxNexusHealth = nexus.GetComponent<NexusHealth>().maxHp;
                    mySlider.value = curNexusHealth / maxNexusHealth;
                    break;
            }
        }
        
    }
}

