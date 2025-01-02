using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class StatsUp : MonoBehaviour
{
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private BaseCharacter playerController;

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        playerController = player.GetComponent<BaseCharacter>();
    }

    public void OnClickButtonHp()
    {
        Debug.Log("statpoints: " + playerController.statPoints);
        if(playerController.statPoints > 0)
        {
            playerController.maxHp += 1;
            playerController.hp += 1;
            playerController.statPoints -= 1;
        }
        
    }

    public void OnClickButtonAttackPower()
    {
        if(playerController.statPoints > 0)
        {
            playerController.damage += 1;
            playerController.statPoints -= 1;
        }
    }

    public void OnClickButtonAttackSpeed()
    {
        if(playerController.statPoints > 0)
        {
            playerController.attackSpeed += 1;
            playerController.fireCooldown = playerController.fireCooldown * 3/playerController.attackSpeed; // 3은 attackSpeed 초기값
            playerController.statPoints -= 1;
        }
    }

    public void OnClickButtonMoveSpeed()
    {
        if(playerController.statPoints > 0)
        {
            playerController.moveSpeed += 1;
            playerController.statPoints -= 1;
        }
    }

}
