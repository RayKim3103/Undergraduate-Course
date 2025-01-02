using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ShieldMovement : MonoBehaviour
{
    
    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private BaseCharacter playerController;
    private ManaShield manaShield;

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        playerController = player.GetComponent<BaseCharacter>();
        manaShield = player.GetComponentInChildren<ManaShield>();

    }

    // Update is called once per frame
    void Update()
    {
        if (playerController.isDied == false)
        {
            gameObject.transform.position = player.transform.position;
        }
        else
        {
            manaShield.manaShieldActive = false;
            Destroy(gameObject);
        }
    }
}
