using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class InstanceFollow : MonoBehaviour
{
   /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private BaseCharacter playerController;

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        playerController = player.GetComponent<BaseCharacter>();

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
            Destroy(gameObject);
        }
    }
}
