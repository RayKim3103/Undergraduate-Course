using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AreaPrefab : MonoBehaviour
{
    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private BaseCharacter baseCharacter;
    private AreaAttack areaAttack;

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        baseCharacter = player.GetComponent<BaseCharacter>();
        areaAttack = player.GetComponentInChildren<AreaAttack>();

    }

    // Update is called once per frame
    void Update()
    {
        if (baseCharacter.isDied == false)
        {
            gameObject.transform.position = player.transform.position;
        }
        else
        {
            areaAttack.isAttacking = false;
            areaAttack.isPrefabSpawned = false;
            Destroy(gameObject);
        }
    }
}
