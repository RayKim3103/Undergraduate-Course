using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PowerMovement : MonoBehaviour
{
    // preFab y축 위치 조정용
    [SerializeField]
    private float balanceY = 0.2f;
    
    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    // private MageController playerController;

    /* preFab 생성 위치 변수 */
    private CapsuleCollider2D capsuleCollider2D;
    // private Vector3 headPosition;

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        // playerController = player.GetComponent<MageController>();

    }

    // Update is called once per frame
    void Update()
    {
        // CapsuleCollider2D에서 bounds를 가져와서 머리 위치 계산
        capsuleCollider2D = player.GetComponent<CapsuleCollider2D>();
        Bounds bounds = capsuleCollider2D.bounds;
        // headPosition = new Vector2(bounds.center.x, bounds.max.y);
        gameObject.transform.position = new Vector3 (player.transform.position.x,player.transform.position.y+balanceY,player.transform.position.z);
    }
}
