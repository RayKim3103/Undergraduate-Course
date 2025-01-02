using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ArcherDestroyAuto : MonoBehaviour
{
    private Vector2 limitMin = new Vector2(-12f, -12f);
    private Vector2 limitMax = new Vector2(12f, 12f);

    // Update is called once per frame
    private void Update()
    {
        // x, y 좌표가 범위 밖으로 벗어나면 object Destroy
        if(transform.position.x < limitMin.x || transform.position.x > limitMax.x
        || transform.position.y < limitMin.y || transform.position.y > limitMax.y)
        {
            // 소문자 gameObject 본인이 소속된 게임 오브젝트
            Destroy(gameObject);
        }
    }
}
