using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SetPlayerParent : MonoBehaviour
{
    void Awake()
    {
        // "Player" 태그를 가진 게임 오브젝트를 찾음
        GameObject player = GameObject.FindGameObjectWithTag("Player");
        
        // 만약 "Player" 태그를 가진 게임 오브젝트가 있다면
        if (player != null)
        {
            // 현재 게임 오브젝트의 부모를 "Player" 태그를 가진 게임 오브젝트로 설정
            transform.SetParent(player.transform);
        }
        else
        {
            Debug.LogError("Player 태그를 가진 게임 오브젝트를 찾을 수 없습니다.");
        }
    }
}
