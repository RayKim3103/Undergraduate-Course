using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Boomerang : MonoBehaviour, ISkill
{
    /* 부메랑 기본 변수 */
    private GameObject player;
    private BaseCharacter baseCharacter;
    public float damage = 1.0f;
    public float damageDecreaseAmount = 0.5f;
    public float speed = 5.0f;
    public float baseSpeed = 5.0f;
    public float maxDistance = 5.0f;
    public float arcHeight; // boomerang Movement에서만 쓰는 변수

    /* 쿨타임 관련 변수 */
    public float lastFireTime;
    public float fireCooldown = 5.0f;
    public float baseFireCooldown = 5.0f;

    public float maxBoomerangs = 1;
    public float baseMaxBoomerangs = 1;
    public GameObject boomerangPrefab;
    
    // private List<GameObject> boomerangs = new List<GameObject>();

    private void Awake() 
    {
        // player = GameObject.FindGameObjectWithTag("Player");
        // baseCharacter = player.GetComponent<BaseCharacter>();
        lastFireTime = Time.time; 
    }

    public void Activate()
    { 
        if (Time.time >= lastFireTime + fireCooldown)
        {
            ThrowBoomerang();
            lastFireTime = Time.time;
            Debug.Log("ThrowBoomerang called / fireCooldown: " + fireCooldown);
        }

        // // 부메랑 관리
        // ManageBoomerangs();
    }

    public void SkillEffectActivate(Transform target)
    {
        // 인터페이스를 위한 공란 함수
    }

    void ThrowBoomerang()
    {
        // // 최대 부메랑 개수에 도달하지 않은 경우에만 부메랑 발사
        // while (boomerangs.Count < maxBoomerangs)
        // {
        //     GameObject boomerang = Instantiate(boomerangPrefab, transform.position, transform.rotation);
        //     boomerangs.Add(boomerang);
        // }
        for(int i = 0; i < maxBoomerangs; i++)
        {
            GameObject boomerang = Instantiate(boomerangPrefab, transform.position, transform.rotation);
        }
    }

    // void ManageBoomerangs()
    // {
    //     // 부메랑 리스트에서 null인 오브젝트 제거
    //     // b => b == null는 람다 식(Lambda Expression)
    //     // b는 리스트의 각 요소
    //     // RemoveAll 메서드는 리스트의 모든 요소를 순회하며, 주어진 조건(b => b == null)을 평가, 조건이 true인 경우, 해당 요소는 리스트에서 제거
    //     boomerangs.RemoveAll(b => b == null);
    // }

    public void LevelUp(float increaseAmount)
    {
        // 0렙때, 0.2f -> 1렙때, 0.4f
        // damageDecreaseAmount += increaseAmount;
        this.maxBoomerangs = baseMaxBoomerangs + baseMaxBoomerangs * increaseAmount;
        this.fireCooldown = baseFireCooldown - increaseAmount;
    }
}
