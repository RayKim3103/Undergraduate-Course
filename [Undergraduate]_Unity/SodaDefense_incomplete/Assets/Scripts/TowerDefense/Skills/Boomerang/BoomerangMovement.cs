using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BoomerangMovement : MonoBehaviour
{
    /* 부메랑 기본 변수 */
    private GameObject player;
    private Boomerang boomerang;
    private BaseCharacter baseCharacter;

    public float arcHeight;
    
    /* 부메랑의 경로 관련 변수 */
    private Vector3 initialPosition;
    private Vector3 targetPosition;
    private Vector3 boomerangDirection;
    private Vector3 perpendicularDirection;
    private bool returning = false;
    private float journeyLength;
    private float startTime;

    void Awake()
    {
        player = GameObject.FindGameObjectWithTag("Player");
        baseCharacter = player.GetComponent<BaseCharacter>();
        boomerang = player.GetComponentInChildren<Boomerang>();

        // /* 부메랑 기본 변수 초기화 */
        // damage = boomerang.damage;
        // speed = boomerang.speed;
        // maxDistance = boomerang.maxDistance;
        // arcHeight = maxDistance * 0.2f;

        /* 부메랑 방향 및 목표 지점 설정 */
        boomerangDirection = GetRandomDirection();
        perpendicularDirection = new Vector3(-boomerangDirection.y, boomerangDirection.x, 0); // boomerangDirection과 직각을 이루는 방향 계산

        initialPosition = transform.position;
        targetPosition = initialPosition + boomerangDirection * baseCharacter.baseMaxDistance;
        journeyLength = Vector3.Distance(initialPosition, targetPosition);
        startTime = Time.time;
    }

    void Update()
    {
        boomerang.damage = baseCharacter.damage * boomerang.damageDecreaseAmount;
        boomerang.maxDistance = baseCharacter.baseMaxDistance;
        arcHeight = boomerang.maxDistance * 0.2f;

        float distCovered = (Time.time - startTime) * boomerang.speed;
        float fractionOfJourney = distCovered / journeyLength;

        if (!returning)
        {
            // 부메랑이 포물선을 그리며 전진하는 로직
            // Vector3.Lerp는 두 점 사이를 선형 보간(Linear Interpolation)
            // initialPosition과 targetPosition 사이의 fractionOfJourney 비율에 해당하는 위치를 계산
            // currentPos += perpendicularDirection * Mathf.Sin(Mathf.PI * fractionOfJourney) * arcHeight는 현재 경로의 직각인 포물선 높이
            // arcHeight는 포물선의 높이를 조정
            Vector3 currentPos = Vector3.Lerp(initialPosition, targetPosition, fractionOfJourney);
            currentPos += perpendicularDirection * Mathf.Sin(Mathf.PI * fractionOfJourney) * arcHeight;
            transform.position = currentPos;

            if (fractionOfJourney >= 1f)
            {
                returning = true;
                // startTime을 다시 설정하여 fractionOfJourney 초기화 되도록 함
                startTime = Time.time;
                // initialPosition을 반환하는 지점으로 다시 설정
                initialPosition = transform.position;
                // player의 위치와 반환되는 지점의 거리 계산
                journeyLength = Vector3.Distance(initialPosition, player.transform.position);
            }
        }
        else
        {
            // 부메랑이 플레이어에게 돌아오는 로직
            Vector3 currentPos = Vector3.Lerp(initialPosition, player.transform.position, fractionOfJourney);
            // currentPos.y -= Mathf.Sin(Mathf.PI * fractionOfJourney) * arcHeight;
            currentPos -= perpendicularDirection * Mathf.Sin(Mathf.PI * fractionOfJourney) * arcHeight;
            transform.position = currentPos;

            if (fractionOfJourney >= 1f)
            {
                Destroy(gameObject); // 부메랑을 제거
            }
        }
    }

    private Vector3 GetRandomDirection()
    {
        // 2D에서 랜덤한 방향을 얻기 위해 0도에서 360도 사이의 랜덤한 각도를 생성
        float angle = Random.Range(0f, 360f);
        
        // 각도를 라디안으로 변환
        float radians = angle * Mathf.Deg2Rad;
        
        // 코사인과 사인을 사용하여 방향 벡터 계산
        float x = Mathf.Cos(radians);
        float y = Mathf.Sin(radians);
        
        // Vector3로 반환 (z는 0으로 설정)
        return new Vector3(x, y, 0f);
    }

    private void OnTriggerEnter2D(Collider2D other)
    {
        // 충돌한 객체의 태그가 "Enemy"인지 확인
        if (other.CompareTag("Enemy"))
        {
            // Enemy 스크립트를 가져와 데미지를 줌
            AttackEnemy enemy = other.GetComponent<AttackEnemy>();
            if (enemy != null)
            {
                enemy.TakeDamage(boomerang.damage);
                Debug.Log("Boomerang Hit Enemy!! / Damage: " + boomerang.damage);
            }
        }
    }
}


