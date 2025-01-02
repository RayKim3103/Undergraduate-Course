using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private Transform playerTransform;  // 플레이어의 Transform
    public Vector3 offset;             // 카메라와 플레이어 간의 오프셋
    public float smoothSpeed = 0.125f; // 카메라 이동의 부드러움 속도
    private float fixedZ;              // 고정된 Z 위치 값

    private void Awake() 
    {
         player = GameObject.FindWithTag(playerTag);
         playerTransform = player.transform;
    }

    void Start()
    {
        // 카메라의 초기 Z 값을 저장
        fixedZ = transform.position.z;
    }

    void LateUpdate()
    {
        // 플레이어 Transform이 할당되지 않은 경우 오류 로그를 출력하고 반환
        if (playerTransform == null)
        {
            Debug.LogError("Player Transform is not assigned.");
            return;
        }

        // Z 값을 고정한 채로 원하는 위치 계산
        Vector3 desiredPosition = new Vector3(playerTransform.position.x + offset.x, playerTransform.position.y + offset.y, fixedZ);
        // 현재 위치와 원하는 위치 사이를 부드럽게 보간
        Vector3 smoothedPosition = Vector3.Lerp(transform.position, desiredPosition, smoothSpeed);
        // 카메라의 위치를 보간된 위치로 설정
        transform.position = smoothedPosition;
    }
}
