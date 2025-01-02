using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Teleport : MonoBehaviour, ISkill
{
    // public bool isKeyCodeAssign;
    public KeyCode keyCodeTeleport = KeyCode.F;  // 텔레포트 키

    // Player 태그를 지정하는 변수
    public string playerTag = "Player";
    private GameObject player;  // 플레이어 게임 오브젝트
    private BaseCharacter baseCharacter;  // 플레이어의 MageController 스크립트
    private SkillController skillController;
    
    [SerializeField]
    private float teleportDistance = 1.0f;  // 텔레포트 거리
    private float baseTeleportDistance = 1.0f;
    
    [SerializeField]
    private float teleportCooldown = 3.0f;  // 텔레포트 쿨다운 시간
    private float lastTeleportTime;  // 마지막 텔레포트 시각

    private void Awake()
    {
        // Player 태그를 가진 오브젝트를 찾음
        player = GameObject.FindWithTag(playerTag);
        // Debug.Log("Teleport Player: " + player);

        // 플레이어 객체를 찾고 MageController 컴포넌트를 가져옵니다.
        baseCharacter = (player != null) ? player.GetComponent<BaseCharacter>() : null;

        // SkillController 컴포넌트 가져옴
        skillController = GetComponentInParent<SkillController>();

        // 플레이어가 없는 경우 에러 로그 출력
        if (player == null)
        {
            Debug.LogError("Player object with tag '" + playerTag + "' not found.");
        }

        // BaseCharacter가 없는 경우 에러 로그 출력
        else if (baseCharacter == null)
        {
            Debug.LogError("BaseCharacter component not found on Player object.");
        }
    }

    public void OnButtonClick()
    {
        if (baseCharacter == null || player == null)
        {
            Debug.LogError("Player or PlayerController is not assigned.");
            return;
        }
        if (!skillController.ArrayContainsValue(skillController.skills, (int)SkillSelection.SkillType.Teleport))
            return;

        // 텔레포트 키가 눌렸는지와 쿨다운이 지났는지 확인
        if (Time.time >= lastTeleportTime + teleportCooldown)
        {
            Vector3 teleportDirection = Vector3.zero;  // 텔레포트 방향 초기화
            teleportDirection = baseCharacter.lastMoveDirection;
            
            // // 방향키 입력 배열과 해당 방향 벡터 배열을 정의
            // KeyCode[] keys = { KeyCode.UpArrow, KeyCode.W, KeyCode.DownArrow, KeyCode.S, KeyCode.LeftArrow, KeyCode.A, KeyCode.RightArrow, KeyCode.D };
            // Vector3[] directions = { Vector3.up, Vector3.up, Vector3.down, Vector3.down, Vector3.left, Vector3.left, Vector3.right, Vector3.right };

            // // 방향키 입력에 따른 텔레포트 방향 설정
            // for (int i = 0; i < keys.Length; i++)
            // {
            //     if (Input.GetKey(keys[i]))
            //     {
            //         teleportDirection = directions[i];
            //         break;
            //     }
            // }

            // 텔레포트 방향이 설정된 경우
            if (teleportDirection != Vector3.zero)
            {
                // 목표 위치 계산
                Vector3 targetPosition = player.transform.position + teleportDirection * teleportDistance;

                // 플레이어 위치를 목표 위치로 이동
                player.transform.position = targetPosition;

                // 마지막 텔레포트 시간을 현재 시간으로 갱신
                lastTeleportTime = Time.time;
            }
        }
    }

    // 텔레포트 활성화 메서드
    public void Activate()
    {
        // 인터페이스를 위한 공란 함수 정의
    }
    public void SkillEffectActivate(Transform target)
    {
        // 인터페이스를 위한 공란 함수 정의
    }

    public void LevelUp(float increaseAmount)
    {
        this.teleportDistance = baseTeleportDistance + baseTeleportDistance * increaseAmount;
        Debug.Log("Teleport Distance: " + teleportDistance);
    }
}






    // // teleportLayerMask를 설정하여 텔레포트할 때 충돌 검사를 제외할 레이어를 지정합니다.
    // // 1 << playerLayer는 playerLayer에 해당하는 비트 위치를 1로 설정하는 비트 연산입니다.
    // // ~ 연산자는 비트를 반전시키므로, 1 << playerLayer의 결과를 반전시킴으로써 플레이어 레이어를 제외한 모든 레이어를 포함하는 마스크를 만듭니다.
    // // 이 레이어 마스크는 나중에 레이캐스트를 할 때 사용되어, 플레이어 자신과의 충돌을 무시하고 다른 객체와의 충돌만을 검사하게 됩니다.
    // playerLayer = player.layer;  // 플레이어 레이어 설정
    // teleportLayerMask = ~(1 << playerLayer);  // 텔레포트 레이어 마스크 설정
    // Debug.Log("** Player Layer: " + playerLayer + " **");
    // Debug.Log("** teleportLayerMask Layer: " + teleportLayerMask + " **");

    // // 방향키 입력에 따른 텔레포트 방향 설정
    // if (Input.GetKey(KeyCode.UpArrow) || Input.GetKey(KeyCode.W))
    // {
    //     teleportDirection = Vector3.up;
    // }
    // else if (Input.GetKey(KeyCode.DownArrow) || Input.GetKey(KeyCode.S))
    // {
    //     teleportDirection = Vector3.down;
    // }
    // else if (Input.GetKey(KeyCode.LeftArrow) || Input.GetKey(KeyCode.A))
    // {
    //     teleportDirection = Vector3.left;
    // }
    // else if (Input.GetKey(KeyCode.RightArrow) || Input.GetKey(KeyCode.D))
    // {
    //     teleportDirection = Vector3.right;
    // }


    // // 디버그 레이 그리기
    // // 유니티의 디버그 함수로, 게임 씬에 보이지 않는 레이를 그려서 개발자가 시각적으로 확인할 수 있게 합니다.
    // // player.transform.position: 레이의 시작 위치 / teleportDirection * teleportDistance: 레이의 방향과 길이
    // // Color.red: 레이의 색상 / 2f: 레이가 씬에 그려진 후 2초 동안 유지
    // Debug.DrawRay(player.transform.position, teleportDirection * teleportDistance, Color.red, 2f);

    // // 레이캐스트를 통해 충돌체 확인
    // RaycastHit2D hit = Physics2D.Raycast(player.transform.position, teleportDirection, teleportDistance, teleportLayerMask);
    // Debug.Log("** hit.collider: " + hit.collider + " **");
    // if (hit.collider != null)
    // {
    //     targetPosition = hit.point;  // 충돌 지점으로 목표 위치 수정
    // }