using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class MinimapViewport : MonoBehaviour
{
    public Camera mainCamera;
    public Camera minimapCamera;
    public RectTransform viewportRect;

    void Update()
    {
        // 메인 카메라의 뷰포트 크기를 가져오기
        float aspectRatio = mainCamera.aspect;
        float height = mainCamera.orthographicSize * 2;
        float width = height * aspectRatio;

        // 미니맵 카메라의 월드 좌표 변환
        Vector3 minimapCenter = minimapCamera.transform.position;
        Vector3 bottomLeft = new Vector3(minimapCenter.x - minimapCamera.orthographicSize * aspectRatio,
                                         minimapCenter.y - minimapCamera.orthographicSize,
                                         minimapCenter.z);
        Vector3 topRight = new Vector3(minimapCenter.x + minimapCamera.orthographicSize * aspectRatio,
                                       minimapCenter.y + minimapCamera.orthographicSize,
                                       minimapCenter.z);

        // 메인 카메라의 월드 좌표 변환
        Vector3 cameraBottomLeft = mainCamera.transform.position - new Vector3(width / 2, 0, height / 2);
        Vector3 cameraTopRight = mainCamera.transform.position + new Vector3(width / 2, 0, height / 2);

        // 미니맵 좌표계를 기준으로 뷰포트 직사각형 위치 및 크기 설정
        Vector2 viewportMin = new Vector2((cameraBottomLeft.x - bottomLeft.x) / (topRight.x - bottomLeft.x),
                                          (cameraBottomLeft.z - bottomLeft.z) / (topRight.z - bottomLeft.z));
        Vector2 viewportMax = new Vector2((cameraTopRight.x - bottomLeft.x) / (topRight.x - bottomLeft.x),
                                          (cameraTopRight.z - bottomLeft.z) / (topRight.z - bottomLeft.z));

        // 뷰포트 직사각형의 앵커 및 크기 조정
        viewportRect.anchorMin = viewportMin;
        viewportRect.anchorMax = viewportMax;
    }
}
