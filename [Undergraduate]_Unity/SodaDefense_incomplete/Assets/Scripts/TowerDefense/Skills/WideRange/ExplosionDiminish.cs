using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ExplosionDiminish : MonoBehaviour
{
    [SerializeField]
    private float diminshTime = 0.2f;
    private void OnEnable()
    {
        // 코루틴 실행
        StartCoroutine(DestroyAfterTime(diminshTime));
    }

    private IEnumerator DestroyAfterTime(float time)
    {
        // 주어진 시간 동안 대기
        yield return new WaitForSeconds(time);
        // 오브젝트 제거
        Destroy(gameObject);
    }
}
