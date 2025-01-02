// #define DEBUG_MODE
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/* Mage의 SKill: IceAttack */
public class IceAttack : MonoBehaviour, ISkill
{
    /* Ice 공격 변수 */
    
    [SerializeField]
    private GameObject footPrefab;  // 발 위치에 붙일 Prefab
    private CapsuleCollider2D capsuleCollider2D;
    private Vector3 footPosition;
    private GameObject footInstance;

    /*************************/
    // public bool isIceAttackActive = false;
    public float iceDamage;
    public float iceDamageDecreaseAmount = 0.5f;
    public float maintainanceTime = 3.0f; // BasicSlow 지속시간
    public float baseMaintainanceTime = 3.0f; // original BasicSlow 지속시간
    /*************************/

    public void Activate()
    {
        // 인터페이스를 위한 공란 함수 정의
    }

    public void SkillEffectActivate(Transform target)
    {
        // CapsuleCollider2D에서 bounds를 가져와서 발 위치 계산
        capsuleCollider2D = target.GetComponent<CapsuleCollider2D>();
        Bounds bounds = capsuleCollider2D.bounds;
        footPosition = new Vector2(bounds.center.x, bounds.min.y);

        // 적이 얼어있지 않으면, 발 위치에 Prefab 인스턴스화
        if(target.GetComponent<EnemyMovement>().isSpeedReduced == false)
        {
            footInstance = Instantiate(footPrefab, footPosition, Quaternion.identity);
            // footInstance.GetComponent<BasicSlowMovement>().isIce = true;
        }
        
        // Slow 효과 99%
        target.GetComponent<EnemyMovement>().ReduceSpeedTemporarily();
    }

    public void LevelUp(float increaseAmount)
    {
        iceDamageDecreaseAmount += increaseAmount;
        this.maintainanceTime = baseMaintainanceTime + baseMaintainanceTime * increaseAmount;
    }
}
