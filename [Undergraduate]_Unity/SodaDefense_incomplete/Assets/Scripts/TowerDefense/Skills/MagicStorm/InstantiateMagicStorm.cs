using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class InstantiateMagicStorm : MonoBehaviour, ISkill
{
    [SerializeField]
    private GameObject magicStormPrefab;
    private GameObject magicStormInstance;

    [SerializeField]
    private float balaceY;
    private Vector3 magicStormPosition;
    public float randomNumber = 0.2f;

    /* MagicStorm 데미지 & 지속시간 관련 변수 */
    public float attackDamage = 1f;        // 줄 데미지 양

    public float magicStormMaintainanceTime = 8.0f;
    public float baseMagicStormMaintainanceTime = 8.0f;

    public float attackDamageDecrease = 0.5f; // 데미지 감소 비율
    
    public void Activate()
    {
        // 인터페이스를 위한 공란 함수 정의
    }

    public void SkillEffectActivate(Transform target)
    {
        // 20% 확률로 ActivateSkill 함수 발동
        if (Random.value <= randomNumber)
        {
            magicStormPosition = new Vector3(target.transform.position.x, target.transform.position.y + balaceY, target.transform.position.z);
            magicStormInstance = Instantiate(magicStormPrefab, magicStormPosition, Quaternion.identity);
        }
    }
    public void LevelUp(float increaseAmount)
    {
        attackDamageDecrease += increaseAmount;
        randomNumber += increaseAmount;
        Debug.Log("attack DamageDecreaseAmount: " + attackDamageDecrease + "randomNum: " + randomNumber);
        // this.magicStormMaintainanceTime = baseMagicStormMaintainanceTime + baseMagicStormMaintainanceTime * increaseAmount;
    }
}
