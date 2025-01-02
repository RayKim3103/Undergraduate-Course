using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

public class Curse : MonoBehaviour, ISkill
{
    public GameObject cursePrefab;
    private GameObject clone;

    // [SerializeField]
    // private float summonCooldown = 3.0f; // 소환 쿨타임
    // private float lastsummonTime; // 마지막 소환 시간
    public float curseDamage; // 공격력

    [SerializeField]
    public float damageDecreaseAmount = 0.2f;

    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private BaseCharacter playerController;

    private void Awake() 
    {
        player = GameObject.FindWithTag(playerTag);
        // playerController = player.GetComponent<BaseCharacter>();
    }
    public void Activate()
    {
        // 인터페이스를 위한 공란 함수 정의
    }

    public void SkillEffectActivate(Transform target)
    {
        // if (Time.time >= lastsummonTime + summonCooldown)
        // {
            clone = Instantiate(cursePrefab, transform.position, Quaternion.identity);

            clone.name = "BasicSummon";
            clone.transform.localScale = clone.transform.localScale * 0.5f;

        //     // 마지막 발사 시간을 현재 시간으로 업데이트
        //     lastsummonTime = Time.time;
        // }
        Instantiate(cursePrefab, target.position, target.rotation);
    }

    // public void ReleaseSkill()
    // {
    //     Destroy(clone);
    // }

    public void LevelUp(float damageIncreaseAmount)
    {
        // 0렙때, 0.2f -> 1렙때, 0.4f
        this.damageDecreaseAmount += damageIncreaseAmount;
    }
}
