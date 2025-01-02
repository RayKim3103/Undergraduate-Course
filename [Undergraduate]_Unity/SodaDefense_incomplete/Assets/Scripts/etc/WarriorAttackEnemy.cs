using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WarriorAttackEnemy : MonoBehaviour
{
    // /* Enemy 체력 */
    // public float hp = 1.0f;

    // /* Character 경험치 및 LevelUp 관련 변수 */
    // public string playerTag = "Player"; // Player 오브젝트의 태그
    // public string basicAttackTag = "BasicAttack"; // BasicAttack 오브젝트의 태그

    // /* Player 정보를 담을 변수 */
    // [SerializeField]
    // private float levelHardness = 5.0f;
    // private GameObject player;
    // private WarriorController playerController;

    // /* BasicSkill 담을 변수 */
    // private MageBasicSkillEffect basicSkillEffect;

    // private void Awake() 
    // {
    //     player = GameObject.FindWithTag(playerTag);
    //     playerController = player.GetComponent<WarriorController>();
    //     basicSkillEffect = GetComponent<MageBasicSkillEffect>();
    // }

    // // Update is called once per frame
    // void Update()
    // {
    //     if(playerController.exp >= levelHardness) //playerController.level * levelHardness ----------------- 현재코드: 디버깅용
    //     {
    //         playerController.exp = 0;
    //         LevelUp();
    //         // Debug 용 출력문
    //         Debug.Log("Player Level Up!! / Current Level: " + playerController.level);
    //     }
    // }
    
    // public void TakeDamage(float damage)
    // {   
    //     hp -= damage;
    //     if (hp <= 0)
    //     {
    //         playerController.exp += 1.0f;
    //         // Debug 용 출력문
    //         Debug.Log("Player EXP Up!! / Current EXP: " + playerController.exp);
    //         Die();
    //     }
    // }

    // public void Die()
    // {
    //     // 적이 죽었을 때의 로직을 구현합니다.
    //     Debug.Log("Enemy died.");
    //     // basicSkillEffect.IceBreak();

    //     Destroy(gameObject);
    // }

    // public void LevelUp()
    // {
    //     // exp가 다 채워지면 LevelUp
    //     playerController.level += 1;  
    // }
}
