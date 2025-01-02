using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SphereAttack : MonoBehaviour, ISkill
{
    /* Player 정보 변수 */
    private GameObject player;
    private BaseCharacter baseCharacter;

    /* SphereAttack 기본 변수 */
    public float damage = 1.0f;
    public float damageDecreaseAmount = 0.2f;
    public float orbitSpeed = 100.0f;
    public float baseOrbitSpeed = 100.0f;
    public float orbitRadius = 2.0f;
    public float startAngle = 0f;

    /* Sphere 개수 관리 및 프리팹 */
    public bool isSphereActivate = false;
    public float maxSpheres = 3;
    public float baseMaxSpheres = 3;
    public GameObject spherePrefab;
    public bool isLevelUp = false;
    
    private void Awake() 
    {
        player = GameObject.FindGameObjectWithTag("Player");
        baseCharacter = player.GetComponent<BaseCharacter>(); 
    }

    public void Activate()
    {
        MakeSphere();
    }
    public void SkillEffectActivate(Transform target)
    {
        // 인터페이스를 위한 공란 함수
    }

    void MakeSphere()
    {
        if(isSphereActivate == false && isLevelUp == false)
        {
            for(int i = 0; i < maxSpheres; i++)
            {
                GameObject sphere = Instantiate(spherePrefab, transform.position, transform.rotation);
                startAngle = 360 * i / maxSpheres;
                sphere.GetComponent<SphereMovement>().AngleSetUp(startAngle);
            }
            // Sphere가 다 생성 되었으므로 다 true로 바꿈
            isSphereActivate = true;
        }
    }
    public void LevelUp(float increaseAmount)
    {
        this.maxSpheres = baseMaxSpheres + increaseAmount;
        this.orbitSpeed = baseOrbitSpeed + baseOrbitSpeed * increaseAmount * 0.2f; // Skill Data를 깔끔하게 하기위해 여긴 하드코딩 하긴 함
        // Sphere을 다시 생성하기 위해
        this.isSphereActivate = false;
        if (baseCharacter.isDied == false)
        {
            this.isLevelUp = true; 
        }
    }

}
