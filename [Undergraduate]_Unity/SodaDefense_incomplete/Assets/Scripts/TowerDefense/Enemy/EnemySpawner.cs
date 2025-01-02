using UnityEngine;

public class EnemySpawner : MonoBehaviour
{
    /* Player 정보를 담을 변수 */
    public string playerTag = "Player"; // Player 오브젝트의 태그
    private GameObject player;
    private BaseCharacter baseCharacter;
    /* Spawner 위치 및 Data 변수들 */
    public GameObject enemyPrefab;
    public Transform[] spawnPoints; // 스폰 포인트 배열
    public SpawnData[] spawnData;
    public GameObject[] walls; // 보스와 일반 적들 나올 때의 collider를 다르게 하기 위해
    public GameObject bossCirclePrefab;
    private GameObject currentBossCirclePrefab;

    // private GameObject currentPrefab;
    public float spawnInterval = 10.0f; // 스폰 간격

    /* Round 정보 변수들 */

    [SerializeField]
    private int roundInterval = 5; // 라운드 간격 ( [roundInterval - 1]번 적들이 spawn 됨 )
    private float roundEndInterval = 10f; // 라운드가 끝나고 쉬는 시간
    private int cnt;
    private int roundCnt = 0;

    /* 적 정보 변수들 */

    [SerializeField]
    private int numberOfEnemies = 5;

    private float timeSinceLastSpawn;
    private bool isBossSpawned = false;

    // // Debug
    // private int callCount = 1;

    private void Awake()
    {
        player = GameObject.FindWithTag(playerTag);
        baseCharacter = player.GetComponent<BaseCharacter>();

        timeSinceLastSpawn = spawnInterval;
    }

    void LateUpdate()
    {
        timeSinceLastSpawn += Time.deltaTime;

        // 보스가 죽었다는 뜻
        if(isBossSpawned == true && GameObject.Find("Boss") == null && InGameUIManager.instance.isGameOver == false)
        {
            // 영역표시하는 프리팹 제거
            Destroy(currentBossCirclePrefab);
            // if문 조건 만족 시 보스가 죽었다는 뜻
            isBossSpawned = false;
            walls[1].SetActive(false); // WallBoss를 false로 함
            walls[0].SetActive(true); // WallEnemy를 true로 함

            // 'Enemy' 태그를 가진 모든 게임 오브젝트를 찾음 -> 각 라운드 Enemy 초기화 (보스가 죽었으니)
            GameObject[] enemies = GameObject.FindGameObjectsWithTag("Enemy");

            // 찾은 모든 게임 오브젝트를 파괴
            foreach (GameObject enemy in enemies)
            {
                Destroy(enemy);
            }

            // 라운드 cnt 초기화 및 Panel 띄우기
            // cnt = 0; // 이미 다른 조건에서 초기화 함
            roundCnt += 1;
            InGameUIManager.instance.RoundCntText(roundCnt);
            InGameUIManager.instance.ShowRoundPanel();
            
            if(roundCnt != 1)
            {
                // 라운드 진행할 수록 적 스폰 시간 짧아짐 (1라운드 제외) & 적 숫자 증가
                spawnInterval = spawnInterval * 0.8f;

                // roundInterval이 1 증가하면 Spawn이 1번 더 됨 = 적 숫자 증가
                roundInterval += 1; // Max값을 정해야 할 수도?
            }

            // spawnInterval이 너무 짧아지지 않게 조절
            if (spawnInterval <= 1.0f)
            {
                spawnInterval = 1.0f;
#if DEBUG_MODE
                Debug.Log("Spawn Interval: " + spawnInterval);
#endif
            }
            
            // 1라운드때는 바로 적 스폰, 그 이외의 경우 roundEndInterval만큼 후에 enemy spawn 시작
            if(roundCnt != 1)
                timeSinceLastSpawn = spawnInterval - roundEndInterval;
            cnt += 1;
        }

        if (timeSinceLastSpawn >= spawnInterval)
        {
            // 1번째 라운드
            if(roundCnt == 0) // cnt % roundInterval == 0 && GameObject.Find("Boss") == null && cnt == 0 && !baseCharacter.isDied || 
            {
                // 라운드 cnt 초기화 및 Panel 띄우기
                roundCnt += 1;
                InGameUIManager.instance.RoundCntText(roundCnt);
                InGameUIManager.instance.ShowRoundPanel();
                cnt += 1;
                
            }
            else if(cnt % roundInterval == 0  && roundCnt != 0)
            {
                cnt = 0;
                walls[0].SetActive(false); // WallEnemy를 false로 함
                walls[1].SetActive(true); // WallBoss를 true로 함
                SpawnEnemies(numberOfEnemies);
                SpawnBoss();
                isBossSpawned = true;
                timeSinceLastSpawn = 0f;
            }
            else
            {
                SpawnEnemies(numberOfEnemies);            
                timeSinceLastSpawn = 0f;
                cnt += 1;
// #if DEBUG_MODE
                Debug.Log("Enemy Spawn CNT: " + cnt);
// #endif
            } 
        }
    }

    void SpawnEnemies(int numberOfEnemies)
    {
        // 스폰 포인트 배열을 랜덤하게 섞습니다.
        ShuffleArray(spawnPoints);

        int roundLevel = roundCnt - 1;
        Debug.Log("EnemySpawner roundLevel: " + roundLevel);
        if(roundLevel >= spawnData.Length)
        {
            roundLevel = spawnData.Length - 1;
        }

       // 섞인 배열에서 앞의 numberOfEnemies 개수만큼 적을 생성합니다.
        for (int i = 0; i < numberOfEnemies; i++)
        {   
            // 마지막 index는 Boss의 정보이기에 제외하고 Random Number 뽑음
            int index = Random.Range(0, spawnData[roundLevel].sprites.Length - 1);
            GameObject currentPrefab = Instantiate(enemyPrefab, spawnPoints[i].position, spawnPoints[i].rotation);
            currentPrefab.name = "Enemy";
            currentPrefab.GetComponent<AttackEnemy>().Init(spawnData[roundLevel], roundLevel, index);
        }
    }

    void SpawnBoss()
    {
        int roundLevel = roundCnt - 1;
        Debug.Log("EnemySpawner roundLevel: " + roundLevel);
        if(roundLevel >= spawnData.Length)
        {
            roundLevel = spawnData.Length - 1;
        }

        if (isBossSpawned == false)
        {
            // 보스가 나왔으므로, 플레이어의 위치와 반경 설정
            currentBossCirclePrefab = Instantiate(bossCirclePrefab, new Vector3(0, 0, 0), Quaternion.identity);
            player.transform.position = Vector3.zero;
            // 각각의 SpawnData[index]의 마지막 index가 Boss 정보
            int index = spawnData[roundLevel].sprites.Length - 1;
            GameObject currentPrefab = Instantiate(enemyPrefab, spawnPoints[0].position, spawnPoints[0].rotation);
            currentPrefab.name = "Boss";
            currentPrefab.GetComponent<Rigidbody2D>().mass = 100;
            currentPrefab.GetComponent<AttackEnemy>().Init(spawnData[roundLevel], roundLevel, index);
        }
    }

    void ShuffleArray<T>(T[] array)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int randomIndex = Random.Range(0, i + 1);
            T temp = array[randomIndex];
            array[randomIndex] = array[i];
            array[i] = temp;
        }
    }
}

// 1개의 스크립트 내에 여러개의 클래스 선언 가능
// 직렬화: 개체를 저장 or 전송하기 위해 변환
[System.Serializable]
public class SpawnData
{
    // public float spawnTime;
    public Sprite[] sprites;
    public int[] hp;
    public float[] speed;
    public float[] damage;
    public float[] attackRange;
    public bool[] isLong;
    public int[] coinIncrease;
}
