using UnityEngine;
public class NexusHealth : MonoBehaviour
{
    public float hp = 1000.0f;
    public float maxHp = 1000.0f;
    public bool isNexusDestroyed = false;

    public void TakeDamage(float damage)
    {
        hp -= damage;
        if (hp <= 0)
        {
            Die();
        }
    }

    void Die()
    {
        if (isNexusDestroyed == false)
        {
            isNexusDestroyed = true;
            InGameUIManager.instance.GameOver();
            // Debug.Log("Nexus Destroyed");
        }
    }
}