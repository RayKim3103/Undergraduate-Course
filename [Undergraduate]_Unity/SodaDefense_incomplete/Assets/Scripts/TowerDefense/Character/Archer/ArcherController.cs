using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ArcherController : BaseCharacter
{
    protected override void Awake() 
    {
        base.Awake();
        attackSpeed = attackSpeed * 2.0f;
        fireCooldown = fireCooldown * 0.5f;
        damage = damage * 0.5f;
    }

    protected override void Update()
    {
        base.Update();
    }
}
