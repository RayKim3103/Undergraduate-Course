using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Singleton<T> : MonoBehaviour where T : MonoBehaviour
{
    private static T instance = null;

    // 싱글톤 인스턴스 설정
    public static T Instance
    {
        get
        {
            if(instance == null)
            {
                instance = (T)FindObjectOfType(typeof(T));

                if (instance == null)
                {
                    GameObject obj = new GameObject(typeof(T).Name, typeof(T));
                    instance = obj.GetComponent<T>();
                }
            }
            Debug.Log("get: " + instance);
            return instance;
        }
    }

    protected virtual void Awake()
    {
        Debug.Log("Sigletone Awake!");
        if(instance == null)
        {
            // instnce가 null이면 T를 this로 설정함
            instance = this as T;
            // 오브젝트가 부모가 있거나, 이 오브젝트가 최상위 오브젝트가 아니면
            if(transform.parent != null && transform.root.gameObject)
            {
                Debug.Log("Singleton Awake 1: " + instance);
                DontDestroyOnLoad(this.transform.root.gameObject);
            }
            else
            {
                Debug.Log("Singleton Awake 2: " + instance);
                DontDestroyOnLoad(this.gameObject);
            }
        }        
        else
        {
            Debug.Log("Have Instance: " + instance);
            Destroy(this.gameObject);
        }
    }
}
