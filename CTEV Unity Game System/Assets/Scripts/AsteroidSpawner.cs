using UnityEngine;
using System.Collections;

public class AsteroidSpawner : MonoBehaviour
{
    [Header("Asteroid Prefabs")]
    public GameObject asteroidSmall;
    public GameObject asteroidMedium;
    public GameObject asteroidLarge;

    [Header("Spawn Settings")]
    public float spawnInterval = 2f; // seconds between spawns

    private RectTransform rectTransform;

    private void Start()
    {
        rectTransform = GetComponent<RectTransform>();
        StartCoroutine(SpawnLoop());
    }

    private IEnumerator SpawnLoop()
    {
        while (true)
        {
            SpawnAsteroid();
            yield return new WaitForSeconds(spawnInterval);
        }
    }

    private void SpawnAsteroid()
    {
        // Pick a random prefab
        GameObject prefab = GetRandomAsteroidPrefab();
        if (prefab == null) return;

        // Get prefab width to keep it inside bounds
        float prefabWidth = GetPrefabWidth(prefab);

        // Get horizontal range inside RectTransform
        float halfWidth = rectTransform.rect.width / 2f;
        float minX = -halfWidth + (prefabWidth / 2f);
        float maxX = halfWidth - (prefabWidth / 2f);

        // Random X position
        float randomX = Random.Range(minX, maxX);

        // Y position slightly above top
        float topY = rectTransform.rect.height / 2f;
        float spawnY = topY + (GetPrefabHeight(prefab) / 2f);

        // Local position relative to RectTransform center
        Vector3 localSpawnPos = new Vector3(randomX, spawnY, 0f);

        // Instantiate and parent it
        GameObject asteroid = Instantiate(prefab, transform);
        asteroid.transform.localPosition = localSpawnPos;
    }

    private GameObject GetRandomAsteroidPrefab()
    {
        int choice = Random.Range(0, 3);
        if (choice == 0) return asteroidSmall;
        if (choice == 1) return asteroidMedium;
        return asteroidLarge;
    }

    private float GetPrefabWidth(GameObject prefab)
    {
        RectTransform rt = prefab.GetComponent<RectTransform>();
        if (rt != null)
            return rt.rect.width;
        return prefab.GetComponent<Renderer>()?.bounds.size.x ?? 0f;
    }

    private float GetPrefabHeight(GameObject prefab)
    {
        RectTransform rt = prefab.GetComponent<RectTransform>();
        if (rt != null)
            return rt.rect.height;
        return prefab.GetComponent<Renderer>()?.bounds.size.y ?? 0f;
    }
}
