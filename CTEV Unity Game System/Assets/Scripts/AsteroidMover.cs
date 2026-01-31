using UnityEngine;

public class AsteroidMover : MonoBehaviour
{
    [Header("Velocity Settings")]
    public float verticalVelocity = 100f; // units per second
    public float horizontalVelocity = 20f; // units per second

    private float verticalScale;
    private float horizontalScale;
    private RectTransform parentRect;

    private void Start()
    {
        // Random scale factors for variation
        verticalScale = Random.Range(0.1f, 1f);
        horizontalScale = Random.Range(-1f, 1f);

        // Get parent RectTransform (Asteroids container)
        parentRect = transform.parent.GetComponent<RectTransform>();

        if (parentRect == null)
        {
            Debug.LogError("AsteroidMover: Parent does not have a RectTransform!");
        }
    }

    private void Update()
    {
        // Movement (local space so it's relative to parent RectTransform)
        Vector3 movement = new Vector3(
            horizontalVelocity * horizontalScale * Time.deltaTime,
            -verticalVelocity * verticalScale * Time.deltaTime,
            0f
        );

        transform.localPosition += movement;

        // Check if out of bounds
        if (IsOutOfBounds())
        {
            Destroy(gameObject);
        }
    }

    private bool IsOutOfBounds()
    {
        if (parentRect == null) return false;

        Vector3 localPos = transform.localPosition;
        float halfWidth = parentRect.rect.width / 2f;
        float halfHeight = parentRect.rect.height / 2f;

        // Consider asteroid's own size so it disappears fully outside
        float asteroidWidth = GetAsteroidWidth() / 2f;
        float asteroidHeight = GetAsteroidHeight() / 2f;

        // If outside vertically or horizontally
        if (localPos.y < -halfHeight - asteroidHeight) return true;
        if (localPos.x < -halfWidth - asteroidWidth) return true;
        if (localPos.x > halfWidth + asteroidWidth) return true;

        return false;
    }

    private float GetAsteroidWidth()
    {
        RectTransform rt = GetComponent<RectTransform>();
        if (rt != null) return rt.rect.width;
        return GetComponent<Renderer>()?.bounds.size.x ?? 0f;
    }

    private float GetAsteroidHeight()
    {
        RectTransform rt = GetComponent<RectTransform>();
        if (rt != null) return rt.rect.height;
        return GetComponent<Renderer>()?.bounds.size.y ?? 0f;
    }
}
