using UnityEngine;

[RequireComponent(typeof(RectTransform))]
public class SpaceshipMover : MonoBehaviour
{
    [Header("Movement Settings")]
    [Tooltip("Horizontal speed in UI units per second (Canvas units).")]
    public float moveSpeed = 300f;
    [Tooltip("Rect that defines the playable area. If not set, parent RectTransform is used.")]
    public RectTransform boundaryRect;
    [Tooltip("Canvas that contains the UI. If null, it will be auto-detected.")]
    public Canvas canvas;
    
    [Header("Touch Settings")]
    [Tooltip("If true, ship follows touch directly. If false, moves toward touch smoothly.")]
    public bool useDirectTouch = true;
    [Tooltip("Movement speed multiplier for smooth touch movement (only when useDirectTouch is false).")]
    public float touchSensitivity = 1f;
    
    private RectTransform shipRect;
    private Camera uiCamera; // null for Screen Space - Overlay; canvas.worldCamera otherwise

    private void Awake()
    {
        shipRect = GetComponent<RectTransform>();
        
        // Use parent as boundary if not provided
        if (boundaryRect == null)
        {
            boundaryRect = shipRect.parent as RectTransform;
            if (boundaryRect == null)
                Debug.LogError("SpaceshipMover: boundaryRect not set and parent is not a RectTransform.");
        }
        
        // Auto-detect canvas if not set
        if (canvas == null)
            canvas = GetComponentInParent<Canvas>();
            
        if (canvas == null)
        {
            Debug.LogWarning("SpaceshipMover: No Canvas found in parents. Touch conversion may be incorrect.");
            uiCamera = null;
        }
        else
        {
            // For Screen Space - Overlay, pass null camera. For Camera/World, use canvas.worldCamera.
            uiCamera = (canvas.renderMode == RenderMode.ScreenSpaceOverlay) ? null : canvas.worldCamera;
        }
        
        // Warn if hierarchy is unexpected (recommended: ship is a child of boundary)
        if (boundaryRect != null && shipRect.parent != boundaryRect)
        {
            Debug.LogWarning("SpaceshipMover: Ship is not a child of boundaryRect. " +
                             "Clamping assumes shipRect.anchoredPosition is relative to boundaryRect.");
        }
    }

    private void Update()
    {
        // Prefer touch on devices that support it; fall back to keyboard/mouse.
        if (Input.touchSupported && Input.touchCount > 0)
        {
            HandleTouchInput();
        }
        else
        {
            HandleKeyboardInput();
        }
    }

    private void HandleKeyboardInput()
    {
        float move = Input.GetAxisRaw("Horizontal") * moveSpeed * Time.deltaTime;
        // Horizontal move in UI space
        Vector2 newPos = shipRect.anchoredPosition + new Vector2(move, 0f);
        shipRect.anchoredPosition = ClampPosition(newPos);
    }

    private void HandleTouchInput()
    {
        Touch touch = Input.GetTouch(0);
        
        // Convert screen touch to boundary's local (anchored) space
        if (boundaryRect == null) return;
        
        if (RectTransformUtility.ScreenPointToLocalPointInRectangle(boundaryRect, touch.position, uiCamera, out var localPoint))
        {
            Vector2 targetPos = new Vector2(localPoint.x, shipRect.anchoredPosition.y);
            Vector2 newPos;

            if (useDirectTouch)
            {
                // Direct touch - ship position follows touch exactly
                newPos = targetPos;
            }
            else
            {
                // Smooth movement toward touch position
                float moveDistance = moveSpeed * touchSensitivity * Time.deltaTime;
                newPos = Vector2.MoveTowards(shipRect.anchoredPosition, targetPos, moveDistance);
            }

            shipRect.anchoredPosition = ClampPosition(newPos);
        }
    }

    private Vector2 ClampPosition(Vector2 pos)
    {
        if (boundaryRect == null) return pos;
        
        // Work in boundary's local rect space. anchoredPosition is relative to boundary's rect.
        Rect boundary = boundaryRect.rect;
        float halfShipWidth = shipRect.rect.width * 0.5f;
        float minX = boundary.xMin + halfShipWidth;
        float maxX = boundary.xMax - halfShipWidth;
        
        pos.x = Mathf.Clamp(pos.x, minX, maxX);
        return pos;
    }

    // Optional: Public method to change movement mode at runtime
    public void SetMovementMode(bool directTouch)
    {
        useDirectTouch = directTouch;
    }
}