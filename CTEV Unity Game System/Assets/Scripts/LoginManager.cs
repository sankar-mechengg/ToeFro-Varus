using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.IO;
using UnityEngine.SceneManagement;

public class LoginManager : MonoBehaviour
{
    [Header("UI References")]
    public TMP_InputField usernameInput;
    public TMP_InputField passwordInput;
    public Button loginButton;

    [Header("Global Variables")]
    public GlobalVariables userGlobals;

    private void Start()
    {
        loginButton.onClick.AddListener(OnLogin);
    }

    private void OnLogin()
    {
        string username = usernameInput.text.Trim();
        string password = passwordInput.text;

        if (string.IsNullOrEmpty(username) || string.IsNullOrEmpty(password))
        {
            Debug.LogWarning("Username or password is empty!");
            return;
        }

        // Set global variables
        userGlobals.Username = username;
        userGlobals.Password = password;

        // Create folder
        string userFolder = Path.Combine(Application.persistentDataPath, username);
        if (!Directory.Exists(userFolder))
        {
            Directory.CreateDirectory(userFolder);
        }

        // Save password to pwd.txt
        string passwordFile = Path.Combine(userFolder, "pwd.txt");
        File.WriteAllText(passwordFile, password);

        Debug.Log($"User folder created at: {userFolder}");

        SceneManager.LoadScene("PredictionScene");
    }
}
