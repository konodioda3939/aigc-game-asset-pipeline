using UnityEditor;
using UnityEngine;

namespace AIGCAssetGenerator
{
    /// <summary>
    /// AI 资产生成器 — 全局配置
    ///
    /// 所有参数都可在 Unity 的 Edit → Preferences → AIGC Settings 中修改。
    /// 修改后即时生效，无需重启 Unity。
    /// </summary>
    public static class AIGCSettings
    {
        // ===== API 连接 =====
        // 推理服务的地址（默认本机 8000 端口）
        private const string KEY_API_URL = "AIGC_ApiBaseUrl";
        private const string DEFAULT_API_URL = "http://127.0.0.1:8000";

        // ===== 生成参数默认值 =====
        private const string KEY_STEPS = "AIGC_DefaultSteps";
        private const string KEY_GUIDANCE = "AIGC_DefaultGuidanceScale";
        private const string KEY_NEGATIVE = "AIGC_DefaultNegativePrompt";

        private const int DEFAULT_STEPS = 25;
        private const float DEFAULT_GUIDANCE = 7.5f;

        // 默认负面提示词 — 屏蔽常见质量问题
        private const string DEFAULT_NEGATIVE =
            "lowres, bad anatomy, bad hands, text, error, extra digit, " +
            "fewer digits, cropped, worst quality, low quality, normal quality, " +
            "jpeg artifacts, signature, watermark, username, blurry";

        // ===== 3D 模型生成默认值 =====
        private const string KEY_MODEL_FORMAT = "AIGC_ModelFormat";
        private const string KEY_MODEL_RESOLUTION = "AIGC_ModelResolution";
        private const string KEY_AUTO_PREFAB = "AIGC_AutoCreatePrefab";

        private const string DEFAULT_MODEL_FORMAT = "glb";
        private const int DEFAULT_MODEL_RESOLUTION = 256;
        private const bool DEFAULT_AUTO_PREFAB = true;

        // ===== 属性（读写 EditorPrefs） =====
        public static string ApiBaseUrl
        {
            get => EditorPrefs.GetString(KEY_API_URL, DEFAULT_API_URL);
            set => EditorPrefs.SetString(KEY_API_URL, value);
        }

        public static int DefaultSteps
        {
            get => EditorPrefs.GetInt(KEY_STEPS, DEFAULT_STEPS);
            set => EditorPrefs.SetInt(KEY_STEPS, Mathf.Clamp(value, 10, 100));
        }

        public static float DefaultGuidanceScale
        {
            get => EditorPrefs.GetFloat(KEY_GUIDANCE, DEFAULT_GUIDANCE);
            set => EditorPrefs.SetFloat(KEY_GUIDANCE, Mathf.Clamp(value, 1f, 20f));
        }

        public static string DefaultNegativePrompt
        {
            get => EditorPrefs.GetString(KEY_NEGATIVE, DEFAULT_NEGATIVE);
            set => EditorPrefs.SetString(KEY_NEGATIVE, value);
        }

        public static string DefaultModelFormat
        {
            get => EditorPrefs.GetString(KEY_MODEL_FORMAT, DEFAULT_MODEL_FORMAT);
            set => EditorPrefs.SetString(KEY_MODEL_FORMAT, value);
        }

        public static int DefaultModelResolution
        {
            get => EditorPrefs.GetInt(KEY_MODEL_RESOLUTION, DEFAULT_MODEL_RESOLUTION);
            set => EditorPrefs.SetInt(KEY_MODEL_RESOLUTION, Mathf.Clamp(value, 128, 512));
        }

        public static bool AutoCreatePrefab
        {
            get => EditorPrefs.GetBool(KEY_AUTO_PREFAB, DEFAULT_AUTO_PREFAB);
            set => EditorPrefs.SetBool(KEY_AUTO_PREFAB, value);
        }

        // ===== 重置为默认值 =====
        public static void ResetAll()
        {
            ApiBaseUrl = DEFAULT_API_URL;
            DefaultSteps = DEFAULT_STEPS;
            DefaultGuidanceScale = DEFAULT_GUIDANCE;
            DefaultNegativePrompt = DEFAULT_NEGATIVE;
            DefaultModelFormat = DEFAULT_MODEL_FORMAT;
            DefaultModelResolution = DEFAULT_MODEL_RESOLUTION;
            AutoCreatePrefab = DEFAULT_AUTO_PREFAB;
        }
    }

    /// <summary>
    /// 在 Preferences 窗口中注册 AIGC 设置页，
    /// 用户可通过 Edit → Preferences → AIGC Settings 修改配置。
    /// </summary>
    public class AIGCSettingsProvider : SettingsProvider
    {
        private const string SETTINGS_PATH = "Preferences/AIGC Settings";

        public AIGCSettingsProvider(string path, SettingsScope scope)
            : base(path, scope) { }

        [SettingsProvider]
        public static SettingsProvider CreateProvider()
        {
            return new AIGCSettingsProvider(SETTINGS_PATH, SettingsScope.User);
        }

        public override void OnGUI(string searchContext)
        {
            EditorGUILayout.LabelField("连接设置", EditorStyles.boldLabel);

            string apiUrl = EditorGUILayout.TextField(
                "API 地址",
                AIGCSettings.ApiBaseUrl
            );
            if (apiUrl != AIGCSettings.ApiBaseUrl)
                AIGCSettings.ApiBaseUrl = apiUrl;

            EditorGUILayout.HelpBox(
                "推理服务默认地址为 http://127.0.0.1:8000\n" +
                "如果服务部署在其他机器上，请修改为对应 IP。",
                MessageType.Info
            );

            EditorGUILayout.Space();

            EditorGUILayout.LabelField("生成参数默认值", EditorStyles.boldLabel);

            int steps = EditorGUILayout.IntSlider(
                "推理步数",
                AIGCSettings.DefaultSteps,
                10, 100
            );
            if (steps != AIGCSettings.DefaultSteps)
                AIGCSettings.DefaultSteps = steps;

            EditorGUILayout.HelpBox(
                "步数越多图片越精细，但生成越慢。推荐 20~30。",
                MessageType.None
            );

            float guidance = EditorGUILayout.Slider(
                "提示词引导强度",
                AIGCSettings.DefaultGuidanceScale,
                1f, 20f
            );
            if (guidance != AIGCSettings.DefaultGuidanceScale)
                AIGCSettings.DefaultGuidanceScale = guidance;

            EditorGUILayout.HelpBox(
                "值越高越贴近你的描述，但过高会失真。推荐 5~10。",
                MessageType.None
            );

            string negative = EditorGUILayout.TextField(
                "默认负面提示词",
                AIGCSettings.DefaultNegativePrompt
            );
            if (negative != AIGCSettings.DefaultNegativePrompt)
                AIGCSettings.DefaultNegativePrompt = negative;

            EditorGUILayout.Space();

            EditorGUILayout.LabelField("3D 模型生成默认值", EditorStyles.boldLabel);

            string modelFormat = EditorGUILayout.TextField(
                "默认输出格式 (glb/obj)",
                AIGCSettings.DefaultModelFormat
            );
            if (modelFormat != AIGCSettings.DefaultModelFormat)
                AIGCSettings.DefaultModelFormat = modelFormat;

            EditorGUILayout.HelpBox(
                "GLB: Unity 原生支持，贴图内嵌，推荐使用。\nOBJ: 通用格式，需要单独处理贴图。",
                MessageType.None
            );

            int modelRes = EditorGUILayout.IntSlider(
                "默认 Mesh 精度",
                AIGCSettings.DefaultModelResolution,
                128, 512
            );
            if (modelRes != AIGCSettings.DefaultModelResolution)
                AIGCSettings.DefaultModelResolution = modelRes;

            bool autoPrefab = EditorGUILayout.Toggle(
                "自动创建 Prefab",
                AIGCSettings.AutoCreatePrefab
            );
            if (autoPrefab != AIGCSettings.AutoCreatePrefab)
                AIGCSettings.AutoCreatePrefab = autoPrefab;

            EditorGUILayout.Space();

            if (GUILayout.Button("恢复默认设置"))
            {
                AIGCSettings.ResetAll();
            }
        }
    }
}
