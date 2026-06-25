using System;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;

namespace AIGCAssetGenerator
{
    /// <summary>
    /// AI 资产生成器 — Editor 主窗口
    ///
    /// 打开方式：
    ///   菜单栏 → Tools → AI Asset Generator
    ///
    /// 使用流程：
    ///   1. 输入文字描述（英文 prompt）
    ///   2. 选择资产类型（图标 / 贴图 / UI元素）
    ///   3. 点击「生成」
    ///   4. 预览生成的图片
    ///   5. 点击「导入到项目」保存为 Unity 资产
    /// </summary>
    public class AIGCWindow : EditorWindow
    {
        // ===== UI 状态 =====
        private string _prompt = "";
        private int _assetTypeIndex = 1; // 默认"贴图"
        private static readonly string[] AssetTypeNames = { "图标", "贴图", "UI元素" };

        // 高级选项
        private bool _showAdvanced = false;
        private int _steps;
        private float _guidanceScale;
        private int? _seed;
        private bool _useFixedSeed = false;
        private int _fixedSeedValue = 42;

        // 生成状态
        private bool _isGenerating = false;
        private string _statusMessage = "";
        private MessageType _statusType = MessageType.Info;

        // 生成结果
        private Texture2D _generatedTexture = null;
        private string _lastGeneratedSeed = "";

        // 滚动位置
        private Vector2 _scrollPosition;

        // 服务状态
        private bool _serverChecked = false;
        private bool _serverOnline = false;

        // ===== 窗口入口 =====

        [MenuItem("Tools/AI Asset Generator")]
        public static void ShowWindow()
        {
            AIGCWindow window = GetWindow<AIGCWindow>("AI 资产生成器");
            window.minSize = new Vector2(380, 500);
            window.Show();
        }

        // ===== 初始化 =====

        private void OnEnable()
        {
            // 从配置加载默认值
            _steps = AIGCSettings.DefaultSteps;
            _guidanceScale = AIGCSettings.DefaultGuidanceScale;

            // 打开窗口时自动检测服务状态
            _ = CheckServerHealth();
        }

        private async Task CheckServerHealth()
        {
            _serverChecked = false;
            var (isReady, status) = await AIGCClient.CheckHealth();
            _serverOnline = isReady;
            _serverChecked = true;
            Repaint();
        }

        // ===== 绘制 UI =====

        private void OnGUI()
        {
            _scrollPosition = EditorGUILayout.BeginScrollView(_scrollPosition);

            DrawHeader();
            DrawServerStatus();
            EditorGUILayout.Space(10);
            DrawPromptField();
            EditorGUILayout.Space(5);
            DrawAssetTypeSelector();
            EditorGUILayout.Space(5);
            DrawAdvancedOptions();
            EditorGUILayout.Space(10);
            DrawGenerateButton();
            EditorGUILayout.Space(5);
            DrawStatusBar();
            EditorGUILayout.Space(10);
            DrawPreviewArea();
            EditorGUILayout.Space(5);
            DrawImportButton();

            EditorGUILayout.EndScrollView();
        }

        // ===== UI 子区域 =====

        private void DrawHeader()
        {
            GUILayout.BeginHorizontal();
            GUILayout.FlexibleSpace();

            GUIStyle titleStyle = new GUIStyle(EditorStyles.boldLabel)
            {
                fontSize = 16,
                alignment = TextAnchor.MiddleCenter
            };
            GUILayout.Label("AI 资产生成器", titleStyle);

            GUILayout.FlexibleSpace();
            GUILayout.EndHorizontal();

            EditorGUILayout.HelpBox(
                "用 AI 生成游戏素材：输入描述 → 点击生成 → 导入为 Unity 资产。\n" +
                "前提：需要先启动 Python 推理服务（双击 inference_server\\start.bat）。",
                MessageType.None
            );
        }

        private void DrawServerStatus()
        {
            GUILayout.BeginHorizontal();

            if (!_serverChecked)
            {
                GUILayout.Label("服务状态: 检测中...", EditorStyles.miniLabel);
            }
            else if (_serverOnline)
            {
                GUI.color = Color.green;
                GUILayout.Label("● 服务在线", EditorStyles.miniLabel);
                GUI.color = Color.white;
            }
            else
            {
                GUI.color = Color.red;
                GUILayout.Label("● 服务离线", EditorStyles.miniLabel);
                GUI.color = Color.white;
            }

            GUILayout.FlexibleSpace();

            if (GUILayout.Button("重新检测", EditorStyles.miniButton, GUILayout.Width(80)))
            {
                _ = CheckServerHealth();
            }

            GUILayout.EndHorizontal();
        }

        private void DrawPromptField()
        {
            EditorGUILayout.LabelField("画面描述 (Prompt)", EditorStyles.boldLabel);

            // 多行输入框
            EditorGUILayout.HelpBox(
                "请输入英文描述。例如:\n" +
                "  图标: \"a wooden chest icon, fantasy RPG style\"\n" +
                "  贴图: \"stone wall texture, medieval, seamless\"\n" +
                "  角色: \"1girl, raiden shogun, purple hair, masterpiece\"",
                MessageType.None
            );

            _prompt = EditorGUILayout.TextArea(_prompt, GUILayout.Height(60));
        }

        private void DrawAssetTypeSelector()
        {
            EditorGUILayout.LabelField("资产类型", EditorStyles.boldLabel);

            // 使用工具栏替代下拉框，更直观
            _assetTypeIndex = GUILayout.Toolbar(
                _assetTypeIndex,
                AssetTypeNames,
                GUILayout.Height(28)
            );

            // 不同类型的说明
            string typeHint = _assetTypeIndex switch
            {
                0 => "图标: 适合技能图标、道具图标，会追加 \"icon, clean design\" 等关键词",
                1 => "贴图: 适合地面、墙壁等材质贴图，会追加 \"seamless, texture\" 等关键词",
                2 => "UI元素: 适合按钮、面板等界面元素，会追加 \"UI element, clean\" 等关键词",
                _ => ""
            };
            EditorGUILayout.LabelField(typeHint, EditorStyles.miniLabel);
        }

        private void DrawAdvancedOptions()
        {
            _showAdvanced = EditorGUILayout.Foldout(_showAdvanced, "高级选项");
            if (!_showAdvanced) return;

            EditorGUI.indentLevel++;

            _steps = EditorGUILayout.IntSlider("推理步数", _steps, 10, 100);
            EditorGUILayout.LabelField(
                $"  当前: {_steps} 步 — 步数越多越精细，但生成越慢。推荐 20~30。",
                EditorStyles.miniLabel
            );

            _guidanceScale = EditorGUILayout.Slider("引导强度 (CFG)", _guidanceScale, 1f, 20f);
            EditorGUILayout.LabelField(
                $"  当前: {_guidanceScale:F1} — 越高越贴近描述，过高会失真。推荐 5~10。",
                EditorStyles.miniLabel
            );

            _useFixedSeed = EditorGUILayout.Toggle("固定随机种子", _useFixedSeed);
            if (_useFixedSeed)
            {
                _fixedSeedValue = EditorGUILayout.IntField("种子值", _fixedSeedValue);
                EditorGUILayout.LabelField(
                    "  相同 Prompt + 相同种子 = 每次生成相同图片。",
                    EditorStyles.miniLabel
                );
            }
            else
            {
                EditorGUILayout.LabelField(
                    "  每次生成随机不同结果。",
                    EditorStyles.miniLabel
                );
            }

            EditorGUI.indentLevel--;
        }

        private void DrawGenerateButton()
        {
            GUI.enabled = !_isGenerating && !string.IsNullOrWhiteSpace(_prompt);

            if (GUILayout.Button(_isGenerating ? "生成中..." : "生成 (Generate)",
                    GUILayout.Height(40)))
            {
                _ = DoGenerate();
            }

            GUI.enabled = true;
        }

        private void DrawStatusBar()
        {
            if (string.IsNullOrEmpty(_statusMessage)) return;

            EditorGUILayout.HelpBox(_statusMessage, _statusType);

            // 如果生成了图片，显示种子信息（方便复现）
            if (!string.IsNullOrEmpty(_lastGeneratedSeed))
            {
                EditorGUILayout.LabelField(
                    $"种子 (Seed): {_lastGeneratedSeed}",
                    EditorStyles.miniLabel
                );
            }
        }

        private void DrawPreviewArea()
        {
            EditorGUILayout.LabelField("生成结果预览", EditorStyles.boldLabel);

            if (_generatedTexture != null)
            {
                // 计算预览尺寸（保持宽高比，最大 256 像素宽）
                float previewWidth = Mathf.Min(position.width - 40, 256);
                float aspect = (float)_generatedTexture.height / _generatedTexture.width;
                float previewHeight = previewWidth * aspect;

                Rect previewRect = GUILayoutUtility.GetRect(previewWidth, previewHeight,
                    GUILayout.MaxWidth(previewWidth), GUILayout.MaxHeight(previewHeight));

                // 居中显示
                float xOffset = (position.width - previewWidth) / 2 - 20;
                if (xOffset < 0) xOffset = 0;
                previewRect.x += xOffset;

                EditorGUI.DrawPreviewTexture(previewRect, _generatedTexture, null, ScaleMode.ScaleToFit);
            }
            else
            {
                // 占位区域
                Rect placeholderRect = GUILayoutUtility.GetRect(256, 128,
                    GUILayout.MaxWidth(256), GUILayout.MaxHeight(128));

                EditorGUI.HelpBox(placeholderRect,
                    "生成图片后将在此处预览",
                    MessageType.Info);
            }
        }

        private void DrawImportButton()
        {
            GUI.enabled = (_generatedTexture != null && !_isGenerating);

            GUILayout.BeginHorizontal();

            if (GUILayout.Button("导入到项目 (Import to Project)", GUILayout.Height(35)))
            {
                DoImport();
            }

            if (GUILayout.Button("打开输出文件夹", GUILayout.Height(35), GUILayout.Width(140)))
            {
                AssetImporter.OpenOutputFolder();
            }

            GUILayout.EndHorizontal();

            GUI.enabled = true;
        }

        // ===== 业务逻辑 =====

        /// <summary>
        /// 执行图片生成（异步，不阻塞 Editor UI）。
        /// </summary>
        private async Task DoGenerate()
        {
            if (string.IsNullOrWhiteSpace(_prompt))
            {
                SetStatus("请输入画面描述 (Prompt)。", MessageType.Warning);
                return;
            }

            if (!_serverOnline)
            {
                SetStatus(
                    "推理服务未连接！请先双击 inference_server\\start.bat 启动 Python 服务。",
                    MessageType.Error
                );
                return;
            }

            // 开始生成
            _isGenerating = true;
            _generatedTexture = null;
            _lastGeneratedSeed = "";
            SetStatus("正在生成图片，请耐心等待（通常 5~20 秒）...", MessageType.Info);

            try
            {
                string assetType = AssetTypeNames[_assetTypeIndex];
                int? seed = _useFixedSeed ? _fixedSeedValue : null;

                _generatedTexture = await AIGCClient.GenerateImage(
                    prompt: _prompt.Trim(),
                    assetType: assetType,
                    seed: seed,
                    steps: _steps,
                    guidanceScale: _guidanceScale
                );

                _lastGeneratedSeed = seed?.ToString() ?? "随机";
                SetStatus(
                    $"生成成功！图片尺寸: {_generatedTexture.width}×{_generatedTexture.height}",
                    MessageType.Info
                );
            }
            catch (AIGCException ex)
            {
                SetStatus(ex.Message, MessageType.Error);
                Debug.LogError($"[AIGC] 生成失败: {ex.Message}");
            }
            catch (Exception ex)
            {
                SetStatus($"未知错误: {ex.Message}", MessageType.Error);
                Debug.LogError($"[AIGC] 未知错误: {ex}");
            }
            finally
            {
                _isGenerating = false;
                Repaint();
            }
        }

        /// <summary>
        /// 将生成的图片导入为项目资产。
        /// </summary>
        private void DoImport()
        {
            if (_generatedTexture == null)
            {
                SetStatus("没有可导入的图片，请先生成。", MessageType.Warning);
                return;
            }

            try
            {
                string assetType = AssetTypeNames[_assetTypeIndex];

                // 用 prompt 前几个词作为文件名
                string baseName = GenerateAssetName(_prompt);

                string assetPath = AssetImporter.SaveAsAsset(_generatedTexture, baseName, assetType);

                SetStatus(
                    $"导入成功！\n" +
                    $"资产路径: {assetPath}\n" +
                    $"可在 Project 窗口中找到拖入场景使用。",
                    MessageType.Info
                );

                Debug.Log($"[AIGC] 资产导入完成: {assetPath}");
            }
            catch (Exception ex)
            {
                SetStatus($"导入失败: {ex.Message}", MessageType.Error);
                Debug.LogError($"[AIGC] 导入失败: {ex}");
            }
        }

        /// <summary>
        /// 根据 prompt 生成资产文件名。
        /// 例如 "a wooden chest icon" → "wooden_chest"
        /// </summary>
        private static string GenerateAssetName(string prompt)
        {
            if (string.IsNullOrWhiteSpace(prompt))
                return "generated_asset";

            // 取前 3 个词作为基础名称
            string[] words = prompt.Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);

            // 跳过常见的无意义前缀词

            var meaningfulWords = new System.Collections.Generic.List<string>();
            foreach (string word in words)
            {
                string lower = word.ToLower().Replace(",", "").Replace(".", "");
                if (lower != "a" && lower != "an" && lower != "the")
                {
                    meaningfulWords.Add(lower);
                }
                if (meaningfulWords.Count >= 3) break;
            }

            if (meaningfulWords.Count == 0)
            {
                // 全部是跳过词，直接用原词
                meaningfulWords.Add(words[0].ToLower().Replace(",", "").Replace(".", ""));
            }

            return string.Join("_", meaningfulWords);
        }

        /// <summary>
        /// 设置状态栏消息并立即刷新 UI。
        /// </summary>
        private void SetStatus(string message, MessageType type)
        {
            _statusMessage = message;
            _statusType = type;
            Repaint();
        }
    }
}
