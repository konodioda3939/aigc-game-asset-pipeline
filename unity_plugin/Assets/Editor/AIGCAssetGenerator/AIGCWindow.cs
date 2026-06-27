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

        // 生成模式
        private int _generationMode = 0; // 0=文生图, 1=草图精修, 2=3D模型生成
        private static readonly string[] GenerationModeNames = { "文生图", "草图精修", "3D模型生成" };

        // ControlNet 参数
        private int _controlModeIndex = 0; // 0=Canny, 1=Scribble
        private static readonly string[] ControlModeNames = { "Canny 线稿精修", "Scribble 草图生成", "Depth 深度保持" };
        private Texture2D _referenceTexture = null;
        private float _controlStrength = 0.8f;

        // 3D 模型生成参数
        private int _modelFormatIndex = 0; // 0=OBJ, 1=GLB
        private static readonly string[] ModelFormatNames = { "OBJ（推荐，Unity 原生支持）", "GLB" };
        private int _modelResolution = 256; // Mesh 精度
        private byte[] _generatedModelBytes = null;
        private string _generatedModelFormat = "glb";
        private int _generatedModelVertices = 0;
        private int _generatedModelFaces = 0;

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
            DrawGenerationModeSwitch();
            EditorGUILayout.Space(5);

            if (_generationMode == 1)
            {
                // 草图精修模式
                DrawReferenceImageField();
                EditorGUILayout.Space(5);
                DrawControlModeSelector();
                EditorGUILayout.Space(5);
            }
            else if (_generationMode == 2)
            {
                // 3D 模型生成模式
                DrawReferenceImageField();
                EditorGUILayout.Space(5);
                DrawModelFormatSelector();
                EditorGUILayout.Space(5);
            }

            // 3D 模式：不需要 prompt + 资产类型
            if (_generationMode != 2)
            {
                DrawPromptField();
                EditorGUILayout.Space(5);
                DrawAssetTypeSelector();
                EditorGUILayout.Space(5);
                DrawAdvancedOptions();
                EditorGUILayout.Space(10);
            }
            else
            {
                Draw3DAdvancedOptions();
                EditorGUILayout.Space(10);
            }
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

        private void DrawGenerationModeSwitch()
        {
            EditorGUILayout.LabelField("生成模式", EditorStyles.boldLabel);
            _generationMode = GUILayout.Toolbar(_generationMode, GenerationModeNames, GUILayout.Height(28));

            if (_generationMode == 0)
                EditorGUILayout.LabelField("纯文字描述 → AI 随机生成图片", EditorStyles.miniLabel);
            else if (_generationMode == 1)
                EditorGUILayout.LabelField("上传参考图（草图/线稿）→ AI 保持结构 + 按描述填充内容", EditorStyles.miniLabel);
            else
                EditorGUILayout.LabelField("上传角色设定图 → AI 自动生成带贴图的 3D 模型（.glb/.obj）", EditorStyles.miniLabel);
        }

        private void DrawReferenceImageField()
        {
            EditorGUILayout.LabelField("参考图（草图/线稿/轮廓）", EditorStyles.boldLabel);

            // ObjectField 拖入/选择 Texture2D
            _referenceTexture = (Texture2D)EditorGUILayout.ObjectField(
                "选择参考图",
                _referenceTexture,
                typeof(Texture2D),
                false,
                GUILayout.Height(18)
            );

            // 拖入提示
            if (_referenceTexture == null)
            {
                Rect dropArea = GUILayoutUtility.GetRect(0, 60, GUILayout.ExpandWidth(true));
                GUI.Box(dropArea, "拖入参考图到这里\n（或点击上方选择）");
            }
            else
            {
                // 预览参考图
                float previewWidth = Mathf.Min(position.width - 40, 200);
                float aspect = (float)_referenceTexture.height / _referenceTexture.width;
                float previewHeight = previewWidth * aspect;
                Rect previewRect = GUILayoutUtility.GetRect(previewWidth, previewHeight,
                    GUILayout.MaxWidth(previewWidth), GUILayout.MaxHeight(previewHeight));
                EditorGUI.DrawPreviewTexture(previewRect, _referenceTexture, null, ScaleMode.ScaleToFit);
            }
        }

        private void DrawControlModeSelector()
        {
            EditorGUILayout.LabelField("控制方式", EditorStyles.boldLabel);
            _controlModeIndex = GUILayout.Toolbar(_controlModeIndex, ControlModeNames, GUILayout.Height(25));

            string hint = _controlModeIndex switch
            {
                0 => "Canny: 适合线稿/轮廓清晰的草图上色",
                1 => "Scribble: 适合随手涂鸦 → 概念图生成",
                2 => "Depth: 适合照片/3D渲染图，保持前后空间关系",
                _ => ""
            };
            EditorGUILayout.LabelField(hint, EditorStyles.miniLabel);

            _controlStrength = EditorGUILayout.Slider("控制力度", _controlStrength, 0.1f, 2.0f);
            EditorGUILayout.LabelField(
                _controlStrength < 0.6f ? "  较松: AI 有更多创作自由" :
                _controlStrength > 1.2f ? "  较紧: 严格贴合参考图结构" :
                "  适中: 推荐值",
                EditorStyles.miniLabel
            );
        }

        private void DrawModelFormatSelector()
        {
            EditorGUILayout.LabelField("输出格式", EditorStyles.boldLabel);
            _modelFormatIndex = GUILayout.Toolbar(_modelFormatIndex, ModelFormatNames, GUILayout.Height(25));

            string hint = _modelFormatIndex switch
            {
                0 => "GLB: Unity 原生支持，贴图内嵌，推荐使用",
                1 => "OBJ: 通用格式，但贴图需单独处理",
                _ => ""
            };
            EditorGUILayout.LabelField(hint, EditorStyles.miniLabel);
        }

        private void Draw3DAdvancedOptions()
        {
            _showAdvanced = EditorGUILayout.Foldout(_showAdvanced, "高级选项");
            if (!_showAdvanced) return;

            EditorGUI.indentLevel++;

            _modelResolution = EditorGUILayout.IntSlider("Mesh 精度", _modelResolution, 128, 512);
            string resolutionHint = _modelResolution switch
            {
                128 => "  快速预览 — 精度较低，速度快（~5秒）",
                256 => "  标准质量 — 推荐值（~10秒）",
                512 => "  最高精度 — 顶点更密，但更慢且更吃显存（~30秒）",
                _ => ""
            };
            EditorGUILayout.LabelField(resolutionHint, EditorStyles.miniLabel);

            _useFixedSeed = EditorGUILayout.Toggle("固定随机种子", _useFixedSeed);
            if (_useFixedSeed)
            {
                _fixedSeedValue = EditorGUILayout.IntField("种子值", _fixedSeedValue);
            }

            EditorGUI.indentLevel--;
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
            // 3D 模式不需要 prompt，但需要参考图
            bool canGenerate = !_isGenerating && (
                _generationMode == 2
                    ? _referenceTexture != null
                    : !string.IsNullOrWhiteSpace(_prompt)
            );
            GUI.enabled = canGenerate;

            string buttonText = _isGenerating ? "生成中..." :
                _generationMode == 2 ? "生成 3D 模型 (Generate 3D)" : "生成 (Generate)";

            if (GUILayout.Button(buttonText, GUILayout.Height(40)))
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

            // 3D 模式：显示模型信息
            if (_generationMode == 2 && _generatedModelBytes != null)
            {
                EditorGUILayout.HelpBox(
                    $"3D 模型已生成！\n" +
                    $"格式: {_generatedModelFormat.ToUpper()}\n" +
                    $"顶点数: {_generatedModelVertices:N0}\n" +
                    $"面数: {_generatedModelFaces:N0}\n" +
                    $"文件大小: {_generatedModelBytes.Length / 1024} KB\n\n" +
                    $"点击「导入到项目」将模型导入为 Prefab。",
                    MessageType.Info
                );
                return;
            }

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
            bool hasResult = _generationMode == 2
                ? _generatedModelBytes != null
                : _generatedTexture != null;
            GUI.enabled = (hasResult && !_isGenerating);

            GUILayout.BeginHorizontal();

            string importLabel = _generationMode == 2
                ? "导入 3D 模型到项目"
                : "导入到项目 (Import to Project)";

            if (GUILayout.Button(importLabel, GUILayout.Height(35)))
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
            // 文生图/草图精修模式需要 prompt；3D 模式不需要
            if (_generationMode != 2 && string.IsNullOrWhiteSpace(_prompt))
            {
                SetStatus("请输入画面描述 (Prompt)。", MessageType.Warning);
                return;
            }

            // 草图精修/3D 模式：必须有参考图
            if ((_generationMode == 1 || _generationMode == 2) && _referenceTexture == null)
            {
                string msg = _generationMode == 2
                    ? "3D 模型生成需要上传参考图！请拖入一张角色或物体图片。"
                    : "草图精修模式下需要上传参考图！请拖入一张草图或线稿。";
                SetStatus(msg, MessageType.Warning);
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
            _generatedModelBytes = null;
            _lastGeneratedSeed = "";

            string waitMsg = _generationMode switch
            {
                0 => "正在生成图片，请耐心等待（通常 5~20 秒）...",
                1 => "ControlNet 生成中，首次需下载模型（约 1.4GB），请耐心等待...",
                2 => "TripoSR 3D 模型生成中，首次需下载模型（约 1.68GB），请耐心等待...",
                _ => "生成中..."
            };
            SetStatus(waitMsg, MessageType.Info);

            try
            {
                string assetType = AssetTypeNames[_assetTypeIndex];
                int? seed = _useFixedSeed ? _fixedSeedValue : null;

                if (_generationMode == 0)
                {
                    // 文生图（原有接口）
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
                else if (_generationMode == 1)
                {
                    // ControlNet 可控生成
                    string controlMode = _controlModeIndex switch { 0 => "canny", 1 => "scribble", 2 => "depth", _ => "canny" };

                    _generatedTexture = await AIGCClient.GenerateControlled(
                        referenceImage: _referenceTexture,
                        prompt: _prompt.Trim(),
                        controlMode: controlMode,
                        assetType: assetType,
                        seed: seed,
                        steps: _steps,
                        guidanceScale: _guidanceScale,
                        controlStrength: _controlStrength
                    );
                    _lastGeneratedSeed = seed?.ToString() ?? "随机";
                    SetStatus(
                        $"生成成功！图片尺寸: {_generatedTexture.width}×{_generatedTexture.height}",
                        MessageType.Info
                    );
                }
                else
                {
                    // 3D 模型生成
                    string format = _modelFormatIndex == 0 ? "obj" : "glb";

                    var (modelData, filename, outFormat, vertices, faces) =
                        await AIGCClient.GenerateModel(
                            referenceImage: _referenceTexture,
                            prompt: _prompt?.Trim() ?? "",
                            outputFormat: format,
                            resolution: _modelResolution,
                            seed: seed
                        );

                    _generatedModelBytes = modelData;
                    _generatedModelFormat = outFormat;
                    _generatedModelVertices = vertices;
                    _generatedModelFaces = faces;
                    _lastGeneratedSeed = seed?.ToString() ?? "随机";

                    SetStatus(
                        $"3D 模型生成成功！\n" +
                        $"格式: {outFormat.ToUpper()}, " +
                        $"顶点: {vertices:N0}, 面: {faces:N0}, " +
                        $"大小: {modelData.Length / 1024} KB",
                        MessageType.Info
                    );
                }
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
            if (_generationMode == 2)
            {
                // 3D 模型导入
                if (_generatedModelBytes == null)
                {
                    SetStatus("没有可导入的 3D 模型，请先生成。", MessageType.Warning);
                    return;
                }

                try
                {
                    string baseName = GenerateAssetName(
                        string.IsNullOrWhiteSpace(_prompt) ? "3d_model" : _prompt
                    );
                    string format = _modelFormatIndex == 0 ? "obj" : "glb";

                    string prefabPath = AssetImporter.SaveAsModel(_generatedModelBytes, baseName, format);

                    SetStatus(
                        $"3D 模型导入成功！\n" +
                        $"Prefab 路径: {prefabPath}\n" +
                        $"可直接拖入 Scene 使用。",
                        MessageType.Info
                    );

                    Debug.Log($"[AIGC] 3D 模型导入完成: {prefabPath}");
                }
                catch (Exception ex)
                {
                    SetStatus($"3D 模型导入失败: {ex.Message}", MessageType.Error);
                    Debug.LogError($"[AIGC] 3D 导入失败: {ex}");
                }

                return;
            }

            // 2D 图片导入
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
