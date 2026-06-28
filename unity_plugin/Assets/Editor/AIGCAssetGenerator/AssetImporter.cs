using System;
using System.IO;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace AIGCAssetGenerator
{
    /// <summary>
    /// 资产导入器：将生成的 Texture2D 保存为项目中的 PNG 资产。
    ///
    /// 核心功能：
    /// - 将 Texture2D 编码为 PNG 写入磁盘
    /// - 自动处理重名（chest → chest_001 → chest_002）
    /// - 触发 AssetDatabase 刷新
    /// - 可选：自动设置 TextureImporter 参数
    /// </summary>
    public static class AssetImporter
    {
        // 生成资产存放的目录（相对于项目根目录）
        private const string OUTPUT_DIRECTORY = "Assets/GeneratedAssets";

        /// <summary>
        /// 将 Texture2D 保存为 Unity 项目资产。
        /// </summary>
        /// <param name="texture">要保存的贴图</param>
        /// <param name="assetName">资产名称（不含扩展名），如 "chest" 或 "sword_icon"</param>
        /// <param name="assetType">资产类型，影响导入设置（图标/sprite，贴图/default）</param>
        /// <returns>资产的相对路径，如 "Assets/GeneratedAssets/chest_001.png"</returns>
        public static string SaveAsAsset(Texture2D texture, string assetName, string assetType = "贴图")
        {
            if (texture == null)
                throw new ArgumentNullException(nameof(texture));

            if (string.IsNullOrWhiteSpace(assetName))
                assetName = "generated_asset";

            // 1. 清理资产名（移除非法字符）
            assetName = SanitizeFileName(assetName);

            // 2. 确保输出目录存在
            string fullOutputDir = Path.Combine(Application.dataPath, "GeneratedAssets");
            if (!Directory.Exists(fullOutputDir))
                Directory.CreateDirectory(fullOutputDir);

            // 3. 处理重名：如果已存在同名文件，自动追加 _001, _002...
            string assetPath = GetUniqueAssetPath(assetName);

            // 4. 将 Texture2D 编码为 PNG 字节并写入磁盘
            byte[] pngData = texture.EncodeToPNG();
            if (pngData == null || pngData.Length == 0)
                throw new Exception("图片编码失败：EncodeToPNG 返回空数据。");

            string fullPath = Path.Combine(Application.dataPath, "GeneratedAssets",
                Path.GetFileName(assetPath));
            File.WriteAllBytes(fullPath, pngData);

            Debug.Log($"[AIGC] PNG 已写入: {fullPath} ({pngData.Length / 1024} KB)");

            // 5. 刷新 Unity 资源数据库（让新文件出现在 Project 窗口中）
            AssetDatabase.Refresh();

            // 6. 配置 TextureImporter 参数
            ConfigureTextureImporter(assetPath, assetType);

            Debug.Log($"[AIGC] 资产已导入: {assetPath}");

            // 7. 在 Project 窗口中高亮新资产
            EditorGUIUtility.PingObject(AssetDatabase.LoadAssetAtPath<Texture2D>(assetPath));

            return assetPath;
        }

        /// <summary>
        /// 将 3D 模型字节数据保存为 Unity 项目资产（.glb / .obj），并创建 Prefab。
        /// </summary>
        /// <param name="modelData">模型文件的原始字节数据</param>
        /// <param name="assetName">资产名称（不含扩展名）</param>
        /// <param name="format">格式："glb" 或 "obj"</param>
        /// <returns>生成的 Prefab 路径</returns>
        public static string SaveAsModel(byte[] modelData, string assetName, string format = "glb")
        {
            if (modelData == null || modelData.Length == 0)
                throw new ArgumentNullException(nameof(modelData));

            if (string.IsNullOrWhiteSpace(assetName))
                assetName = "generated_model";

            // 1. 清理资产名
            assetName = SanitizeFileName(assetName);

            // 2. 确保输出目录存在
            string fullOutputDir = Path.Combine(Application.dataPath, "GeneratedAssets");
            if (!Directory.Exists(fullOutputDir))
                Directory.CreateDirectory(fullOutputDir);

            // 3. 获取唯一路径
            string ext = format == "obj" ? ".obj" : ".glb";
            string assetPath = GetUniqueAssetPath(assetName, ext);

            // 4. 写入磁盘
            string fullPath = Path.Combine(Application.dataPath, "GeneratedAssets",
                Path.GetFileName(assetPath));
            File.WriteAllBytes(fullPath, modelData);

            Debug.Log($"[AIGC] 模型文件已写入: {fullPath} ({modelData.Length / 1024} KB)");

            // 5. 刷新 Unity 资源数据库
            AssetDatabase.Refresh();

            // 6. 配置 ModelImporter
            ConfigureModelImporter(assetPath);

            // 7. 创建 Prefab
            string prefabPath = CreatePrefabFromModel(assetPath);

            Debug.Log($"[AIGC] 3D 模型已导入: {assetPath}, Prefab: {prefabPath}");

            // 8. 在 Project 窗口中高亮 Prefab
            EditorGUIUtility.PingObject(AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath));

            return prefabPath;
        }

        /// <summary>
        /// 将 PBR 纹理贴图保存为 Unity Standard Shader Material。
        ///
        /// 纹理映射:
        ///   - basecolor.png         → _MainTex (sRGB)
        ///   - normal.png            → _BumpMap (Normal Map import)
        ///   - metallic_smoothness.png → _MetallicGlossMap (R=Metallic, A=Smoothness)
        ///   - height.png            → _ParallaxMap (可选)
        /// </summary>
        /// <param name="textures">贴图名称→字节数据的字典</param>
        /// <param name="materialName">材质名称（不含扩展名）</param>
        /// <returns>Material 资产路径</returns>
        public static string SaveAsPBRMaterial(
            Dictionary<string, byte[]> textures,
            string materialName)
        {
            if (textures == null || textures.Count == 0)
                throw new ArgumentNullException(nameof(textures));

            if (string.IsNullOrWhiteSpace(materialName))
                materialName = "pbr_material";

            materialName = SanitizeFileName(materialName);

            // 1. 确保输出目录存在
            string fullOutputDir = Path.Combine(Application.dataPath, "GeneratedAssets");
            if (!Directory.Exists(fullOutputDir))
                Directory.CreateDirectory(fullOutputDir);

            // 2. 保存各纹理为 PNG 资产
            var savedTexturePaths = new Dictionary<string, string>();

            foreach (var kvp in textures)
            {
                string mapType = Path.GetFileNameWithoutExtension(kvp.Key); // "basecolor", "normal", etc.
                byte[] data = kvp.Value;

                if (data == null || data.Length == 0) continue;

                // 跳过存档调试用的贴图，不导入 Material
                if (mapType == "roughness_raw" || mapType == "metallic_raw" || mapType == "preview")
                    continue;

                string textureName = $"{materialName}_{mapType}";
                string assetPath = $"{OUTPUT_DIRECTORY}/{textureName}.png";

                // 处理重名
                string uniquePath = GetUniqueAssetPath(textureName, ".png");

                string fullPath = Path.Combine(Application.dataPath, "GeneratedAssets",
                    Path.GetFileName(uniquePath));
                File.WriteAllBytes(fullPath, data);

                savedTexturePaths[mapType] = uniquePath;
                Debug.Log($"[AIGC] PBR 贴图已写入: {fullPath} ({data.Length / 1024} KB)");
            }

            // 3. 刷新 AssetDatabase 使新贴图可见
            AssetDatabase.Refresh();

            // 4. 配置纹理导入参数
            foreach (var kvp in savedTexturePaths)
            {
                ConfigurePBRTextureImporter(kvp.Value, kvp.Key);
            }

            // 5. 创建 Material
            string materialPath = CreatePBRMaterialAsset(materialName, savedTexturePaths);

            Debug.Log($"[AIGC] PBR 材质已导入: {materialPath}");

            // 6. 在 Project 窗口中高亮 Material
            EditorGUIUtility.PingObject(AssetDatabase.LoadAssetAtPath<Material>(materialPath));

            return materialPath;
        }

        /// <summary>
        /// 配置 PBR 贴图的导入参数。
        /// 不同贴图类型需要不同的 sRGB/NormalMap/wrapMode 设置。
        /// </summary>
        private static void ConfigurePBRTextureImporter(string assetPath, string mapType)
        {
            TextureImporter importer = UnityEditor.AssetImporter.GetAtPath(assetPath) as TextureImporter;
            if (importer == null) return;

            bool changed = false;

            switch (mapType)
            {
                case "basecolor":
                    // 颜色贴图: sRGB=On, Default type, Repeat wrap
                    if (!importer.sRGBTexture) { importer.sRGBTexture = true; changed = true; }
                    if (importer.textureType != TextureImporterType.Default)
                        { importer.textureType = TextureImporterType.Default; changed = true; }
                    if (importer.wrapMode != TextureWrapMode.Repeat)
                        { importer.wrapMode = TextureWrapMode.Repeat; changed = true; }
                    break;

                case "normal":
                    // 法线贴图: NormalMap type, sRGB=Off, Repeat wrap
                    if (importer.textureType != TextureImporterType.NormalMap)
                        { importer.textureType = TextureImporterType.NormalMap; changed = true; }
                    if (importer.sRGBTexture) { importer.sRGBTexture = false; changed = true; }
                    if (importer.wrapMode != TextureWrapMode.Repeat)
                        { importer.wrapMode = TextureWrapMode.Repeat; changed = true; }
                    break;

                case "metallic_smoothness":
                    // Metallic/Smoothness packed: sRGB=Off, Default type, Repeat wrap
                    if (importer.sRGBTexture) { importer.sRGBTexture = false; changed = true; }
                    if (importer.textureType != TextureImporterType.Default)
                        { importer.textureType = TextureImporterType.Default; changed = true; }
                    if (importer.wrapMode != TextureWrapMode.Repeat)
                        { importer.wrapMode = TextureWrapMode.Repeat; changed = true; }
                    break;

                case "height":
                    // 高度贴图: sRGB=Off, Default, Repeat wrap
                    if (importer.sRGBTexture) { importer.sRGBTexture = false; changed = true; }
                    if (importer.wrapMode != TextureWrapMode.Repeat)
                        { importer.wrapMode = TextureWrapMode.Repeat; changed = true; }
                    break;
            }

            // 禁用 Non-Power-of-2 缩放（512×512 已是 POT）
            if (importer.npotScale != TextureImporterNPOTScale.None)
                { importer.npotScale = TextureImporterNPOTScale.None; changed = true; }

            if (changed)
            {
                importer.SaveAndReimport();
                Debug.Log($"[AIGC] PBR TextureImporter 已配置: {mapType}");
            }
        }

        /// <summary>
        /// 创建 Standard Shader Material 并赋值 PBR 贴图。
        ///
        /// Standard Shader 的 PBR 属性:
        ///   _MainTex          = Albedo (Base Color)
        ///   _BumpMap          = Normal Map
        ///   _MetallicGlossMap = R=Metallic, A=Smoothness (Gloss)
        ///   _ParallaxMap      = Height/Displacement
        /// </summary>
        private static string CreatePBRMaterialAsset(
            string materialName,
            Dictionary<string, string> texturePaths)
        {
            // 生成唯一 Material 路径
            string materialPath = GetUniqueAssetPath(materialName, ".mat");

            // 创建 Triplanar PBR Material（无视 UV，AI 模型直接可用）
            Shader triplanarShader = Shader.Find("AIGC/TriplanarPBR");
            if (triplanarShader == null)
            {
                Debug.LogWarning("[AIGC] TriplanarPBR shader 未找到，回退到 Standard。"
                    + "请将 TriplanarPBR.shader 放入项目中。");
                triplanarShader = Shader.Find("Standard");
            }
            Material mat = new Material(triplanarShader);

            // 赋值纹理
            if (texturePaths.TryGetValue("basecolor", out string basecolorPath))
            {
                Texture2D basecolor = AssetDatabase.LoadAssetAtPath<Texture2D>(basecolorPath);
                if (basecolor != null)
                {
                    mat.SetTexture("_MainTex", basecolor);
                    mat.mainTexture = basecolor;
                }
            }

            if (texturePaths.TryGetValue("normal", out string normalPath))
            {
                Texture2D normal = AssetDatabase.LoadAssetAtPath<Texture2D>(normalPath);
                if (normal != null)
                {
                    mat.SetTexture("_BumpMap", normal);
                    mat.EnableKeyword("_NORMALMAP");
                }
            }

            if (texturePaths.TryGetValue("metallic_smoothness", out string metallicPath))
            {
                Texture2D metallic = AssetDatabase.LoadAssetAtPath<Texture2D>(metallicPath);
                if (metallic != null)
                {
                    mat.SetTexture("_MetallicGlossMap", metallic);
                    mat.EnableKeyword("_METALLICGLOSSMAP");
                }
            }

            if (texturePaths.TryGetValue("height", out string heightPath))
            {
                Texture2D height = AssetDatabase.LoadAssetAtPath<Texture2D>(heightPath);
                if (height != null)
                {
                    mat.SetTexture("_ParallaxMap", height);
                    mat.EnableKeyword("_PARALLAXMAP");
                }
            }

            // 创建 .mat 资产文件
            AssetDatabase.CreateAsset(mat, materialPath);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();

            Debug.Log($"[AIGC] Standard Material 已创建: {materialPath}");
            return materialPath;
        }

        /// <summary>
        /// 配置 ModelImporter 参数（材质导入模式、缩放等）。
        /// </summary>
        private static void ConfigureModelImporter(string assetPath)
        {
            ModelImporter importer = UnityEditor.AssetImporter.GetAtPath(assetPath) as ModelImporter;
            if (importer == null) return;

            bool changed = false;

            // 导入材质（TripoSR 生成的 glb 内嵌顶点色/贴图）
            if (importer.materialImportMode != ModelImporterMaterialImportMode.ImportStandard)
            {
                importer.materialImportMode = ModelImporterMaterialImportMode.ImportStandard;
                changed = true;
            }

            // 保持原始缩放
            if (Math.Abs(importer.globalScale - 1.0f) > 0.001f)
            {
                importer.globalScale = 1.0f;
                changed = true;
            }

            // 网格压缩（可选，减小文件大小）
            if (importer.meshCompression != ModelImporterMeshCompression.Off)
            {
                importer.meshCompression = ModelImporterMeshCompression.Off;
                changed = true;
            }

            if (changed)
            {
                importer.SaveAndReimport();
                Debug.Log("[AIGC] ModelImporter 已配置: materialImport=Standard, scale=1.0");
            }
        }

        /// <summary>
        /// 从导入的 3D 模型创建 Prefab。
        ///
        /// Unity 导入 .glb/.obj 后会生成一个模型资产（GameObject），
        /// 用 PrefabUtility 将其转为 Prefab，方便拖入场景。
        /// </summary>
        private static string CreatePrefabFromModel(string modelAssetPath)
        {
            // 尝试加载模型（Unity 的 glTF 导入器可能不兼容 trimesh 导出的 glb）
            GameObject modelRoot = AssetDatabase.LoadAssetAtPath<GameObject>(modelAssetPath);
            if (modelRoot == null)
            {
                // Prefab 创建失败不影响使用，用户可手动拖 glb 到场景
                Debug.Log($"[AIGC] GLB 模型已就绪: {modelAssetPath}，可直接拖到场景中使用。");
                return modelAssetPath;
            }

            // 生成 Prefab 路径（同目录，.prefab 扩展名）
            string prefabPath = Path.ChangeExtension(modelAssetPath, ".prefab");

            // 如果已存在同名 Prefab，删除旧的
            if (AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) != null)
            {
                AssetDatabase.DeleteAsset(prefabPath);
            }

            // 创建 Prefab
            GameObject prefab = PrefabUtility.SaveAsPrefabAsset(modelRoot, prefabPath);

            if (prefab != null)
            {
                Debug.Log($"[AIGC] Prefab 已创建: {prefabPath}");
                return prefabPath;
            }

            Debug.LogWarning($"[AIGC] Prefab 创建失败，使用原始模型路径。");
            return modelAssetPath;
        }

        /// <summary>
        /// 生成唯一资产路径，避免覆盖已有文件。
        /// 例如 chest.png 已存在 → chest_001.png, chest_002.png ...
        /// </summary>
        private static string GetUniqueAssetPath(string baseName, string extension = ".png")
        {
            string candidate = $"{OUTPUT_DIRECTORY}/{baseName}{extension}";

            if (!File.Exists(Path.Combine(Application.dataPath, "GeneratedAssets",
                $"{baseName}{extension}")))
            {
                return candidate;
            }

            // 已有同名文件，追加序号
            for (int i = 1; i <= 999; i++)
            {
                string numberedName = $"{baseName}_{i:D3}";
                candidate = $"{OUTPUT_DIRECTORY}/{numberedName}{extension}";

                if (!File.Exists(Path.Combine(Application.dataPath, "GeneratedAssets",
                    $"{numberedName}{extension}")))
                {
                    return candidate;
                }
            }

            // 极端情况：999 个重名，用时间戳兜底
            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            return $"{OUTPUT_DIRECTORY}/{baseName}_{timestamp}{extension}";
        }

        /// <summary>
        /// 根据资产类型配置 TextureImporter。
        /// </summary>
        private static void ConfigureTextureImporter(string assetPath, string assetType)
        {
            TextureImporter importer = UnityEditor.AssetImporter.GetAtPath(assetPath) as TextureImporter;
            if (importer == null) return;

            bool changed = false;

            // 所有类型统一设置为 Sprite（方便拖到 Canvas/Image 上）
            if (importer.textureType != TextureImporterType.Sprite)
            {
                importer.textureType = TextureImporterType.Sprite;
                changed = true;
            }

            // 图标和 UI 元素通常需要更高的滤波质量
            if (assetType == "图标" || assetType == "UI")
            {
                if (importer.filterMode != FilterMode.Bilinear)
                {
                    importer.filterMode = FilterMode.Bilinear;
                    changed = true;
                }
            }
            else // 贴图类型
            {
                if (importer.filterMode != FilterMode.Bilinear)
                {
                    importer.filterMode = FilterMode.Bilinear;
                    changed = true;
                }
                // 贴图可能需要可重复（tiling），但默认不强制设置
                importer.wrapMode = TextureWrapMode.Repeat;
                changed = true;
            }

            // 确保 sRGB 开启（颜色贴图需要）
            if (!importer.sRGBTexture)
            {
                importer.sRGBTexture = true;
                changed = true;
            }

            if (changed)
            {
                importer.SaveAndReimport();
                Debug.Log($"[AIGC] TextureImporter 已配置: type=Sprite, filter=Bilinear, sRGB=true");
            }
        }

        /// <summary>
        /// 清理文件名中的非法字符。
        /// </summary>
        private static string SanitizeFileName(string name)
        {
            // 移除 Windows 文件名不允许的字符
            char[] invalidChars = Path.GetInvalidFileNameChars();
            foreach (char c in invalidChars)
            {
                name = name.Replace(c.ToString(), "_");
            }

            // 去首尾空白、限制长度
            name = name.Trim();
            if (name.Length > 100)
                name = name.Substring(0, 100);

            // 替换空格为下划线（Unity 资产命名惯例）
            name = name.Replace(" ", "_");

            return string.IsNullOrEmpty(name) ? "generated_asset" : name;
        }

        /// <summary>
        /// 打开 GeneratedAssets 文件夹（在系统文件管理器中）。
        /// </summary>
        public static void OpenOutputFolder()
        {
            string path = Path.Combine(Application.dataPath, "GeneratedAssets");
            if (!Directory.Exists(path))
                Directory.CreateDirectory(path);

            EditorUtility.RevealInFinder(path);
        }
    }
}
