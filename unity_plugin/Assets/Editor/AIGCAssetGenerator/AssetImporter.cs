using System;
using System.IO;
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
        /// 生成唯一资产路径，避免覆盖已有文件。
        /// 例如 chest.png 已存在 → chest_001.png, chest_002.png ...
        /// </summary>
        private static string GetUniqueAssetPath(string baseName)
        {
            string candidate = $"{OUTPUT_DIRECTORY}/{baseName}.png";

            if (!File.Exists(Path.Combine(Application.dataPath, "GeneratedAssets",
                $"{baseName}.png")))
            {
                return candidate;
            }

            // 已有同名文件，追加序号
            for (int i = 1; i <= 999; i++)
            {
                string numberedName = $"{baseName}_{i:D3}";
                candidate = $"{OUTPUT_DIRECTORY}/{numberedName}.png";

                if (!File.Exists(Path.Combine(Application.dataPath, "GeneratedAssets",
                    $"{numberedName}.png")))
                {
                    return candidate;
                }
            }

            // 极端情况：999 个重名，用时间戳兜底
            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            return $"{OUTPUT_DIRECTORY}/{baseName}_{timestamp}.png";
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
