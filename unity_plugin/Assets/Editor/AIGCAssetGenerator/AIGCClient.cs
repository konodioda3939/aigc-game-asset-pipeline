using System;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;

namespace AIGCAssetGenerator
{
    /// <summary>
    /// HTTP 客户端：与 Python 推理服务通信，发送 prompt 并接收生成的图片。
    ///
    /// 使用方式：
    ///   Texture2D tex = await AIGCClient.GenerateImage("a sword icon", "图标");
    ///
    /// 前提条件：
    ///   Python 推理服务必须已启动（双击 inference_server/start.bat）
    /// </summary>
    public static class AIGCClient
    {
        /// <summary>
        /// 调用推理服务生成一张图片。
        /// </summary>
        /// <param name="prompt">用户输入的文字描述（英文）</param>
        /// <param name="assetType">资产类型（图标/贴图/UI），会自动追加风格后缀</param>
        /// <param name="seed">随机种子，留空则每次生成不同结果</param>
        /// <param name="steps">推理步数，默认使用配置值</param>
        /// <param name="guidanceScale">引导强度，默认使用配置值</param>
        /// <returns>生成的 Texture2D 图片</returns>
        public static async Task<Texture2D> GenerateImage(
            string prompt,
            string assetType = "贴图",
            int? seed = null,
            int? steps = null,
            float? guidanceScale = null)
        {
            // 1. 根据资产类型自动追加风格关键词
            string fullPrompt = BuildPrompt(prompt, assetType);

            // 2. 构造 JSON 请求体（手动拼接，避免 Unity JsonUtility 对 Nullable/snake_case 的限制）
            string json = BuildRequestBody(
                fullPrompt,
                AIGCSettings.DefaultNegativePrompt,
                steps ?? AIGCSettings.DefaultSteps,
                guidanceScale ?? AIGCSettings.DefaultGuidanceScale,
                seed
            );

            // 3. 发送 HTTP POST 请求
            string url = $"{AIGCSettings.ApiBaseUrl}/generate";

            using (var request = new UnityWebRequest(url, "POST"))
            {
                byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
                request.uploadHandler = new UploadHandlerRaw(bodyRaw);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");
                request.timeout = 120; // 生成图片可能需要较长时间

                // 发送请求并等待完成
                var tcs = new TaskCompletionSource<bool>();
                var operation = request.SendWebRequest();
                operation.completed += _ => tcs.SetResult(true);
                await tcs.Task;

                // 4. 检查响应
                if (request.result != UnityWebRequest.Result.Success)
                {
                    string detail = request.downloadHandler?.text ?? request.error;
                    throw new AIGCException(
                        $"请求失败: {request.error}\n" +
                        $"详细信息: {detail}\n\n" +
                        $"请确认推理服务已启动：\n" +
                        $"  双击 inference_server\\start.bat\n" +
                        $"  或访问 {AIGCSettings.ApiBaseUrl}/docs 验证"
                    );
                }

                // 5. 解析图片
                byte[] imageData = request.downloadHandler.data;
                if (imageData == null || imageData.Length == 0)
                {
                    throw new AIGCException("服务返回了空数据，请重试。");
                }

                Texture2D texture = new Texture2D(2, 2, TextureFormat.RGBA32, false);
                if (!texture.LoadImage(imageData))
                {
                    throw new AIGCException("图片数据解析失败，返回的不是有效图片格式。");
                }

                return texture;
            }
        }

        /// <summary>
        /// ControlNet 可控生成：上传参考图 + prompt，AI 保持结构骨架生成精修图。
        /// </summary>
        /// <param name="referenceImage">参考图（草图/线稿/轮廓）</param>
        /// <param name="prompt">画面描述</param>
        /// <param name="controlMode">控制方式："canny" 或 "scribble"</param>
        /// <param name="assetType">资产类型</param>
        /// <param name="seed">随机种子</param>
        /// <param name="steps">推理步数</param>
        /// <param name="guidanceScale">引导强度</param>
        /// <param name="controlStrength">ControlNet 控制力度（0.1~2.0）</param>
        /// <returns>生成的 Texture2D 图片</returns>
        public static async Task<Texture2D> GenerateControlled(
            Texture2D referenceImage,
            string prompt,
            string controlMode = "canny",
            string assetType = "贴图",
            int? seed = null,
            int? steps = null,
            float? guidanceScale = null,
            float controlStrength = 0.8f)
        {
            if (referenceImage == null)
                throw new ArgumentNullException(nameof(referenceImage));

            string fullPrompt = BuildPrompt(prompt, assetType);

            // 确保贴图可读（Unity 默认锁定贴图像素，需要用 RenderTexture 桥接一次）
            Texture2D readableTexture = MakeTextureReadable(referenceImage);
            try
            {
                // 将 Texture2D 编码为 PNG 字节
                byte[] pngBytes = readableTexture.EncodeToPNG();
                if (pngBytes == null || pngBytes.Length == 0)
                    throw new AIGCException("参考图编码失败。");

                // 构造 multipart/form-data
                var formData = new System.Collections.Generic.List<UnityEngine.Networking.IMultipartFormSection>
                {
                    new MultipartFormFileSection("image", pngBytes, "reference.png", "image/png"),
                    new MultipartFormDataSection("prompt", fullPrompt),
                    new MultipartFormDataSection("control_mode", controlMode),
                    new MultipartFormDataSection("steps", (steps ?? AIGCSettings.DefaultSteps).ToString()),
                    new MultipartFormDataSection("guidance_scale", (guidanceScale ?? AIGCSettings.DefaultGuidanceScale).ToString("F1", System.Globalization.CultureInfo.InvariantCulture)),
                    new MultipartFormDataSection("control_strength", controlStrength.ToString("F2", System.Globalization.CultureInfo.InvariantCulture)),
                    new MultipartFormDataSection("negative_prompt", AIGCSettings.DefaultNegativePrompt),
                };

                if (seed.HasValue)
                    formData.Add(new MultipartFormDataSection("seed", seed.Value.ToString()));

                string url = $"{AIGCSettings.ApiBaseUrl}/generate-controlled";

                using (var request = UnityWebRequest.Post(url, formData))
                {
                    request.timeout = 180;

                    var tcs = new TaskCompletionSource<bool>();
                    var operation = request.SendWebRequest();
                    operation.completed += _ => tcs.SetResult(true);
                    await tcs.Task;

                    if (request.result != UnityWebRequest.Result.Success)
                    {
                        string detail = request.downloadHandler?.text ?? request.error;
                        throw new AIGCException(
                            $"生成失败: {request.error}\n" +
                            $"详细信息: {detail}\n\n" +
                            $"请确认推理服务已启动。\n" +
                            $"首次使用 ControlNet 需要下载模型（约 1.4GB），请检查服务端日志。"
                        );
                    }

                    byte[] imageData = request.downloadHandler.data;
                    if (imageData == null || imageData.Length == 0)
                        throw new AIGCException("服务返回了空数据。");

                    Texture2D texture = new Texture2D(2, 2, TextureFormat.RGBA32, false);
                    if (!texture.LoadImage(imageData))
                        throw new AIGCException("图片数据解析失败。");

                    return texture;
                }
            }
            finally
            {
                // 清理临时可读贴图（避免内存泄漏）
                if (readableTexture != referenceImage)
                    UnityEngine.Object.DestroyImmediate(readableTexture);
            }
        }

        /// <summary>
        /// 让任意 Texture2D 变为可读（即使是 Unity 锁定像素的导入贴图）。
        /// 原理：通过 RenderTexture 桥接，把像素数据拷贝到一个新贴图里。
        /// </summary>
        private static Texture2D MakeTextureReadable(Texture2D source)
        {
            // 如果本身就可读，直接返回
            try
            {
                source.GetPixel(0, 0);
                return source;
            }
            catch (UnityException)
            {
                // 不可读，走 RenderTexture 桥接
            }

            RenderTexture rt = RenderTexture.GetTemporary(
                source.width, source.height, 0,
                RenderTextureFormat.Default, RenderTextureReadWrite.Linear
            );

            Graphics.Blit(source, rt);
            RenderTexture previous = RenderTexture.active;
            RenderTexture.active = rt;

            Texture2D readable = new Texture2D(source.width, source.height, TextureFormat.RGBA32, false);
            readable.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            readable.Apply();

            RenderTexture.active = previous;
            RenderTexture.ReleaseTemporary(rt);

            return readable;
        }

        /// <summary>
        /// 检查推理服务是否在线。
        /// </summary>
        /// <returns>true 表示服务就绪</returns>
        public static async Task<(bool isReady, string status)> CheckHealth()
        {
            try
            {
                string url = $"{AIGCSettings.ApiBaseUrl}/health";

                using (var request = UnityWebRequest.Get(url))
                {
                    request.timeout = 5;

                    var tcs = new TaskCompletionSource<bool>();
                    var operation = request.SendWebRequest();
                    operation.completed += _ => tcs.SetResult(true);
                    await tcs.Task;

                    if (request.result == UnityWebRequest.Result.Success)
                    {
                        return (true, request.downloadHandler.text);
                    }
                    else
                    {
                        return (false, "无法连接到推理服务");
                    }
                }
            }
            catch (Exception ex)
            {
                return (false, $"连接异常: {ex.Message}");
            }
        }

        // ===== 内部方法 =====

        /// <summary>
        /// 根据资产类型在 prompt 后追加风格关键词，让生成结果更符合游戏素材需求。
        /// </summary>
        private static string BuildPrompt(string prompt, string assetType)
        {
            string suffix = assetType switch
            {
                "图标" => ", game icon, clean design, centered, transparent background, simple shapes, masterpiece",
                "贴图" => ", game texture, seamless, high quality, detailed, game asset, masterpiece",
                "UI"   => ", game UI element, clean interface, professional, simple, game asset, masterpiece",
                _     => ", masterpiece"
            };
            return prompt + suffix;
        }

        /// <summary>
        /// 手动构造 JSON 请求体。
        ///
        /// 不使用 JsonUtility 的原因：
        /// - JsonUtility 不支持 snake_case 字段名
        /// - JsonUtility 对 Nullable<int> 处理不理想
        /// - 手动拼接更可控，且这里结构简单
        /// </summary>
        private static string BuildRequestBody(
            string prompt,
            string negativePrompt,
            int steps,
            float guidanceScale,
            int? seed)
        {
            // 用 C# 11 的原始字符串字面量（raw string literal）让 JSON 更清晰
            // 注意：需要 Unity 2022+ 支持 C# 9+，如果编译器报错，回退到传统拼接
            string seedJson = seed.HasValue ? seed.Value.ToString() : "null";

            return
                "{" +
                $"\"prompt\": {EscapeJsonString(prompt)}," +
                $"\"negative_prompt\": {EscapeJsonString(negativePrompt)}," +
                $"\"steps\": {steps}," +
                $"\"guidance_scale\": {guidanceScale.ToString("F1", System.Globalization.CultureInfo.InvariantCulture)}," +
                $"\"seed\": {seedJson}" +
                "}";
        }

        /// <summary>
        /// 将字符串转为 JSON 安全的字符串值（加引号、转义特殊字符）
        /// </summary>
        private static string EscapeJsonString(string value)
        {
            if (string.IsNullOrEmpty(value))
                return "\"\"";

            var sb = new StringBuilder(value.Length + 2);
            sb.Append('"');
            foreach (char c in value)
            {
                switch (c)
                {
                    case '"':  sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n");  break;
                    case '\r': sb.Append("\\r");  break;
                    case '\t': sb.Append("\\t");  break;
                    default:   sb.Append(c);      break;
                }
            }
            sb.Append('"');
            return sb.ToString();
        }
    }

    /// <summary>
    /// AIGC 相关操作的异常类型，方便上层捕获并展示友好错误信息。
    /// </summary>
    public class AIGCException : Exception
    {
        public AIGCException(string message) : base(message) { }
        public AIGCException(string message, Exception inner) : base(message, inner) { }
    }
}
