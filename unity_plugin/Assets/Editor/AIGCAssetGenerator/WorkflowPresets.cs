using System;
using System.Collections.Generic;
using UnityEngine;

namespace AIGCAssetGenerator
{
    /// <summary>
    /// 工作流预设定义 — 4 个标准化游戏美术工作流。
    ///
    /// 每个预设定义了：
    /// - 工作流 ID（对应后端 /workflows/run 的 workflow 参数）
    /// - 名称、描述、图标
    /// - 输入参数 schema（用于动态生成 UI 表单）
    /// - Prompt 后缀（自动追加到用户输入）
    /// </summary>
    [Serializable]
    public class WorkflowPreset
    {
        public string Id;
        public string Name;
        public string NameEn;
        public string Description;
        public string Icon;
        public List<WorkflowField> Fields;

        /// <summary>
        /// 获取适合该工作流的默认 Prompt 前缀（帮助非技术人员写更好的 prompt）。
        /// </summary>
        public string GetPromptHint()
        {
            return Id switch
            {
                "character_concept" => "例如: female knight, silver armor, blue cape, fantasy style",
                "asset_generator" => "例如: golden sword / ancient forest ruins / dark fantasy panel",
                _ => "输入英文描述..."
            };
        }
    }

    /// <summary>
    /// 工作流输入字段定义。
    /// </summary>
    [Serializable]
    public class WorkflowField
    {
        public string Key;
        public string Label;
        public string Type; // "text", "image", "select", "boolean", "multiselect"
        public string Default;
        public string Placeholder;
        public List<WorkflowOption> Options;
        public bool Required;
    }

    /// <summary>
    /// 选项（用于 select / multiselect 类型字段）。
    /// </summary>
    [Serializable]
    public class WorkflowOption
    {
        public string Value;
        public string Label;
    }

    /// <summary>
    /// 4 个游戏美术工作流预设的静态定义。
    /// </summary>
    public static class WorkflowPresets
    {
        public static readonly WorkflowPreset CharacterConcept = new WorkflowPreset
        {
            Id = "character_concept",
            Name = "角色概念图",
            NameEn = "Character Concept",
            Description = "输入角色描述，AI 一次生成含正面/侧面/背面/3/4 的多角度转身图，角色天然一致。",
            Icon = "🎭",
            Fields = new List<WorkflowField>
            {
                new WorkflowField
                {
                    Key = "prompt", Label = "角色描述", Type = "text", Required = true,
                    Placeholder = "例如: female knight, silver armor, blue cape, fantasy style"
                },
            }
        };

        public static readonly WorkflowPreset AssetGenerator = new WorkflowPreset
        {
            Id = "asset_generator",
            Name = "游戏素材生成",
            NameEn = "Game Asset Generator",
            Description = "选风格（图标/场景/UI）+ 选模式（文字直出 / ControlNet精修），统一出图。",
            Icon = "🎯",
            Fields = new List<WorkflowField>
            {
                new WorkflowField
                {
                    Key = "prompt", Label = "描述文字", Type = "text", Required = true,
                    Placeholder = "例如: golden sword / ancient forest / dark fantasy panel"
                },
                new WorkflowField
                {
                    Key = "style", Label = "素材风格", Type = "select",
                    Default = "icon",
                    Options = new List<WorkflowOption>
                    {
                        new WorkflowOption { Value = "icon", Label = "⚔️ 图标" },
                        new WorkflowOption { Value = "scene", Label = "🏞️ 场景" },
                        new WorkflowOption { Value = "ui", Label = "🎨 UI 元素" },
                    }
                },
                new WorkflowField
                {
                    Key = "reference_image", Label = "参考图（可选）", Type = "image",
                    Placeholder = "拖入参考图 → ControlNet 精修；留空 → 纯文字生成"
                },
                new WorkflowField
                {
                    Key = "control_mode", Label = "ControlNet 模式（有参考图时）", Type = "select",
                    Default = "canny",
                    Options = new List<WorkflowOption>
                    {
                        new WorkflowOption { Value = "canny", Label = "📐 Canny 线稿精修" },
                        new WorkflowOption { Value = "scribble", Label = "✏️ Scribble 草图" },
                        new WorkflowOption { Value = "depth", Label = "📏 Depth 深度" },
                    }
                },
            }
        };

        /// <summary>
        /// 获取所有工作流预设列表。
        /// </summary>
        public static readonly WorkflowPreset Model3D = new WorkflowPreset
        {
            Id = "model_3d",
            Name = "3D 模型生成",
            NameEn = "3D Model",
            Description = "上传物体/道具图片（白色背景最佳），AI 生成带贴图的 3D 模型。默认 OBJ，Unity 原生直接导入。",
            Icon = "📦",
            Fields = new List<WorkflowField>
            {
                new WorkflowField
                {
                    Key = "prompt", Label = "模型描述（可选）", Type = "text",
                    Placeholder = "仅用于存档命名，不影响生成结果"
                },
                new WorkflowField
                {
                    Key = "reference_image", Label = "参考图（必须）", Type = "image",
                    Placeholder = "上传物体/角色正面照，推荐白色背景"
                },
                new WorkflowField
                {
                    Key = "resolution", Label = "Mesh 精度", Type = "select",
                    Default = "256",
                    Options = new List<WorkflowOption>
                    {
                        new WorkflowOption { Value = "128", Label = "⚡ 128 — 快速预览" },
                        new WorkflowOption { Value = "256", Label = "🎯 256 — 标准（推荐）" },
                        new WorkflowOption { Value = "512", Label = "💎 512 — 最高精度" },
                    }
                },
                new WorkflowField
                {
                    Key = "output_format", Label = "输出格式", Type = "select",
                    Default = "obj",
                    Options = new List<WorkflowOption>
                    {
                        new WorkflowOption { Value = "obj", Label = "📦 OBJ — Unity 原生支持" },
                        new WorkflowOption { Value = "glb", Label = "🎯 GLB — 贴图内嵌（需插件）" },
                    }
                },
            }
        };

        public static readonly WorkflowPreset PBRMaterial = new WorkflowPreset
        {
            Id = "pbr_material",
            Name = "PBR 材质",
            NameEn = "PBR Material",
            Description = "输入材质描述（如 'rough stone wall'），生成全套 PBR 贴图（颜色/法线/粗糙度/金属度）。",
            Icon = "🧱",
            Fields = new List<WorkflowField>
            {
                new WorkflowField
                {
                    Key = "prompt", Label = "材质描述", Type = "text", Required = true,
                    Placeholder = "例如: rough stone wall, wooden floor planks"
                },
                new WorkflowField
                {
                    Key = "tileable", Label = "无缝平铺 (Tileable)", Type = "boolean",
                    Default = "true"
                },
            }
        };

        public static List<WorkflowPreset> GetAll()
        {
            return new List<WorkflowPreset>
            {
                CharacterConcept,
                AssetGenerator,
                Model3D,
                PBRMaterial,
            };
        }

        /// <summary>
        /// 根据 ID 查找工作流预设。
        /// </summary>
        public static WorkflowPreset GetById(string id)
        {
            return GetAll().Find(p => p.Id == id);
        }
    }

    /// <summary>
    /// 工作流生成结果（从 HTTP 响应解析）。
    /// </summary>
    public class WorkflowResult
    {
        public string WorkflowId;
        public string Seed;
        public string Format; // "png" 或 "zip"
        public Texture2D PreviewImage; // 单张预览（PNG 格式时）
        public byte[] ImageData;         // 原始图片数据（composite）
        public byte[] ZipData;           // ZIP 数据（ZIP 格式时）
        public string[] ElementNames;    // ZIP 中包含的元素名列表
        public int ElementCount;
        public string Mood;
    }
}
