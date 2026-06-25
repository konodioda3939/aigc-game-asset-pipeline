# Unity Editor 插件 — AI 资产生成器

在 Unity 编辑器中输入文字描述，一键生成游戏素材（图标、贴图、UI元素），并自动导入为项目资产。

## 工作原理

```
Unity Editor  ──HTTP──→  Python 推理服务  ──SD+LoRA──→  生成图片  ──→  Assets/GeneratedAssets/
```

## 安装方法

1. 将 `Assets/Editor/AIGCAssetGenerator/` 整个文件夹复制到你的 Unity 项目的 `Assets/Editor/` 目录下
2. 等待 Unity 编译完成（底部状态栏显示编译进度）
3. 编译完成后，菜单栏会出现 `Tools → AI Asset Generator`

## 使用步骤

### 第一步：启动 Python 推理服务

```
双击 d:\aigc-project\inference_server\start.bat
```

看到以下输出说明启动成功：
```
服务已就绪 → http://127.0.0.1:8000
```

### 第二步：打开 Unity 插件

菜单栏 → `Tools` → `AI Asset Generator`

### 第三步：生成资产

1. 在「画面描述」输入框中输入英文描述
   - 例如: `a wooden chest icon, fantasy RPG style`
2. 选择资产类型（图标 / 贴图 / UI元素）
3. 点击「生成」按钮
4. 等待 5~20 秒，预览区域会显示生成的图片

### 第四步：导入到项目

1. 点击「导入到项目」
2. 图片会自动保存到 `Assets/GeneratedAssets/` 目录
3. 在 Project 窗口中可以看到新生成的资产
4. 拖到场景中的材质球或 Image 组件上即可使用

## 文件说明

| 文件 | 功能 |
|------|------|
| `AIGCWindow.cs` | Editor 窗口 UI（菜单、输入框、预览区、按钮） |
| `AIGCClient.cs` | HTTP 通信（调用 Python 推理服务的 `/generate` 接口） |
| `AssetImporter.cs` | 资产导入（PNG 编码、写入磁盘、刷新 AssetDatabase） |
| `AIGCSettings.cs` | 全局配置（API 地址、默认参数，可在 Preferences 中修改） |

## 配置说明

在 Unity 的 `Edit → Preferences → AIGC Settings` 中可以修改：

- **API 地址**: 推理服务的 URL，默认 `http://127.0.0.1:8000`
- **推理步数**: 默认 25，范围 10~100
- **引导强度**: 默认 7.5，范围 1~20
- **默认负面提示词**: 自动屏蔽低画质、畸形等

## 生成结果存档

生成的图片会自动存档在两个位置：
- `inference_server/outputs/` — Python 服务端的存档
- `Assets/GeneratedAssets/` — Unity 项目内的资产

## 常见问题

**Q: 点击生成后报错「推理服务未连接」？**
A: 确认 Python 推理服务已启动（双击 `start.bat`），然后点击「重新检测」按钮。

**Q: 生成报错「GPU 显存不足」？**
A: 在高级选项中降低推理步数（如 15 步），或关闭其他占用 GPU 的程序。

**Q: 生成的图片不太符合预期？**
A: 尝试更详细的 prompt，例如加上风格描述 (`fantasy style`, `cartoon`, `realistic`) 或调整资产类型。
