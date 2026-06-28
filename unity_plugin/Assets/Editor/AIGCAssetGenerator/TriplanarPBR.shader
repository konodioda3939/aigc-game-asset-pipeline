Shader "AIGC/TriplanarPBR"
{
    // ============================================================
    // Triplanar PBR Shader — 世界坐标三平面纹理投射
    //
    // 无视模型 UV，从 X/Y/Z 三个方向投射纹理并自动混合。
    // AI 生成的模型（UV 差/无 UV）直接可用。
    // ============================================================

    Properties
    {
        _MainTex ("Base Color", 2D) = "white" {}
        _BumpMap ("Normal Map", 2D) = "bump" {}
        _MetallicGlossMap ("Metallic(R) Smoothness(A)", 2D) = "white" {}
        _ParallaxMap ("Height Map", 2D) = "gray" {}

        _TexScale ("Tex Scale", Range(0.1, 20)) = 1.0
        _BlendSharpness ("Blend Sharpness", Range(1, 64)) = 8
        _NormalStrength ("Normal Strength", Range(0, 2)) = 1.0

        _MetalOverride ("Metallic Override", Range(0, 1)) = 0.0
        _SmoothOverride ("Smoothness Override", Range(0, 1)) = 0.5
        _Color ("Color Tint", Color) = (1, 1, 1, 1)
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 300

        CGPROGRAM
        #pragma surface surf Standard fullforwardshadows
        #pragma target 3.0

        sampler2D _MainTex;
        sampler2D _BumpMap;
        sampler2D _MetallicGlossMap;
        sampler2D _ParallaxMap;

        float _TexScale;
        float _BlendSharpness;
        float _NormalStrength;
        float _MetalOverride;
        float _SmoothOverride;
        float4 _Color;

        struct Input
        {
            float3 worldPos;
            float3 worldNormal;
            INTERNAL_DATA
        };

        void surf(Input IN, inout SurfaceOutputStandard o)
        {
            // ==== 通过 INTERNAL_DATA 获取 TBN 基向量 ====
            // WorldNormalVector 将切线空间向量转世界空间
            float3 wN = WorldNormalVector(IN, float3(0, 0, 1));
            float3 wT = WorldNormalVector(IN, float3(1, 0, 0));
            float3 wB = WorldNormalVector(IN, float3(0, 1, 0));

            // ==== 混合权重 ====
            float3 wabs = abs(wN);
            float3 weights = pow(wabs, _BlendSharpness);
            weights /= (weights.x + weights.y + weights.z);

            // ==== 三平面 UV ====
            float2 uvX = IN.worldPos.zy * _TexScale;
            float2 uvY = IN.worldPos.xz * _TexScale;
            float2 uvZ = IN.worldPos.xy * _TexScale;

            // ==== Albedo ====
            float3 cx = tex2D(_MainTex, uvX).rgb;
            float3 cy = tex2D(_MainTex, uvY).rgb;
            float3 cz = tex2D(_MainTex, uvZ).rgb;
            o.Albedo = (cx * weights.x + cy * weights.y + cz * weights.z) * _Color.rgb;

            // ==== Metallic + Smoothness ====
            float4 mx = tex2D(_MetallicGlossMap, uvX);
            float4 my = tex2D(_MetallicGlossMap, uvY);
            float4 mz = tex2D(_MetallicGlossMap, uvZ);
            o.Metallic   = mx.r * weights.x + my.r * weights.y + mz.r * weights.z;
            o.Smoothness = mx.a * weights.x + my.a * weights.y + mz.a * weights.z;

            // ==== Triplanar Normal Map ====
            float3 nxTS = UnpackNormal(tex2D(_BumpMap, uvX));
            float3 nyTS = UnpackNormal(tex2D(_BumpMap, uvY));
            float3 nzTS = UnpackNormal(tex2D(_BumpMap, uvZ));

            // 每个投影面的世界空间 TBN
            float sx = sign(wN.x);
            float sy = sign(wN.y);
            float sz = sign(wN.z);

            float3 nxW = nxTS.x * float3(0, 0, sx) + nxTS.y * float3(0, -sx, 0) + nxTS.z * float3(sx, 0, 0);
            float3 nyW = nyTS.x * float3(sy, 0, 0) + nyTS.y * float3(0, 0, sy) + nyTS.z * float3(0, sy, 0);
            float3 nzW = nzTS.x * float3(sz, 0, 0) + nzTS.y * float3(0, -sz, 0) + nzTS.z * float3(0, 0, sz);

            // 加权混合世界空间法线
            float3 blendWN = normalize(
                nxW * weights.x + nyW * weights.y + nzW * weights.z);

            // lerp 顶点法线与贴图法线
            blendWN = normalize(lerp(wN, blendWN, _NormalStrength));

            // 世界空间 → 切线空间（TBN 正交，逆=转置）
            float3 tangentN;
            tangentN.x = dot(blendWN, wT);
            tangentN.y = dot(blendWN, wB);
            tangentN.z = dot(blendWN, wN);
            o.Normal = normalize(tangentN);

            o.Alpha = 1;
        }
        ENDCG
    }

    FallBack "Standard"
}
