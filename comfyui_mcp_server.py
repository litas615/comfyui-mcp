import json
import os
import sys
import time
import random
import uuid
import urllib.request
import urllib.parse
import urllib.error
import base64
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from mcp.server.fastmcp import FastMCP

# 匯入配置
try:
    from config import COMFYUI_URL, A1111_URL, WORKFLOW_DIR, COMFYUI_OUTPUT_DIR, ASPECT_RATIOS
except ImportError:
    from .config import COMFYUI_URL, A1111_URL, WORKFLOW_DIR, COMFYUI_OUTPUT_DIR, ASPECT_RATIOS

mcp = FastMCP("Antigravity-ComfyUI-Bridge")


def test_comfyui_connection() -> Tuple[bool, str]:
    """測試 ComfyUI 是否在線"""
    url = f"{COMFYUI_URL}/system_stats"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Antigravity-MCP"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            devices = data.get("devices", [])
            gpu_info = devices[0].get("name", "Unknown GPU") if devices else "CPU"
            vram_free = devices[0].get("vram_free", 0) / (1024**3) if devices else 0
            return True, f"ComfyUI 連線正常 (裝置: {gpu_info}, 可用顯存: {vram_free:.2f} GB)"
    except Exception as e:
        return False, f"無法連線至 ComfyUI ({COMFYUI_URL}): {str(e)}"


def test_a1111_connection() -> Tuple[bool, str]:
    """測試 Stable Diffusion WebUI (A1111) 是否在線"""
    url = f"{A1111_URL}/sdapi/v1/sd-models"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Antigravity-MCP"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            models = json.loads(resp.read().decode("utf-8"))
            model_count = len(models)
            current_model = models[0].get("title", "Unknown") if models else "無載入模型"
            return True, f"A1111 連線正常 (目前模型: {current_model}, 可用模型數: {model_count})"
    except Exception as e:
        return False, f"無法連線至 A1111 ({A1111_URL}): {str(e)}"


def queue_prompt(workflow: dict, client_id: str) -> str:
    """向 ComfyUI 提交工作流佇列任務"""
    url = f"{COMFYUI_URL}/prompt"
    payload = {"prompt": workflow, "client_id": client_id}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "Antigravity-MCP"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            if "prompt_id" not in res:
                raise RuntimeError(f"ComfyUI 回應未包含 prompt_id: {res}")
            return res["prompt_id"]
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI 佇列提交失敗 (HTTP {e.code}): {err_body}")


def wait_for_completion(prompt_id: str, timeout: int = 240) -> Dict[str, str]:
    """輪詢 ComfyUI 歷史紀錄，等待生成完成並取得圖片資訊"""
    start_time = time.time()
    url = f"{COMFYUI_URL}/history/{prompt_id}"
    
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Antigravity-MCP"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                history = json.loads(resp.read().decode("utf-8"))
                if prompt_id in history:
                    prompt_data = history[prompt_id]
                    # 檢查是否有執行異常
                    status = prompt_data.get("status", {})
                    if status.get("status_str") == "error":
                        messages = status.get("messages", [])
                        raise RuntimeError(f"ComfyUI 算圖發生內部錯誤: {messages}")
                    
                    outputs = prompt_data.get("outputs", {})
                    for node_id, node_output in outputs.items():
                        if "images" in node_output and len(node_output["images"]) > 0:
                            image_info = node_output["images"][0]
                            return {
                                "filename": image_info["filename"],
                                "subfolder": image_info.get("subfolder", ""),
                                "type": image_info.get("type", "output"),
                                "node_id": str(node_id)
                            }
        except urllib.error.URLError:
            pass
        
        time.sleep(1.5)

    raise TimeoutError(f"ComfyUI 算圖逾時（已等待 {timeout} 秒）")


def inject_workflow_parameters(
    workflow: dict,
    scene_prompt: str,
    negative_prompt: str,
    seed: int,
    aspect_ratio: str,
    cfg_scale: Optional[float] = None,
    steps: Optional[int] = None,
) -> Tuple[dict, dict]:
    """
    智能拓撲解析與參數注入：
    1. 追蹤 KSampler 的 positive / negative 連結節點
    2. 追蹤 EmptyLatentImage 進行解析度切換
    3. 覆蓋種子碼與取樣設定
    """
    injected_info = {}
    
    # 解析解析度
    width, height = ASPECT_RATIOS.get(aspect_ratio, (1024, 1024))
    injected_info["resolution"] = f"{width}x{height} ({aspect_ratio})"
    injected_info["seed"] = seed

    # 1. 尋找 KSampler 節點
    ksampler_id = None
    positive_node_id = None
    negative_node_id = None
    latent_node_id = None

    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        if "KSampler" in class_type:
            ksampler_id = node_id
            inputs = node.get("inputs", {})
            
            # 從 KSampler 連結反查
            pos_link = inputs.get("positive")
            if isinstance(pos_link, list) and len(pos_link) > 0:
                positive_node_id = str(pos_link[0])
            
            neg_link = inputs.get("negative")
            if isinstance(neg_link, list) and len(neg_link) > 0:
                negative_node_id = str(neg_link[0])
                
            latent_link = inputs.get("latent_image")
            if isinstance(latent_link, list) and len(latent_link) > 0:
                latent_node_id = str(latent_link[0])
                
            # 注入 seed, cfg, steps 到 KSampler
            inputs["seed"] = seed
            if steps is not None and steps > 0:
                inputs["steps"] = steps
                injected_info["steps"] = steps
            if cfg_scale is not None and cfg_scale > 0:
                inputs["cfg"] = cfg_scale
                injected_info["cfg"] = cfg_scale
            break

    # 2. 降級策略：若 KSampler 拓撲未直接指出，則從 Title 或 Class Type 尋找
    if not positive_node_id or not negative_node_id:
        for node_id, node in workflow.items():
            class_type = node.get("class_type", "")
            title = node.get("_meta", {}).get("title", "").lower()
            
            if "CLIPTextEncode" in class_type or "TextEncode" in class_type:
                if not positive_node_id and ("pos" in title or "正面" in title or node_id in ["3", "6"]):
                    positive_node_id = node_id
                elif not negative_node_id and ("neg" in title or "負面" in title or node_id in ["4", "7"]):
                    negative_node_id = node_id

    # 3. 注入 Prompt
    if positive_node_id and positive_node_id in workflow:
        workflow[positive_node_id]["inputs"]["text"] = scene_prompt
        injected_info["positive_node"] = positive_node_id
    else:
        # 最後手段：找到第一個 CLIPTextEncode 塞入正向
        for node_id, node in workflow.items():
            if "CLIPTextEncode" in node.get("class_type", ""):
                node["inputs"]["text"] = scene_prompt
                injected_info["positive_node"] = node_id
                break

    if negative_node_id and negative_node_id in workflow:
        workflow[negative_node_id]["inputs"]["text"] = negative_prompt
        injected_info["negative_node"] = negative_node_id

    # 4. 尋找並更新 Latent 節點尺寸
    target_latent_node = None
    if latent_node_id and latent_node_id in workflow:
        target_latent_node = workflow[latent_node_id]
    else:
        for node_id, node in workflow.items():
            if "EmptyLatentImage" in node.get("class_type", "") or "EmptySD3LatentImage" in node.get("class_type", ""):
                target_latent_node = node
                break
                
    if target_latent_node and "inputs" in target_latent_node:
        target_latent_node["inputs"]["width"] = width
        target_latent_node["inputs"]["height"] = height

    return workflow, injected_info


@mcp.tool()
def get_comfyui_status() -> str:
    """
    檢查本地端 ComfyUI 繪圖引擎連線狀態、顯存與可用工作流。
    在進行繪圖前可用此工具確認 ComfyUI 是否已啟動。
    """
    online, msg = test_comfyui_connection()
    
    # 掃描可用工作流
    workflows = []
    if WORKFLOW_DIR.exists():
        workflows = [f.stem for f in WORKFLOW_DIR.glob("*.json")]
    
    status_report = [
        f"### ComfyUI 狀態檢測",
        f"- **連線狀態**: {'✅ 在線' if online else '❌ 離線'}",
        f"- **位址**: `{COMFYUI_URL}`",
        f"- **詳細資訊**: {msg}",
        f"- **可用工作流範本**: {', '.join(workflows) if workflows else '尚未發現 (workflows/)'}",
        f"- **本機輸出目錄**: `{COMFYUI_OUTPUT_DIR}`"
    ]
    
    if not online:
        status_report.append(
            "\n> ⚠️ **提示**：若尚未啟動 ComfyUI，請在本地執行啟動指令（如 `start_minimax_music3.bat` 或你的 ComfyUI 啟動腳本）。"
        )
        
    return "\n".join(status_report)


@mcp.tool()
def render_art_with_comfyui(
    scene_prompt: str,
    negative_prompt: str = "low quality, bad anatomy, blurry, watermark, deformed, signature, worst quality",
    aspect_ratio: str = "1:1",
    seed: int = -1,
    workflow_name: str = "workflow_sdxl_base",
    cfg_scale: float = 7.0,
    steps: int = 25
) -> str:
    """
    呼叫本地 ComfyUI 無審查算圖引擎渲染圖片。
    
    參數:
    - scene_prompt: 完整的畫面描述（高密度自然語言 + 鏡頭視角、構圖、服裝、光影、材質與風格標籤）。
    - negative_prompt: 負向提示詞，過濾崩壞肢體與不良特徵。
    - aspect_ratio: 寬高比例，支援 '1:1', '16:9', '9:16', '4:3', '3:4', '21:9' 等。
    - seed: 隨機種子碼，-1 表示自動隨機生成。
    - workflow_name: 使用的工作流範本名稱（位於 workflows/ 目錄下，如 'workflow_sdxl_base'）。
    - cfg_scale: 提示詞相關係數（預設 7.0）。
    - steps: 取樣步數（預設 25）。
    """
    # 1. 檢查 ComfyUI 是否在線
    online, conn_msg = test_comfyui_connection()
    if not online:
        return (
            f"❌ 調用失敗：{conn_msg}\n\n"
            f"ComfyUI 目前未在 `{COMFYUI_URL}` 運行。\n"
            f"請先在本機啟動 ComfyUI 後再重新要求繪圖。"
        )

    # 2. 定位並讀取工作流 JSON
    workflow_file = WORKFLOW_DIR / f"{workflow_name}.json"
    if not workflow_file.exists():
        workflow_file = WORKFLOW_DIR / f"{workflow_name}"
    if not workflow_file.exists():
        available = [f.stem for f in WORKFLOW_DIR.glob("*.json")] if WORKFLOW_DIR.exists() else []
        return (
            f"❌ 錯誤：找不到工作流設定檔 `{workflow_name}.json`。\n"
            f"工作流目錄路徑：`{WORKFLOW_DIR}`\n"
            f"目前可用工作流：{available if available else '無'}"
        )

    try:
        with open(workflow_file, "r", encoding="utf-8") as f:
            workflow_data = json.load(f)
    except Exception as e:
        return f"❌ 讀取工作流 JSON 失敗：{str(e)}"

    # 3. 處理隨機種子
    actual_seed = seed if seed != -1 else random.randint(1, 2**32 - 1)

    # 4. 拓撲解析並注入動態參數
    try:
        injected_workflow, meta_info = inject_workflow_parameters(
            workflow=workflow_data,
            scene_prompt=scene_prompt,
            negative_prompt=negative_prompt,
            seed=actual_seed,
            aspect_ratio=aspect_ratio,
            cfg_scale=cfg_scale,
            steps=steps
        )
    except Exception as e:
        return f"❌ 注入參數至工作流時出錯：{str(e)}"

    # 5. 提交至 ComfyUI 佇列並等待完成
    try:
        client_id = str(uuid.uuid4())
        prompt_id = queue_prompt(injected_workflow, client_id)
        image_result = wait_for_completion(prompt_id)
    except Exception as e:
        return f"❌ 算圖執行失敗：{str(e)}"

    # 6. 計算本機絕對路徑與標準 Markdown URI
    filename = image_result["filename"]
    subfolder = image_result.get("subfolder", "")
    
    if subfolder:
        local_abs_path = COMFYUI_OUTPUT_DIR / subfolder / filename
    else:
        local_abs_path = COMFYUI_OUTPUT_DIR / filename
        
    abs_path_str = str(local_abs_path.resolve())
    # 格式化為 file:///URI (將 Windows 反斜線改為正斜線)
    file_uri = f"file:///{abs_path_str.replace('\\', '/')}"

    return (
        f"🎨 **繪製成功！**\n\n"
        f"- **本機檔案路徑**: `{abs_path_str}`\n"
        f"- **圖片 URI**: `{file_uri}`\n"
        f"- **解析度**: {meta_info.get('resolution')}\n"
        f"- **種子碼 (Seed)**: `{actual_seed}`\n"
        f"- **使用工作流**: `{workflow_name}`\n\n"
        f"**預覽語法（請將以下 Markdown 嵌入至對話中直接顯示圖片）**:\n"
        f"![視覺渲染作品]({file_uri})"
    )


@mcp.tool()
def get_a1111_status() -> str:
    """
    檢查本地端 Stable Diffusion WebUI (A1111) 繪圖引擎連線狀態、已載入模型。
    在進行 A1111 繪圖前可用此工具確認服務是否已啟動。
    """
    online, msg = test_a1111_connection()
    status_report = [
        "### A1111 (Stable Diffusion WebUI) 狀態檢測",
        f"- **連線狀態**: {'✅ 在線' if online else '❌ 離線'}",
        f"- **位址**: `{A1111_URL}`",
        f"- **詳細資訊**: {msg}",
        f"- **本機輸出目錄**: `{COMFYUI_OUTPUT_DIR}`"
    ]
    if not online:
        status_report.append(
            "\n> 💡 **提示**：若尚未啟動 A1111，請在本地執行 `A:\\AI_Workspace\\comfyui-mcp\\start_a1111_for_massagelive.bat`。"
        )
    return "\n".join(status_report)


@mcp.tool()
def render_art_with_a1111(
    scene_prompt: str,
    negative_prompt: str = "low quality, bad anatomy, blurry, watermark, deformed, signature, worst quality",
    aspect_ratio: str = "16:9",
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: int = 25,
    cfg_scale: float = 7.0,
    sampler_name: str = "Euler a",
    seed: int = -1,
    anti_yellow_filter: bool = True
) -> str:
    """
    呼叫本地 Stable Diffusion WebUI (A1111) 繪圖引擎渲染圖片，支援防皮膚過黃與自然膚色優化。

    參數:
    - scene_prompt: 完整的畫面描述（自然語言提示詞、鏡頭視角、角色特徵、光影與環境）。
    - negative_prompt: 負向提示詞，過濾崩壞肢體與不良特徵。
    - aspect_ratio: 寬高比例，支援 '16:9', '9:16', '1:1', '4:3', '3:4', '21:9' 等。
    - steps: 取樣步數（預設 25）。
    - cfg_scale: 提示詞相關係數（預設 7.0）。
    - sampler_name: 取樣方法（如 'Euler a', 'DPM++ 2M Karras'）。
    - seed: 隨機種子碼，-1 表示自動隨機。
    - anti_yellow_filter: 是否自動注入防皮膚過黃與過飽和過濾詞（預設 true）。
    """
    online, conn_msg = test_a1111_connection()
    if not online:
        return (
            f"❌ 調用失敗：{conn_msg}\n\n"
            f"A1111 目前未在 `{A1111_URL}` 運行。\n"
            f"請執行 `A:\\AI_Workspace\\comfyui-mcp\\start_a1111_for_massagelive.bat` 啟動後重試。"
        )

    # 寬高計算
    target_width, target_height = 1344, 768
    if aspect_ratio in ASPECT_RATIOS:
        ar_key = f"{aspect_ratio}_sdxl" if f"{aspect_ratio}_sdxl" in ASPECT_RATIOS else aspect_ratio
        target_width, target_height = ASPECT_RATIOS[ar_key]
    if width:
        target_width = width
    if height:
        target_height = height

    # 隨機種子
    actual_seed = seed if seed != -1 else random.randint(1, 2**32 - 1)

    # 防過黃負向濾鏡
    final_negative = negative_prompt
    if anti_yellow_filter and "yellow tint" not in final_negative.lower():
        final_negative = f"{final_negative}, (yellow tint:1.3), (jaundice:1.4), (yellow skin:1.3), (sepia:1.3), (oversaturated yellow:1.3)"

    payload = {
        "prompt": scene_prompt,
        "negative_prompt": final_negative,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "width": target_width,
        "height": target_height,
        "sampler_name": sampler_name,
        "seed": actual_seed
    }

    url = f"{A1111_URL}/sdapi/v1/txt2img"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "Antigravity-MCP"}
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            images = result.get("images", [])
            if not images:
                return "❌ A1111 繪圖失敗：回應中未包含圖片資料"

            img_b64 = images[0]
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]
            img_bytes = base64.b64decode(img_b64)

            COMFYUI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"A1111_{timestamp}_{actual_seed}.png"
            local_abs_path = COMFYUI_OUTPUT_DIR / filename
            with open(local_abs_path, "wb") as f:
                f.write(img_bytes)

            abs_path_str = str(local_abs_path.resolve())
            file_uri = f"file:///{abs_path_str.replace('\\', '/')}"

            return (
                f"🎨 **A1111 繪製成功！**\n\n"
                f"- **本機檔案路徑**: `{abs_path_str}`\n"
                f"- **圖片 URI**: `{file_uri}`\n"
                f"- **解析度**: {target_width}x{target_height} ({aspect_ratio})\n"
                f"- **種子碼 (Seed)**: `{actual_seed}`\n"
                f"- **採樣配置**: `{sampler_name}` ({steps} steps, CFG {cfg_scale})\n\n"
                f"**預覽語法**:\n"
                f"![A1111渲染作品]({file_uri})"
            )
    except Exception as e:
        return f"❌ A1111 算圖執行失敗：{str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
