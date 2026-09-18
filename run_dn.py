import os
import re
import json
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ============================================================
# 設定
# ============================================================

BASE_URL = "https://www.tropicaltidbits.com/analysis/models/ecmwf/"

# 輸出資料夾
OUTPUT_DIR = Path("./")

# 是否保留舊 run
KEEP_OLD_RUNS = True

# 預報時效
FORECASTS = {
    8: 24,
    17: 48,
    25: 72,
    33: 96,
    41: 120
}

# 需要下載的三種圖
PRODUCTS = {
    "mslp_uv850": "ecmwf_mslp_uv850_wpac",
    "z500_vort": "ecmwf_z500_vort_wpac",
    "mslp_pcpn": "ecmwf_mslp_wind_wpac"
}

# HTTP
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/142.0.0.0 Safari/537.36"
    )
}

TIMEOUT = 30


# ============================================================
# 建立資料夾
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 下載 HTML
# ============================================================

def get_model_index():
    print("=" * 70)
    print("正在取得 ECMWF 最新 run 列表...")
    print(BASE_URL)
    print("=" * 70)

    response = requests.get(
        BASE_URL,
        headers=HEADERS,
        timeout=TIMEOUT
    )

    response.raise_for_status()

    return response.text


# ============================================================
# 找出 YYYYMMDDHH 資料夾
# ============================================================

def find_available_runs(html):

    # 嚴格限制為：
    # 2026091806/
    # 2026091106/
    #
    # 不會抓：
    # ecmwf/
    # ../
    #
    pattern = r'href="(\d{10})/"'

    runs = re.findall(pattern, html)

    # 去除重複
    runs = sorted(set(runs))

    if not runs:
        raise RuntimeError(
            "找不到任何符合 YYYYMMDDHH 格式的 ECMWF run。"
        )

    return runs


# ============================================================
# 選最新 run
# ============================================================

def get_latest_run(runs):

    latest = max(
        runs,
        key=lambda x: datetime.strptime(x, "%Y%m%d%H")
    )

    return latest


# ============================================================
# 檢查圖片
# ============================================================

# ============================================================
# 檢查圖片
# ============================================================

def download_file(url, output_path):

    print(f"下載：{url}")

    try:
        # 複製全域 HEADERS 並加入 Referer 與 Accept 繞過防盜鏈機制
        req_headers = HEADERS.copy()
        req_headers["Referer"] = "https://www.tropicaltidbits.com/"
        req_headers["Accept"] = "image/webp,image/apng,image/*,*/*;q=0.8"

        response = requests.get(
            url,
            headers=req_headers,
            timeout=TIMEOUT
        )

        response.raise_for_status()

        # 基本圖片檢查
        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        if not response.content:
            raise RuntimeError("檔案為空")

        # Tropical Tidbits 圖片應該是 image/*
        # 但某些伺服器 Content-Type 可能不標準，
        # 所以不強制阻擋。
        if "image" not in content_type:
            print(
                f"⚠ 注意：Content-Type = {content_type}"
            )

        with open(output_path, "wb") as f:
            f.write(response.content)

        print(
            f"✓ 完成：{output_path.name} "
            f"({len(response.content) / 1024:.1f} KB)"
        )

        return True

    except Exception as e:

        print(f"✗ 下載失敗：{e}")

        if output_path.exists():
            try:
                output_path.unlink()
            except:
                pass

        return False

# ============================================================
# 產生 index.json
# ============================================================

def create_index_json(
    run,
    successful_files,
    failed_files
):

    # UTC run
    run_dt = datetime.strptime(
        run,
        "%Y%m%d%H"
    ).replace(tzinfo=timezone.utc)

    # 台灣時間
    taiwan_tz = timezone(
        timedelta(hours=8)
    )

    run_local = run_dt.astimezone(
        taiwan_tz
    )

    index_data = {

        "model": "ECMWF",

        "source": BASE_URL,

        "latest_run": run,

        "run_time_utc": (
            run_dt.strftime(
                "%Y-%m-%d %H:%M UTC"
            )
        ),

        "run_time_taiwan": (
            run_local.strftime(
                "%Y-%m-%d %H:%M CST"
            )
        ),

        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "forecast_hours": [
            24,
            48,
            72,
            96,
            120
        ],

        "files": {},

        "successful_files": successful_files,

        "failed_files": failed_files
    }

    for product in PRODUCTS.keys():

        index_data["files"][product] = {}

        for _, fh in FORECASTS.items():

            filename = (
                f"{product}_f{fh}.png"
            )

            if filename in successful_files:

                index_data["files"][
                    product
                ][str(fh)] = filename

    index_path = OUTPUT_DIR / "index.json"

    with open(
        index_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            index_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 70)
    print(f"✓ index.json 已建立：{index_path}")
    print("=" * 70)


# ============================================================
# 主程式
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. 取得列表
    # --------------------------------------------------------

    html = get_model_index()

    # --------------------------------------------------------
    # 2. 找 run
    # --------------------------------------------------------

    runs = find_available_runs(html)

    print()
    print("找到的 ECMWF runs：")

    for run in runs:
        print(f"  {run}")

    # --------------------------------------------------------
    # 3. 最新 run
    # --------------------------------------------------------

    latest_run = get_latest_run(runs)

    print()
    print("=" * 70)
    print(f"最新 ECMWF run：{latest_run}")
    print("=" * 70)

    # --------------------------------------------------------
    # 4. 建立 run 資料夾
    # --------------------------------------------------------

    run_dir = OUTPUT_DIR / latest_run

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 5. 開始下載
    # --------------------------------------------------------

    successful_files = []
    failed_files = []

    print()

    for product, url_prefix in PRODUCTS.items():

        for forecast_index, forecast_hour in FORECASTS.items():

            # 原始 URL
            #
            # 例如：
            # 2026091806ecmwf_z500_vort_wpac_8.png
            #
            filename_original = (
                f"{url_prefix}"
                f"_{forecast_index}.png"
            )

            url = (
                BASE_URL
                + latest_run
                + "/"
                + filename_original
            )

            # 最終檔名
            #
            # z500_vort_f24.png
            # mslp_uv850_f48.png
            # mslp_pcpn_f72.png

            output_filename = (
                f"{product}_f"
                f"{forecast_hour}.png"
            )

            output_path = (
                run_dir
                / output_filename
            )

            print()
            print("-" * 70)
            print(
                f"{product} | "
                f"F{forecast_hour:03d}"
            )

            success = download_file(
                url,
                output_path
            )

            if success:
                successful_files.append(
                    output_filename
                )
            else:
                failed_files.append(
                    output_filename
                )

    # --------------------------------------------------------
    # 6. 建立 latest 目錄
    # --------------------------------------------------------

    latest_dir = OUTPUT_DIR / "latest"

    latest_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # 複製最新圖片到 latest
    import shutil

    for filename in successful_files:

        src = run_dir / filename
        dst = latest_dir / filename

        shutil.copy2(
            src,
            dst
        )

    # --------------------------------------------------------
    # 7. latest index
    # --------------------------------------------------------

    create_index_json(
        latest_run,
        successful_files,
        failed_files
    )

    # --------------------------------------------------------
    # 8. 額外建立 latest_run.txt
    # --------------------------------------------------------

    with open(
        latest_dir / "latest_run.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(latest_run)

    # --------------------------------------------------------
    # 9. 結果
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("下載完成")
    print("=" * 70)

    print(f"ECMWF run：{latest_run}")
    print(
        f"成功：{len(successful_files)} / 15"
    )
    print(
        f"失敗：{len(failed_files)} / 15"
    )

    print()
    print("最新資料：")
    print(latest_dir)

    print()
    print("Index：")
    print(OUTPUT_DIR / "index.json")

    if failed_files:

        print()
        print("失敗檔案：")

        for filename in failed_files:
            print(f"  - {filename}")


# ============================================================
# 執行
# ============================================================

if __name__ == "__main__":
    main()
