"""
power_realtime 시드 생성 (수업 전 강사 실행)

- 30일 × 24시간 = 720시간, 시간당 1행 (measured_at = 정각)
- device_id 기본값: RP2040-EMU-01 (실습매뉴얼 .env 와 동일)
- 일출~일몰 형태의 합성 발전량 → power_hourly VIEW / LSTM join 용

사용 예:
  uv run python seed/generate_power_seed.py
  uv run python seed/generate_power_seed.py --days 30 --sql seed/power_realtime_seed.sql
  uv run python seed/generate_power_seed.py --import-db
"""

from __future__ import annotations

import argparse
import math
import os
import random
from datetime import date, datetime, timedelta
from pathlib import Path


def _solar_power_kw(hour: int, day_offset: int, rng: random.Random) -> float:
    """6~19시 부근 종형 곡선 + 약간의 일별·난수 변동."""
    if hour < 5 or hour > 20:
        return round(rng.uniform(0.0, 0.05), 3)
    x = (hour - 5) / 15.0 * math.pi
    peak = 4.5 + 0.3 * math.sin(day_offset * 0.4)
    base = max(0.0, math.sin(x) * peak)
    noise = rng.uniform(-0.15, 0.15)
    return round(max(0.0, base + noise), 3)


def _panel_temp(hour: int, rng: random.Random) -> float:
    ambient = 12 + 8 * math.sin((hour - 14) / 24 * 2 * math.pi)
    return round(ambient + rng.uniform(-1.5, 1.5), 2)


def _panel_humidity(hour: int, rng: random.Random) -> float:
    base = 55 - 10 * math.sin((hour - 14) / 24 * 2 * math.pi)
    return round(max(20.0, min(95.0, base + rng.uniform(-3, 3))), 2)


def build_rows(
    days: int = 30,
    end: date | None = None,
    device_id: str = "RP2040-EMU-01",
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    end = end or (date.today() - timedelta(days=1))
    start = end - timedelta(days=days - 1)
    rows: list[dict] = []
    day_offset = 0
    cur = start
    while cur <= end:
        for hour in range(24):
            ts = datetime(cur.year, cur.month, cur.day, hour, 0, 0)
            p = _solar_power_kw(hour, day_offset, rng)
            rows.append(
                {
                    "measured_at": ts,
                    "device_id": device_id,
                    "power_kw": p,
                    "temperature": _panel_temp(hour, rng),
                    "humidity": _panel_humidity(hour, rng),
                    "raw_payload": f"seed:regs=[{int(p*100)},{int(_panel_temp(hour, rng)*10)},{int(_panel_humidity(hour, rng)*10)}]",
                }
            )
        day_offset += 1
        cur += timedelta(days=1)
    return rows


def write_sql(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "-- power_realtime 시드 (30일×24h). 수업 전 import.",
        "USE weather;",
        "",
    ]
    for r in rows:
        ts = r["measured_at"].strftime("%Y-%m-%d %H:%M:%S")
        lines.append(
            "INSERT INTO power_realtime "
            "(measured_at, device_id, power_kw, temperature, humidity, raw_payload) "
            f"VALUES ('{ts}', '{r['device_id']}', {r['power_kw']}, "
            f"{r['temperature']}, {r['humidity']}, '{r['raw_payload']}') "
            "ON DUPLICATE KEY UPDATE "
            "power_kw=VALUES(power_kw), temperature=VALUES(temperature), "
            "humidity=VALUES(humidity), raw_payload=VALUES(raw_payload);"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] SQL {len(rows)}건 → {path}")


def import_mysql(rows: list[dict]) -> None:
    import pymysql
    from dotenv import load_dotenv

    load_dotenv()
    conn = pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "weather"),
        password=os.getenv("MYSQL_PASSWORD", "weatherpass"),
        database=os.getenv("MYSQL_DATABASE", "weather"),
        charset="utf8mb4",
    )
    try:
        with conn.cursor() as cur:
            for r in rows:
                cur.execute(
                    """
                    INSERT INTO power_realtime
                    (measured_at, device_id, power_kw, temperature, humidity, raw_payload)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                      power_kw=VALUES(power_kw),
                      temperature=VALUES(temperature),
                      humidity=VALUES(humidity),
                      raw_payload=VALUES(raw_payload)
                    """,
                    (
                        r["measured_at"],
                        r["device_id"],
                        r["power_kw"],
                        r["temperature"],
                        r["humidity"],
                        r["raw_payload"],
                    ),
                )
        conn.commit()
        print(f"[OK] MySQL power_realtime {len(rows)}건 저장")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="power_realtime 30일 시드 생성")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (기본: 어제)")
    parser.add_argument("--device-id", type=str, default="RP2040-EMU-01")
    parser.add_argument("--sql", type=str, default="seed/power_realtime_seed.sql")
    parser.add_argument("--import-db", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    end = date.fromisoformat(args.end) if args.end else None
    rows = build_rows(days=args.days, end=end, device_id=args.device_id, seed=args.seed)
    write_sql(rows, Path(args.sql))
    if args.import_db:
        import_mysql(rows)


if __name__ == "__main__":
    main()
