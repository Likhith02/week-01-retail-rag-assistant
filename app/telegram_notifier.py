from __future__ import annotations

import httpx


async def send_telegram_message(bot_token: str, chat_id: str, text: str) -> bool:
    if not bot_token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code != 200:
                return False
            data = response.json()
            return bool(data.get("ok"))
    except Exception:
        return False
