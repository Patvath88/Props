# /balldontlie_client.py
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests


@dataclass(frozen=True)
class APIConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # if provided, validate; else auto-detect


KNOWN_BASES = (
    "https://api.balldontlie.io/v1",     # Pro/current
    "https://api.balldontlie.io/v2",     # Pro/alt
    "https://www.balldontlie.io/api/v1", # Legacy free
)


class BallDontLieClient:
    def __init__(self, cfg: APIConfig):
        self.api_key = (cfg.api_key or "").strip() or None
        self._explicit_base = (cfg.base_url or "").rstrip("/") or None
        self.base_url: Optional[str] = None  # resolved lazily
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    # ---- public helpers ----
    def search_players(self, search: str) -> List[Dict]:
        return self._paged("players", {"search": search}, per_page=50, max_pages=2)

    def player_stats_by_seasons(self, player_id: int, seasons: List[int]) -> List[Dict]:
        rows: List[Dict] = []
        for s in seasons:
            rows.extend(self._paged("stats", {"player_ids[]": player_id, "seasons[]": s}))
        return rows

    def next_team_game(self, team_id: int, start_date) -> Optional[Dict]:
        data = self._get("games", {"team_ids[]": team_id, "start_date": str(start_date), "per_page": 1, "page": 1})
        items = data.get("data", [])
        return items[0] if items else None

    # ---- core HTTP ----
    def _paged(self, path: str, params: Dict, per_page: int = 100, max_pages: int = 50) -> List[Dict]:
        out: List[Dict] = []
        page = 1
        while page <= max_pages:
            p = dict(params)
            p.update({"page": page, "per_page": per_page})
            data = self._get(path, p)
            items = data.get("data", [])
            out.extend(items)
            meta = data.get("meta", {}) or {}
            if not meta.get("next_page") or len(items) < per_page:
                break
            page += 1
        return out

    def _get(self, path: str, params: Dict) -> Dict:
        if self.base_url is None:
            self.base_url = self._resolve_base_or_raise()
        url = f"{self.base_url}/{path.lstrip('/')}"
        for attempt in range(5):
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 404 and attempt == 0:
                # why: user-provided base may be wrong or API moved; re-probe once
                self.base_url = self._resolve_base_or_raise(force=True)
                url = f"{self.base_url}/{path.lstrip('/')}"
                continue
            if resp.status_code == 429 and attempt < 4:
                time.sleep(1.0 + 0.5 * attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("Exhausted retries calling BALLDONTLIE.")

    # ---- base resolution ----
    def _resolve_base_or_raise(self, force: bool = False) -> str:
        if self._explicit_base and not force:
            self._assert_route_exists(self._explicit_base)
            return self._explicit_base.rstrip("/")

        tried: List[Tuple[str, int]] = []
        for base in ([self._explicit_base] if self._explicit_base else []) + list(KNOWN_BASES):
            if not base:
                continue
            try:
                code = self._probe_players_endpoint(base)
                if code in (200, 401):  # 401 is fine → route exists
                    return base.rstrip("/")
                tried.append((base, code))
            except requests.RequestException:
                tried.append((base, -1))
        detail = ", ".join([f"{b}→{c}" for b, c in tried]) or "no bases tried"
        raise RuntimeError(
            "No working BALLDONTLIE base URL found. "
            f"Tried: {detail}. 404 → wrong path/domain. "
            "Use https://api.balldontlie.io/v1 (Pro) or set an explicit base."
        )

    def _probe_players_endpoint(self, base: str) -> int:
        url = f"{base.rstrip('/')}/players"
        resp = self.session.get(url, params={"per_page": 1, "search": "a"}, timeout=15)
        return resp.status_code

    def _assert_route_exists(self, base: str) -> None:
        code = self._probe_players_endpoint(base)
        if code not in (200, 401):
            raise RuntimeError(
                f"Provided base invalid for /players: {base} → HTTP {code}."
            )
