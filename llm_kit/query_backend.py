"""
クエリバックエンド抽象（ポート＆アダプタ）

エージェントのツールが必要とするデータアクセスを、ストレージから切り離すための層。
ツール側（tools.py）は QueryBackend のメソッドだけを呼び、結果の整形のみを担う。
スキーマや接続方法（SQLite / 将来的に Athena 等）は各アダプタに閉じ込める。

現状は SQLite アダプタのみ。バックエンドを増やすときは:
  1. QueryBackend を継承したアダプタを足す
  2. make_backend() のディスパッチに分岐を1つ足す
だけで、tools.py / nodes.py は無変更で切り替わる。

ベクトル検索（vector_search_intents）は SQLite(sqlite_vec) 固有。Athena 等には近傍検索が
無いため、別バックエンドではここを別系統（専用ベクトルストア）に逃がす想定。今は未対応。
"""

from __future__ import annotations

import struct
from abc import ABC, abstractmethod

MAX_ROWS = 500


class QueryBackend(ABC):
    """データアクセスのポート。返すのは生データ（整形はツール側の責務）。"""

    @abstractmethod
    def select(self, sql: str, params: tuple = ()) -> dict:
        """任意の SELECT/WITH を実行（execute_sql 用）。
        成功: {sql, columns, rows, row_count, truncated} / 失敗: {error}"""

    @abstractmethod
    def keyword_search_intents(
        self, keyword: str, scope: str | None = None, limit: int = 30
    ) -> list[tuple]:
        """intents をキーワード部分一致で検索。各行 (intent_id, name, kind, intent_type)。"""

    @abstractmethod
    def vector_search_intents(
        self, query: str, top_k: int = 10, scope: str | None = None
    ) -> list[tuple]:
        """intents をベクトル類似度で検索。各行 (intent_id, name, kind, intent_type, parent_id, distance)。"""

    @abstractmethod
    def scope_knowledge_sets(self, scope: str) -> tuple[set, set]:
        """(スコープ内で採用可能な knowledge_id 集合, 全 knowledge_id 集合) を返す。"""

    @abstractmethod
    def knowledge_titles(self) -> dict[str, str]:
        """knowledge_id -> title の辞書（レビューCSV等のオフライン用途）。"""


class SqliteBackend(QueryBackend):
    """analysis.db（SQLite）アダプタ。sqlite3 / sqlite_vec はここに閉じる。"""

    def __init__(self, db_path: str, max_rows: int = MAX_ROWS):
        self.db_path = db_path
        self.max_rows = max_rows
        self._scope_cache: dict[str, tuple[set, set]] = {}

    def _connect(self):
        import sqlite3

        return sqlite3.connect(self.db_path)

    def select(self, sql: str, params: tuple = ()) -> dict:
        stripped = sql.strip().upper()
        if not stripped.startswith("SELECT") and not stripped.startswith("WITH"):
            return {"error": "SELECT文またはWITH句（CTE）のみ実行できます。"}
        try:
            conn = self._connect()
            cursor = conn.execute(sql, params)
            rows = cursor.fetchmany(self.max_rows + 1)
            columns = [desc[0] for desc in cursor.description]
            truncated = len(rows) > self.max_rows
            if truncated:
                rows = rows[: self.max_rows]
            rows_list = [list(row) for row in rows]
            conn.close()
            return {
                "sql": sql,
                "columns": columns,
                "rows": rows_list,
                "row_count": len(rows_list),
                "truncated": truncated,
            }
        except Exception as e:  # noqa: BLE001
            return {"error": f"SQLエラー: {e}"}

    def keyword_search_intents(
        self, keyword: str, scope: str | None = None, limit: int = 30
    ) -> list[tuple]:
        conn = self._connect()
        try:
            if scope:
                return conn.execute(
                    """
                    SELECT i.id, i.name, i.kind, i.intent_type
                    FROM intents i
                    JOIN intent_scopes s ON i.id = s.intent_id
                    WHERE i.name LIKE ? AND s.scope_value = ?
                    ORDER BY LENGTH(i.name)
                    LIMIT ?
                    """,
                    (f"%{keyword}%", scope, limit),
                ).fetchall()
            return conn.execute(
                """
                SELECT id, name, kind, intent_type
                FROM intents
                WHERE name LIKE ?
                ORDER BY LENGTH(name)
                LIMIT ?
                """,
                (f"%{keyword}%", limit),
            ).fetchall()
        finally:
            conn.close()

    def vector_search_intents(
        self, query: str, top_k: int = 10, scope: str | None = None
    ) -> list[tuple]:
        import sqlite_vec

        from llm_kit.embedding import embed_query

        query_vec = embed_query(query, provider="openai", model="text-embedding-3-small")
        query_blob = struct.pack(f"{len(query_vec)}f", *query_vec)

        conn = self._connect()
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        try:
            scoped_ids = None
            if scope:
                scoped_ids = {
                    r[0]
                    for r in conn.execute(
                        "SELECT intent_id FROM intent_scopes WHERE scope_value = ?",
                        (scope,),
                    ).fetchall()
                }

            k = top_k if not scoped_ids else min(top_k * 5, 500)
            rows = conn.execute(
                """
                SELECT e.intent_id, i.name, i.kind, i.intent_type, i.parent_id, e.distance
                FROM intent_embeddings e
                JOIN intents i ON e.intent_id = i.id
                WHERE e.embedding MATCH ?
                  AND k = ?
                ORDER BY e.distance
                """,
                (query_blob, k),
            ).fetchall()
        finally:
            conn.close()

        if scoped_ids is not None:
            return [r for r in rows if r[0] in scoped_ids][:top_k]
        return rows[:top_k]

    def scope_knowledge_sets(self, scope: str) -> tuple[set, set]:
        if scope not in self._scope_cache:
            con = self._connect()
            allowed = {
                r[0]
                for r in con.execute(
                    "SELECT DISTINCT ik.knowledge_id FROM intent_knowledges ik "
                    "JOIN intent_scopes s ON ik.intent_id = s.intent_id "
                    "WHERE s.scope_value = ?",
                    (scope,),
                )
            }
            allk = {r[0] for r in con.execute("SELECT id FROM knowledges")}
            con.close()
            self._scope_cache[scope] = (allowed, allk)
        return self._scope_cache[scope]

    def knowledge_titles(self) -> dict[str, str]:
        con = self._connect()
        try:
            return {str(i): t for i, t in con.execute("SELECT id, title FROM knowledges")}
        finally:
            con.close()


# データソース指定 -> バックエンド（同一指定はキャッシュして使い回す）
_BACKEND_CACHE: dict[str, QueryBackend] = {}


def make_backend(spec: str) -> QueryBackend:
    """データソース指定からバックエンドを生成する。

    現状は SQLite のみ（spec = DBファイルパス）。
    将来 'athena://...' のようなURIスキームで分岐を足す想定。
    """
    if spec not in _BACKEND_CACHE:
        # TODO: spec のスキームで分岐（例: athena:// → AthenaBackend）
        _BACKEND_CACHE[spec] = SqliteBackend(spec)
    return _BACKEND_CACHE[spec]
