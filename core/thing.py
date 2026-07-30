from __future__ import annotations

import uuid
from typing import Any, Dict, Optional


class Thing:
    """
    事物 - 宇宙中的实体。

    每个事物具有唯一标识、属性集合，并可与其他事物形成连接。
    事物的本质由其与其他事物的连接总和所决定。
    """

    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.attributes: Dict[str, Any] = attributes or {}
        self._connections_in: set[str] = set()
        self._connections_out: set[str] = set()

    def add_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def get_attribute(self, key: str, default: Any = None) -> Any:
        return self.attributes.get(key, default)

    def remove_attribute(self, key: str) -> None:
        self.attributes.pop(key, None)

    def register_connection(self, connection_id: str, direction: str = 'out') -> None:
        if direction == 'out':
            self._connections_out.add(connection_id)
        else:
            self._connections_in.add(connection_id)

    def unregister_connection(self, connection_id: str) -> None:
        self._connections_out.discard(connection_id)
        self._connections_in.discard(connection_id)

    @property
    def connection_count(self) -> int:
        return len(self._connections_in) + len(self._connections_out)

    @property
    def is_isolated(self) -> bool:
        return self.connection_count == 0

    def similarity(self, other: Thing) -> float:
        """
        计算两个事物之间的属性相似度 (0-1)。
        这是连接度计算的基础。
        """
        if not self.attributes and not other.attributes:
            return 0.0

        all_keys = set(self.attributes.keys()) | set(other.attributes.keys())
        if not all_keys:
            return 0.0

        match_count = 0
        for key in all_keys:
            v1 = self.attributes.get(key)
            v2 = other.attributes.get(key)
            if v1 is not None and v2 is not None:
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    max_val = max(abs(v1), abs(v2), 1)
                    match_count += 1.0 - min(abs(v1 - v2) / max_val, 1.0)
                elif v1 == v2:
                    match_count += 1.0
                else:
                    match_count += 0.0
            else:
                match_count += 0.0

        return match_count / len(all_keys)

    def __repr__(self) -> str:
        return f"<Thing: {self.name} ({self.id}) attributes={self.attributes}>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Thing):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
