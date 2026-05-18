# dynamixel_cfg.py
import yaml
from typing import Any

class DynamixelCfg:
    """
    YAML を読み込み、ネストした dict をドットアクセス可能にするラッパ。
    例:
        cfg = DynamixelCfg("./config/Dynamixel_config.yaml")
        print(cfg.id.port)              # '/dev/ttyUSB0'
        print(cfg.pos.gripper.close)    # 111
    """
    class _DotDict(dict):
        """内部用: dict をドットアクセス可能にする"""
        def __getattr__(self, key: str) -> Any:
            if key in self:
                return self[key]
            # dict として存在しない属性アクセスは AttributeError を投げる
            raise AttributeError(f"{key!r} not found")

        def __setattr__(self, key: str, value: Any) -> None:
            # 通常はキーとして格納
            self[key] = value

        def __delattr__(self, key: str) -> None:
            try:
                del self[key]
            except KeyError:
                raise AttributeError(f"{key!r} not found")

    def __init__(self, file_path: str):
        self._path = file_path
        with open(file_path, "r") as f:
            data = yaml.safe_load(f) or {}
        # 再帰的に _DotDict / list を構成
        self._data = self._to_dot(data)

    # ------- パブリックAPI -------
    def to_dict(self) -> dict:
        """生の dict を返す（必要ならシリアライズ等に）"""
        return self._to_plain(self._data)

    def __getattr__(self, key: str) -> Any:
        # cfg.id のように最上位にもドットアクセスを提供
        if isinstance(self._data, dict) and key in self._data:
            return self._data[key]
        raise AttributeError(f"{key!r} not found at top-level")

    def __repr__(self) -> str:
        return f"DynamixelCfg(path={self._path!r})"

    # ------- ユーティリティ（任意）-------
    def print_values_and_types(self) -> None:
        """全キーの値と型を 'a.b.c : value (type)' 形式で出力"""
        def walk(node, prefix: str = ""):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, prefix + k + ".")
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, prefix + f"[{i}]" + ".")
            else:
                tname = type(node).__name__
                keypath = prefix[:-1]  # 末尾のドットを削除
                print(f"{keypath} : {node} ({tname})")
        walk(self._data)

    # ------- 内部: 変換器 -------
    @classmethod
    def _to_dot(cls, obj: Any) -> Any:
        """dict を _DotDict、list を再帰的に変換"""
        if isinstance(obj, dict):
            dd = cls._DotDict()
            for k, v in obj.items():
                dd[k] = cls._to_dot(v)
            return dd
        elif isinstance(obj, list):
            return [cls._to_dot(v) for v in obj]
        else:
            return obj

    @classmethod
    def _to_plain(cls, obj: Any) -> Any:
        """_DotDict/list をふつうの dict/list に戻す"""
        if isinstance(obj, cls._DotDict):
            return {k: cls._to_plain(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [cls._to_plain(v) for v in obj]
        else:
            return obj

if __name__ == '__main__':
    
    cfg = DynamixelCfg("./config/Dynamixel_config.yaml")
    
    print(cfg.id.port)
    cfg.print_values_and_types()