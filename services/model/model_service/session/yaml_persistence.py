"""
YAML Persistence for Door Session (v4.5).

Door Session을 YAML 파일로 저장하고 복구합니다.

v4.5 변경사항:
- Atomic write 패턴 적용 (temp file → rename)
- threading.Lock() 추가로 동시 쓰기 방지

저장 경로:
    data/sessions/
    ├── active/                    # 진행 중
    │   └── door_zone_1_260201_143000.yaml
    └── completed/                 # 완료됨
        └── 2026-02-01/
            └── door_zone_1_260201_143000.yaml

사용법:
    persistence = YamlPersistence(base_dir="data/sessions")

    # 저장
    path = persistence.save(door_session)

    # 로드
    session = persistence.load(path)

    # 복구
    active_sessions = persistence.recover_active_sessions()
"""

import logging
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from .door_session import DoorSession

logger = logging.getLogger(__name__)


class YamlPersistence:
    """
    Door Session YAML 영속화 (v4.5).

    활성 세션은 active/ 디렉토리에,
    완료 세션은 completed/{날짜}/ 디렉토리에 저장합니다.

    Thread-safe: 동시 쓰기 방지를 위한 Lock 사용.
    Atomic write: temp file → rename 패턴으로 파일 손상 방지.
    """

    def __init__(self, base_dir: str = "data/sessions"):
        """
        Initialize YamlPersistence.

        Args:
            base_dir: 기본 저장 디렉토리
        """
        if not HAS_YAML:
            raise ImportError(
                "PyYAML is required for YAML persistence. "
                "Install with: pip install pyyaml"
            )

        self._base_dir = Path(base_dir)
        self._active_dir = self._base_dir / "active"
        self._completed_dir = self._base_dir / "completed"
        self._lock = threading.Lock()  # v4.5: 동시 쓰기 방지

        # 디렉토리 생성
        self._active_dir.mkdir(parents=True, exist_ok=True)
        self._completed_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"YamlPersistence initialized: base_dir={base_dir}")

    def save(self, session: DoorSession) -> Path:
        """
        Door Session을 YAML 파일로 저장.

        v4.5: Atomic write 패턴 (temp file → rename)
              Thread-safe (Lock 사용)

        활성 세션은 active/, 완료 세션은 completed/{날짜}/에 저장.

        Args:
            session: 저장할 DoorSession

        Returns:
            저장된 파일 경로
        """
        filename = f"{session.door_session_id}.yaml"

        if session.status == "active":
            # 활성 세션: active/
            file_path = self._active_dir / filename
            target_dir = self._active_dir
        else:
            # 완료 세션: completed/{날짜}/
            date_str = datetime.fromtimestamp(session.created_at).strftime("%Y-%m-%d")
            date_dir = self._completed_dir / date_str
            date_dir.mkdir(parents=True, exist_ok=True)
            file_path = date_dir / filename
            target_dir = date_dir

            # 활성 디렉토리에서 제거
            active_path = self._active_dir / filename
            if active_path.exists():
                try:
                    active_path.unlink()
                    logger.debug(f"Removed active session file: {active_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove active session file: {e}")

        # v4.5: Atomic write with Lock
        data = session.to_dict()
        with self._lock:
            self._atomic_write(file_path, data, target_dir)

        logger.debug(f"Session saved (atomic): {file_path}")
        return file_path

    def _atomic_write(self, file_path: Path, data: dict, target_dir: Path) -> None:
        """
        Atomic write (temp file → rename).

        파일 손상 방지를 위해 임시 파일에 먼저 쓴 후 rename.

        Args:
            file_path: 최종 파일 경로
            data: 저장할 데이터
            target_dir: 임시 파일 생성 디렉토리
        """
        # 같은 디렉토리에 임시 파일 생성 (rename이 원자적이 되도록)
        fd, temp_path = tempfile.mkstemp(
            suffix=".yaml.tmp",
            prefix="door_session_",
            dir=str(target_dir),
        )
        try:
            # 임시 파일에 쓰기
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.dump(
                    data,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )

            # 원자적 rename (Windows에서는 기존 파일 먼저 삭제 필요)
            temp_path_obj = Path(temp_path)
            if os.name == "nt" and file_path.exists():
                # Windows: 기존 파일 삭제 후 rename
                file_path.unlink()
            temp_path_obj.rename(file_path)

        except Exception as e:
            # 실패 시 임시 파일 정리
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass
            raise e

    def load(self, path: Path) -> Optional[DoorSession]:
        """
        YAML 파일에서 Door Session 로드.

        Args:
            path: YAML 파일 경로

        Returns:
            DoorSession 또는 None (로드 실패)
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            session = DoorSession.from_dict(data)
            logger.debug(f"Session loaded: {path}")
            return session

        except Exception as e:
            logger.error(f"Failed to load session from {path}: {e}")
            return None

    def recover_active_sessions(self) -> Dict[int, DoorSession]:
        """
        서비스 시작 시 활성 세션 복구.

        active/ 디렉토리의 모든 YAML 파일을 로드합니다.

        Returns:
            zone -> DoorSession 딕셔너리
        """
        recovered: Dict[int, DoorSession] = {}

        if not self._active_dir.exists():
            return recovered

        for yaml_file in self._active_dir.glob("*.yaml"):
            try:
                session = self.load(yaml_file)
                if session is not None:
                    # 같은 zone에 여러 세션이 있으면 최신 것 사용
                    if session.zone not in recovered:
                        recovered[session.zone] = session
                    else:
                        existing = recovered[session.zone]
                        if session.last_trigger_at > existing.last_trigger_at:
                            recovered[session.zone] = session
                            logger.warning(
                                f"Multiple active sessions for zone {session.zone}, "
                                f"using newer: {session.door_session_id}"
                            )
            except Exception as e:
                logger.error(f"Failed to recover session from {yaml_file}: {e}")

        if recovered:
            logger.info(
                f"Recovered {len(recovered)} active door sessions: "
                f"{[s.door_session_id for s in recovered.values()]}"
            )

        return recovered

    def delete(self, session: DoorSession) -> bool:
        """
        Door Session 파일 삭제.

        Args:
            session: 삭제할 DoorSession

        Returns:
            삭제 성공 여부
        """
        filename = f"{session.door_session_id}.yaml"

        # active 디렉토리 확인
        active_path = self._active_dir / filename
        if active_path.exists():
            try:
                active_path.unlink()
                logger.debug(f"Deleted active session file: {active_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete active session file: {e}")
                return False

        # completed 디렉토리 확인
        date_str = datetime.fromtimestamp(session.created_at).strftime("%Y-%m-%d")
        completed_path = self._completed_dir / date_str / filename
        if completed_path.exists():
            try:
                completed_path.unlink()
                logger.debug(f"Deleted completed session file: {completed_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete completed session file: {e}")
                return False

        logger.warning(f"Session file not found: {filename}")
        return False

    def cleanup_old_sessions(self, days_to_keep: int = 7) -> int:
        """
        오래된 완료 세션 정리.

        Args:
            days_to_keep: 보관 기간 (일)

        Returns:
            삭제된 파일 수
        """
        deleted_count = 0
        cutoff_date = datetime.now().date()

        for date_dir in self._completed_dir.iterdir():
            if not date_dir.is_dir():
                continue

            try:
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d").date()
                days_old = (cutoff_date - dir_date).days

                if days_old > days_to_keep:
                    # 디렉토리 내 파일 삭제
                    for yaml_file in date_dir.glob("*.yaml"):
                        try:
                            yaml_file.unlink()
                            deleted_count += 1
                        except Exception as e:
                            logger.error(f"Failed to delete {yaml_file}: {e}")

                    # 빈 디렉토리 삭제
                    try:
                        date_dir.rmdir()
                    except OSError:
                        pass  # 디렉토리가 비어있지 않음

            except ValueError:
                # 날짜 형식이 아닌 디렉토리 무시
                continue

        if deleted_count > 0:
            logger.info(
                f"Cleaned up {deleted_count} old session files "
                f"(older than {days_to_keep} days)"
            )

        return deleted_count

    def get_stats(self) -> dict:
        """
        영속화 통계 반환.

        Returns:
            통계 정보
        """
        active_count = len(list(self._active_dir.glob("*.yaml")))

        completed_count = 0
        for date_dir in self._completed_dir.iterdir():
            if date_dir.is_dir():
                completed_count += len(list(date_dir.glob("*.yaml")))

        return {
            "active_count": active_count,
            "completed_count": completed_count,
            "base_dir": str(self._base_dir),
        }
