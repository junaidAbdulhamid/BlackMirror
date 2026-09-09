"""Atomic, immutable checkpoints and an exclusive duplicate-start lock."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from blackmirror.resimulation.schemas import ResimulationResult, StopReason


class ResimulationStore:
    def __init__(self, artifact_root: Path) -> None:
        self.root = Path(artifact_root) / "resimulations"

    def directory(self, resimulation_id: str) -> Path:
        _validate_id(resimulation_id)
        return self.root / resimulation_id

    def read(self, resimulation_id: str) -> ResimulationResult:
        return ResimulationResult.model_validate_json(
            (self.directory(resimulation_id) / "state.json").read_text(encoding="utf-8")
        )

    def exists(self, resimulation_id: str) -> bool:
        return (self.directory(resimulation_id) / "state.json").is_file()

    def checkpoint(self, result: ResimulationResult) -> Path:
        destination = self.directory(result.request.resimulation_id)
        destination.mkdir(parents=True, exist_ok=True)
        payload = result.model_dump_json(indent=2)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        history = destination / "history"
        history.mkdir(exist_ok=True)
        snapshot = history / f"{digest}.json"
        if not snapshot.exists():
            with snapshot.open("x", encoding="utf-8") as immutable:
                immutable.write(payload)
                immutable.flush()
                os.fsync(immutable.fileno())
        handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - atomically renamed below
            mode="w",
            encoding="utf-8",
            prefix=".state.",
            suffix=".tmp",
            dir=destination,
            delete=False,
        )
        temporary = Path(handle.name)
        try:
            with handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination / "state.json")
            _fsync(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    @contextmanager
    def lock(self, resimulation_id: str) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / f".{resimulation_id}.lock"
        descriptor = self._acquire_lock(lock_path, resimulation_id)
        try:
            yield
        finally:
            os.close(descriptor)
            lock_path.unlink(missing_ok=True)

    @staticmethod
    def _acquire_lock(lock_path: Path, resimulation_id: str) -> int:
        for _ in range(2):
            descriptor, temporary_text = tempfile.mkstemp(
                prefix=f".{resimulation_id}.owner.", dir=lock_path.parent
            )
            temporary = Path(temporary_text)
            os.write(descriptor, str(os.getpid()).encode())
            os.fsync(descriptor)
            try:
                os.link(temporary, lock_path)
                temporary.unlink()
                return descriptor
            except FileExistsError as exc:
                os.close(descriptor)
                temporary.unlink(missing_ok=True)
                try:
                    owner = int(lock_path.read_text(encoding="utf-8"))
                    os.kill(owner, 0)
                except OSError:
                    lock_path.unlink(missing_ok=True)
                    continue
                except ValueError:
                    raise RuntimeError(
                        f"resimulation {resimulation_id!r} has an unreadable active lock"
                    ) from exc
                raise RuntimeError(f"resimulation {resimulation_id!r} is already active") from exc
        raise RuntimeError(f"could not recover lock for resimulation {resimulation_id!r}")

    def active_owner(self, resimulation_id: str) -> int | None:
        """The pid of a live worker holding this run's lock, if there is one.

        A run whose status is ``active`` but whose worker died leaves no trace
        in its own state file, because the process that would have written the
        failure is the process that vanished. Reading the lock is the only way
        to tell "still running" from "abandoned, press resume", so the answer is
        derived here on demand and never persisted, where it could go stale.
        """
        _validate_id(resimulation_id)
        lock_path = self.root / f".{resimulation_id}.lock"
        try:
            owner = int(lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        try:
            os.kill(owner, 0)
        except OSError:
            return None
        return owner

    def recover(self, resimulation_id: str) -> ResimulationResult:
        """Ignore abandoned temp dirs; only an atomically renamed state is durable."""
        return self.read(resimulation_id)

    def request_stop(self, resimulation_id: str, reason: StopReason) -> None:
        path = self.directory(resimulation_id) / "stop.request"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(reason.value, encoding="utf-8")
        os.replace(temporary, path)

    def stop_requested(self, resimulation_id: str) -> StopReason | None:
        path = self.directory(resimulation_id) / "stop.request"
        return StopReason(path.read_text(encoding="utf-8")) if path.is_file() else None

    def clear_stop(self, resimulation_id: str) -> None:
        (self.directory(resimulation_id) / "stop.request").unlink(missing_ok=True)


def _validate_id(value: str) -> None:
    if value in {".", ".."} or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value) is None:
        raise ValueError("invalid resimulation id")


def _fsync(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
        os.close(descriptor)
    except OSError:
        pass
