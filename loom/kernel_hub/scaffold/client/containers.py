import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Mount:
    source: str
    target: str
    readonly: bool = False

    def as_arg(self) -> str:
        suffix = ":ro" if self.readonly else ""
        return f"{self.source}:{self.target}{suffix}"


@dataclass
class ContainerRuntime:
    binary: str

    @classmethod
    def detect(cls, preferred: str | None = None) -> "ContainerRuntime":
        candidates = [preferred] if preferred else ["docker", "podman"]
        for name in candidates:
            if name and shutil.which(name):
                return cls(binary=name)
        raise RuntimeError(f"no container runtime found (looked for: {candidates})")

    @property
    def is_podman(self) -> bool:
        return self.binary == "podman"

    def _run(self, args: list[str], capture: bool = False) -> str:
        result = subprocess.run(
            [self.binary, *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
        )
        return result.stdout.strip() if capture else ""

    def build(self, image: str, dockerfile: Path, context: Path) -> None:
        self._run(["build", "-t", image, "-f", str(dockerfile), str(context)])

    def exists(self, name: str) -> bool:
        out = self._run(["ps", "-aq", "--filter", f"name={name}"], capture=True)
        return bool(out)

    def list_by_prefix(self, prefix: str) -> list[str]:
        out = self._run(
            ["ps", "-a", "--filter", f"name={prefix}", "--format", "{{.Names}}"],
            capture=True,
        )
        return [line for line in out.splitlines() if line]

    def remove(self, name: str, force: bool = True) -> None:
        args = ["rm"]
        if force:
            args.append("-f")
        args.append(name)
        subprocess.run([self.binary, *args], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self, name: str) -> None:
        subprocess.run([self.binary, "stop", name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def run_detached(self, name: str, image: str, command: list[str], env: dict[str, str],
                     mounts: list[Mount], network: str = "host", user: str | None = None) -> str:
        args = ["run", "-dt", "--name", name, f"--network={network}"]
        if user is not None:
            args += ["--user", user]
        for key, value in env.items():
            args += ["-e", f"{key}={value}"]
        for mount in mounts:
            args += ["-v", mount.as_arg()]
        args += [image, *command]
        return self._run(args, capture=True)

    def logs_command(self, name: str) -> str:
        return f"{self.binary} logs -f {name}"
