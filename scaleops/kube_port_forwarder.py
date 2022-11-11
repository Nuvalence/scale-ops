import logging
import os
import subprocess
import time
from contextlib import ContextDecorator
from logging import Logger


# noinspection SubprocessShellMode
class ShellKubePortForwarder(ContextDecorator):
    _logger: Logger

    def __init__(self, kube_context: str,
                 namespace: str, name: str, port: int, sleep: int = 4,
                 service=False):
        self._kube_previous_context = None
        self._kube_context = kube_context
        self._namespace = namespace
        self._name = name
        self._port = port
        self._sleep = sleep
        self._service = service
        self._p = None
        self._shell_env = os.environ.copy()
        self._logger = logging.getLogger(__name__)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self._logger.warn(f'{exc_val} in {exc_type}')
        self.stop()

    def start(self):
        if not self._p:
            previous_context = ShellKubePortForwarder._get_current_context(
                    self._shell_env
            )
            self._kube_previous_context = previous_context.stdout.strip()
            self._logger.info(
                    f'changing kubectx from {self._kube_previous_context} to {self._kube_context}')
            change_context = ShellKubePortForwarder._change_context(
                    self._kube_context,
                    self._shell_env
            )
            new_context = ShellKubePortForwarder._get_current_context(
                    self._shell_env)
            if self._kube_context != new_context.stdout.strip():
                raise RuntimeError(
                        f'Expected {self._kube_context} but got {new_context.stdout}')
            self._logger.info(
                    f'changing kubectx from {self._kube_previous_context} with result: {change_context.stdout}')

            if self._service:
                forward_type = 'service'
                forward_name = f'svc/{self._name}'
            else:
                forward_type = 'pod'
                forward_name = ShellKubePortForwarder._get_pod_name_by_prefix(
                        self._shell_env,
                        self._name
                )
                forward_name = forward_name.stdout.strip()
            self._logger.info(
                    f'port-forwarding to {forward_type} {forward_name} with ports 9090:{self._port}'
            )

            self._p = ShellKubePortForwarder._get_portforward_proc(
                    self._namespace,
                    forward_name,
                    self._port,
                    self._shell_env,
                    self._sleep,
                    self._logger.level == logging.DEBUG
            )

    def stop(self):
        if self._p:
            self._p.terminate()
            try:
                stdout_data, stderr_data = self._p.communicate(timeout=0.2)
                self._logger.debug(
                        f'== subprocess exited with rc = {self._p.returncode}'
                )
                self._logger.debug(stdout_data.decode(
                        'utf-8')) if stdout_data is not None else None
                self._logger.error(stderr_data.decode(
                        'utf-8')) if stderr_data is not None else None
            except subprocess.TimeoutExpired:
                self._p.kill()
                self._logger.debug('subprocess did not terminate in time.')
            if self._kube_previous_context:
                old_context = ShellKubePortForwarder._change_context(
                        self._kube_previous_context,
                        self._shell_env
                )
                self._logger.info(
                        f'reset context to {self._kube_context} result {old_context}')
            self._p = None

    @staticmethod
    def _get_portforward_proc(namespace, name, port, shell_env, sleep,
                              show_output):
        cmd = f'kubectl port-forward --namespace {namespace} {name} 9090:{port}'
        p = subprocess.Popen(
                cmd.split(),
                env=shell_env,
                stdout=subprocess.PIPE if show_output else subprocess.DEVNULL,
                stderr=subprocess.PIPE if show_output else subprocess.DEVNULL
        )
        time.sleep(sleep)
        return p

    @staticmethod
    def _get_pod_name_by_prefix(shell_env, prefix):
        return subprocess.run(
                f"kubectl get pods --namespace monitoring | grep {prefix} | head -n 1 | awk '{{ print $1 }}'",
                check=True,
                shell=True,
                capture_output=True,
                text=True,
                env=shell_env
        )

    @staticmethod
    def _get_current_context(shell_env):
        return subprocess.run(
                'kubectx -c',
                check=True,
                shell=True,
                capture_output=True,
                text=True,
                env=shell_env
        )

    @staticmethod
    def _change_context(context, shell_env):
        return subprocess.run(
                args=f'kubectx {context}',
                check=True,
                shell=True,
                capture_output=True,
                text=True,
                env=shell_env
        )
