import logging
import os
import re
import shlex, subprocess
import tempfile
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

            try:
                (self._p, _) = ShellKubePortForwarder._get_portforward_proc(
                    self._namespace,
                    forward_name,
                    self._port,
                    self._shell_env,
                )
            except subprocess.CalledProcessError as cpe:
                self._logger.error(f'Command `{cpe.cmd}` failed with status {cpe.returncode} and stderr:\n{cpe.stderr}')
                raise with_refined_feedback(cpe)

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
    def _get_portforward_proc(namespace, name, port, shell_env, stdout=None, stderr=None, host_port=9090, timeout=10):
        """
        Spawn a process for port forwarding.

        This returns a tuple of (process, host_port) - only the first is relevant at the moment but it seemed worth adding the
        second while messing with relevant code.
        Passing an empty host_port allows the kernel to select an available port (and therefore returning it is useful).

        This is different than the others since background process/job management is not supported by the
        higher level ``subprocess`` functions.

        This can be passed file handles of ``stdout`` and optionally ``stderr`` (defaults to reusing ``stdout``)
        to which outputs will be redirected. These outputs will then be polled to detect errors or success.
        Relying on tempfile should allow the files to be cleaned up when it is no longer referenced,
        but otherwise this should be owned by something more aligned with the lifecycle.
        """
        if (not stdout):
            stdout = tempfile.TemporaryFile()
        if (not stderr):
            stderr = stdout
        cmd = f'kubectl port-forward --namespace {namespace} {name} {host_port}:{port}'
        p = subprocess.Popen(
            shlex.split(cmd),
            env=shell_env,
            stderr=stdout,
            stdout=stderr,
            text=True,
        )
        # Define output which indicates forwarding is ready.
        success_output = re.compile("Forwarding from 127\.0\.0\.1:(\d+)")
        for _ in range(timeout):
            # Mind the location as we consume from handles.
            stdout.seek(0)

            # There's some stringification stuff in here which would likely be better handled by preferring the
            # buffers, but this seems reasonable given the use.

            # If the process has completed it indicates failure, so surface relevant information.
            if (p.poll()):
                error_stdout = f'{stdout.read()}'
                error_stderr = error_stdout if stderr == stdout else f'{stderr.read()}'
                raise subprocess.CalledProcessError(p.returncode, cmd, error_stdout, error_stderr)

            # If success message has been output to file return process and captured host port.
            forwarding = success_output.search(f'{stdout.read()}')
            if (forwarding):
                return (p, forwarding.group(1))
            time.sleep(1)
        raise RuntimeError(f'Call to {cmd} did not complete in {timeout} seconds.')

    @staticmethod
    def _get_pod_name_by_prefix(shell_env, prefix):
        return subprocess.run(
            f"kubectl get pods --namespace monitoring | grep {prefix} | head -n 1 | awk '{{ print $1 }}'",
            **with_default_process_args(env=shell_env))

    @staticmethod
    def _get_current_context(shell_env):
        return subprocess.run(
            'kubectx -c',
            **with_default_process_args(env=shell_env))

    @staticmethod
    def _change_context(context, shell_env):
        return subprocess.run(**with_default_process_args(
            args=f'kubectx {context}',
            env=shell_env))

def with_default_process_args(**kwargs):
    """
    Provide local default parameters for subprocesses.
    This can be called with keyword argument overrides
    and should be unpacked into the underlying ``subprocess`` call such as::
    **with_default_process_args(my_override: 'Foo')
    """
    return {
        'check': True,
        'shell': True,
        'capture_output': True,
        'text': True,
        **kwargs,
    }

def with_refined_feedback(cpe: subprocess.CalledProcessError):
    """
    Attempt to provide better a more helpful error based on output.
    This can/should be refiend as neeeded.
    """
    if ('MFA' in cpe.stderr):
        return Exception('Refresh your MFA or SSO session outside of this program!')
    return cpe

