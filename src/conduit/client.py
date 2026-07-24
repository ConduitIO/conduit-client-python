"""``Client``: a gRPC client bound to a Conduit engine's control-plane API.

Constructed via :func:`conduit.connect` (an already-running, independently
deployed engine -- Mode 2) or :func:`conduit.local` (a supervised subprocess
-- Mode 1). Both return the same :class:`Client` type and expose the identical
surface; only how the underlying channel/process came to exist differs (design
doc, "``conduit.local()`` and ``conduit.connect(addr)``").

Every RPC call here goes through :meth:`Client._call`, which translates any
``grpc.RpcError`` into a :class:`conduit.errors.ConduitError` -- callers never
see a raw gRPC exception or stack trace (CLAUDE.md, "errors are API").
"""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import Protocol, TypeVar

import grpc

import conduit._grpc  # noqa: F401  -- sets up sys.path, see conduit._grpc.__init__
from api.v1 import api_pb2, api_pb2_grpc
from conduit.errors import ConduitError
from conduit.pipeline import Pipeline
from conduit.run import Run

_Req = TypeVar("_Req")
_Resp = TypeVar("_Resp")
_ReqContra = TypeVar("_ReqContra", contravariant=True)
_RespCo = TypeVar("_RespCo", covariant=True)


class _UnaryUnary(Protocol[_ReqContra, _RespCo]):
    """A gRPC unary-unary stub method's call signature.

    The generated stubs (``api_pb2_grpc.py``, vendored, unannotated) bind
    each RPC as a plain callable at ``__init__`` time with no static type --
    this Protocol is hand-written just to give :meth:`Client._call` a real,
    checkable signature (``request``, keyword-only ``timeout``) instead of
    losing type safety at every call site.
    """

    def __call__(self, request: _ReqContra, *, timeout: float | None = ...) -> _RespCo: ...


class Client:
    """Bound to one Conduit engine's control-plane gRPC API.

    Not constructed directly -- use :func:`conduit.connect` or
    :func:`conduit.local`.
    """

    def __init__(
        self,
        channel: grpc.Channel,
        *,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        """Wrap a gRPC channel with the control-plane stubs this client drives.

        Args:
            channel: an already-constructed gRPC channel (insecure loopback
                for :func:`conduit.local`, whatever :func:`conduit.connect`
                was given).
            on_close: called once, after the channel closes, by
                :meth:`close`/``__exit__`` -- :func:`conduit.local` uses this
                to tear down the supervised subprocess; :func:`conduit.connect`
                leaves it ``None`` (no process to own).
        """
        self._channel = channel
        self._on_close = on_close
        self._closed = False
        self._pipelines = api_pb2_grpc.PipelineServiceStub(channel)  # type: ignore[no-untyped-call]
        self._connectors = api_pb2_grpc.ConnectorServiceStub(channel)  # type: ignore[no-untyped-call]
        self._processors = api_pb2_grpc.ProcessorServiceStub(channel)  # type: ignore[no-untyped-call]
        self._info = api_pb2_grpc.InformationServiceStub(channel)  # type: ignore[no-untyped-call]

    def __enter__(self) -> Client:
        """Support ``with conduit.connect(...) as client:``."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the channel (and, for ``local()``, tear down the subprocess)."""
        self.close()

    def close(self) -> None:
        """Close the gRPC channel and, if this client owns a subprocess, stop it.

        Idempotent -- safe to call more than once (a second call is a no-op).
        """
        if self._closed:
            return
        self._closed = True
        self._channel.close()
        if self._on_close is not None:
            self._on_close()

    @staticmethod
    def _call(stub_method: _UnaryUnary[_Req, _Resp], request: _Req, *, timeout: float) -> _Resp:
        """Invoke a unary gRPC stub method, translating failures to :class:`ConduitError`."""
        try:
            return stub_method(request, timeout=timeout)
        except grpc.RpcError as e:
            raise ConduitError.from_rpc_error(e) from e

    def get_info(self, *, timeout: float = 5.0) -> api_pb2.Info:
        """Call ``InformationService.GetInfo`` -- the running engine's version/os/arch.

        Used at connect-time by :func:`conduit.local`/:func:`conduit.connect`
        to confirm engine-version compatibility (design doc, failure mode 3).
        """
        resp: api_pb2.GetInfoResponse = self._call(
            self._info.GetInfo, api_pb2.GetInfoRequest(), timeout=timeout
        )
        return resp.info

    def run(self, pipeline: Pipeline, *, start: bool = True, timeout: float = 10.0) -> Run:
        """Submit a :class:`Pipeline` and (by default) start it.

        Drives, in order: ``CreatePipeline`` (response carries the
        server-assigned ``pipeline_id`` every subsequent call needs) ->
        ``CreateConnector`` per declared source/destination ->
        ``CreateProcessor`` per declared processor -> ``UpdateDLQ`` if
        ``.dlq(...)`` was called -> ``StartPipeline`` unless ``start=False``.

        No partial-pipeline cleanup on a mid-sequence failure: an error here
        raises immediately with whatever step failed named in the message,
        leaving already-created resources in place for the caller to inspect
        or delete -- silently rolling back would hide exactly the information
        needed to fix the config that failed.

        Args:
            pipeline: the built pipeline (see :class:`conduit.pipeline.Pipeline`).
            start: whether to call ``StartPipeline`` after creating every
                resource. ``False`` leaves the pipeline created-but-stopped
                (``STATUS_STOPPED`` / ``STOPPED_REASON_USER``).
            timeout: per-RPC timeout in seconds, applied to each call in the
                sequence independently.

        Returns:
            A :class:`conduit.run.Run` handle bound to the created pipeline's id.

        Raises:
            ConduitError: on any RPC failure, including partial-creation
                failures (see above).
        """
        plan = pipeline.build_requests()

        create_resp = self._call(
            self._pipelines.CreatePipeline, plan.create_pipeline, timeout=timeout
        )
        pipeline_id = create_resp.pipeline.id

        for connector_req in plan.create_connectors:
            connector_req.pipeline_id = pipeline_id
            self._call(self._connectors.CreateConnector, connector_req, timeout=timeout)

        for processor_req in plan.create_processors:
            processor_req.parent.type = api_pb2.Processor.Parent.TYPE_PIPELINE
            processor_req.parent.id = pipeline_id
            self._call(self._processors.CreateProcessor, processor_req, timeout=timeout)

        if plan.dlq is not None:
            plan.dlq.id = pipeline_id
            self._call(self._pipelines.UpdateDLQ, plan.dlq, timeout=timeout)

        if start:
            self._call(
                self._pipelines.StartPipeline,
                api_pb2.StartPipelineRequest(id=pipeline_id),
                timeout=timeout,
            )

        return Run(self, pipeline_id)


def connect(
    addr: str,
    *,
    credentials: grpc.ChannelCredentials | None = None,
    check_version: bool = True,
    timeout: float = 5.0,
) -> Client:
    """Dial an already-running, independently deployed Conduit engine.

    This is Mode 2 (design doc, "Deployment modes -- framed honestly"): the
    production shape. The returned :class:`Client` is a thin, stateless gRPC
    client -- the engine's lifecycle, restart policy, and persistence are
    managed independently (systemd, Kubernetes, ...), not by this process.
    ``close()``/``__exit__`` only closes the channel; no subprocess is ever
    involved.

    Args:
        addr: ``host:port`` of the engine's gRPC API (``Options.API.GRPCAddress``
            on the server side).
        credentials: gRPC channel credentials for a secure channel; ``None``
            uses an insecure channel (loopback dev default; pass real
            credentials for anything crossing an untrusted network).
        check_version: call ``GetInfo`` immediately and log-warn (via
            :class:`ConduitError` on outright unreachability) if the engine
            can't be reached at all. Does not fail on an old/new engine
            version by itself in Slice 1 -- minimum-supported-version gating
            is a fast-follow once this library has a real version to check
            against (design doc, "Upgrade / rollback").
        timeout: seconds to wait for the initial ``GetInfo`` check.

    Returns:
        A :class:`Client` bound to the given address.

    Raises:
        ConduitError: if ``check_version`` is true and the engine is
            unreachable within ``timeout``.
    """
    channel = grpc.secure_channel(addr, credentials) if credentials else grpc.insecure_channel(addr)
    client = Client(channel)
    if check_version:
        client.get_info(timeout=timeout)
    return client
