"""Problem: internalTrafficPolicy: Local on a ClusterIP service silently drops
in-cluster traffic from pods on nodes that have no local backend pod.

Real-world story
----------------
``internalTrafficPolicy: Local`` (GA in Kubernetes 1.26) was introduced to let
platform teams reduce cross-node network hops for latency-sensitive services.
When the policy is set, kube-proxy only programmes routing rules to backend pods
that are running on the *same node* as the calling pod.  If there are no local
endpoints on a node, kube-proxy drops the connection entirely — no TCP RST, no
ICMP unreachable, no HTTP error — the socket just hangs until the application
timeout fires.

The failure mode is subtle and dangerous:

* ``kubectl get pods`` shows every pod Running and Ready.
* ``kubectl get endpoints recommendation`` shows a populated endpoint list.
* ``kubectl get svc recommendation -o yaml`` shows ``internalTrafficPolicy: Local``
  — easy to miss when scanning a long YAML blob.
* Services on nodes that happen to share the backend pod work perfectly; only
  callers on other nodes silently fail.

SREGym simulation
-----------------
We target the astronomy-shop ``recommendation`` service (1 replica, ClusterIP,
port 8080).  ``frontend`` calls ``recommendation`` on the critical request path
via Kubernetes service DNS, so requests that land on a worker node without a
local ``recommendation`` pod will fail silently.

1. ``inject_fault``
   a. Select two worker nodes: *pod_node* and *victim_node*.
   b. Pin the ``recommendation`` Deployment to *pod_node* via ``nodeSelector``
      so the single replica is guaranteed to run there.
   c. Patch the ``recommendation`` Service:
      ``spec.internalTrafficPolicy: Local``.

2. ``recover_fault``
   a. Restore ``spec.internalTrafficPolicy: Cluster`` on the Service.
   b. Remove the ``nodeSelector`` from the Deployment so the pod may
      reschedule freely.

Valid agent mitigations (all accepted by the oracle)
-----------------------------------------------------
* Set ``internalTrafficPolicy`` to ``Cluster`` (or delete the field).
* Scale ``recommendation`` replicas so every worker node has at least one
  ready pod (making ``Local`` safe because no node is left without a local pod).
"""

import contextlib
import time

from kubernetes import client
from kubernetes.client.rest import ApiException

from sregym.conductor.oracles.internal_traffic_policy_mitigation import InternalTrafficPolicyMitigationOracle
from sregym.conductor.oracles.llm_as_a_judge.llm_as_a_judge_oracle import LLMAsAJudgeOracle
from sregym.conductor.problems.base import Problem
from sregym.service.apps.astronomy_shop import AstronomyShop
from sregym.service.kubectl import KubeCtl
from sregym.utils.decorators import mark_fault_injected

_CONTROL_PLANE_LABELS = frozenset(["node-role.kubernetes.io/control-plane", "node-role.kubernetes.io/master"])


class InternalTrafficPolicyLocalAstronomyShop(Problem):
    """Sets ``internalTrafficPolicy: Local`` on the astronomy-shop
    ``recommendation`` ClusterIP service and pins its single pod to one
    worker node, leaving all other worker nodes unable to reach the service
    in-cluster."""

    FAULTY_SERVICE = "recommendation"
    SERVICE_PORT = 8080
    POD_LABEL_SELECTOR = "app.kubernetes.io/component=recommendation"

    def __init__(self):
        self.app = AstronomyShop()
        super().__init__(app=self.app, namespace=self.app.namespace)
        self.kubectl = KubeCtl()
        self.core_v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.faulty_service = self.FAULTY_SERVICE

        self.pod_node: str | None = None
        self.victim_node: str | None = None

        self.root_cause = self.build_structured_root_cause(
            component=f"service/{self.FAULTY_SERVICE}",
            namespace=self.namespace,
            description=(
                f"The `{self.FAULTY_SERVICE}` Service in namespace `{self.namespace}` has "
                "`spec.internalTrafficPolicy: Local` set. "
                "This instructs kube-proxy to route in-cluster traffic **only** to pods on the "
                "**same node** as the calling pod. "
                f"The `{self.FAULTY_SERVICE}` Deployment has a single replica, so only one worker "
                "node hosts the pod. "
                "Any in-cluster caller (e.g. the `frontend` service) running on a different worker node "
                "will have its connection silently dropped by kube-proxy — the socket hangs until the "
                "application's own timeout fires, yielding no HTTP response and no TCP error. "
                f"All Kubernetes health signals appear normal: the `{self.FAULTY_SERVICE}` pod is "
                "Running and Ready, its endpoints are populated, and the Service object exists. "
                "The fault is only visible in `service.spec.internalTrafficPolicy` and the mismatch "
                "between pod placement and caller node topology. "
                "Valid mitigations: change `internalTrafficPolicy` back to `Cluster` (or remove the "
                f"field), or scale `{self.FAULTY_SERVICE}` so every worker node has at least one "
                "ready pod."
            ),
        )

        self.diagnosis_oracle = LLMAsAJudgeOracle(problem=self, expected=self.root_cause)
        self.mitigation_oracle = InternalTrafficPolicyMitigationOracle(problem=self)
        self.app.create_workload()

    # ------------------------------------------------------------------
    # Node helpers
    # ------------------------------------------------------------------

    def worker_nodes(self) -> list[str]:
        """Return sorted list of worker node names (no control-plane)."""
        return sorted(
            node.metadata.name
            for node in self.kubectl.list_nodes().items
            if not _CONTROL_PLANE_LABELS & set((node.metadata.labels or {}).keys())
        )

    def _select_nodes(self) -> tuple[str, str]:
        workers = self.worker_nodes()
        if len(workers) < 2:
            raise RuntimeError("internal_traffic_policy_local_astronomy_shop requires at least two worker nodes")
        return workers[0], workers[1]  # (pod_node, victim_node)

    def _nodes_with_running_pod(self) -> set[str]:
        """Return the set of nodes that currently have a Running recommendation pod."""
        pods = self.core_v1.list_namespaced_pod(
            self.namespace,
            label_selector=self.POD_LABEL_SELECTOR,
        )
        return {pod.spec.node_name for pod in pods.items if pod.status.phase == "Running" and pod.spec.node_name}

    # ------------------------------------------------------------------
    # Deployment / service helpers
    # ------------------------------------------------------------------

    def _pin_deployment_to_node(self, node: str) -> None:
        self.apps_v1.patch_namespaced_deployment(
            self.FAULTY_SERVICE,
            self.namespace,
            {"spec": {"template": {"spec": {"nodeSelector": {"kubernetes.io/hostname": node}}}}},
        )

    def _clear_deployment_node_selector(self) -> None:
        self.apps_v1.patch_namespaced_deployment(
            self.FAULTY_SERVICE,
            self.namespace,
            {"spec": {"template": {"spec": {"nodeSelector": {}}}}},
        )

    def _set_internal_traffic_policy(self, policy: str) -> None:
        self.kubectl.patch_service(
            self.FAULTY_SERVICE,
            self.namespace,
            {"spec": {"internalTrafficPolicy": policy}},
        )

    def _wait_for_pod_on_node(self, target_node: str, timeout: int = 180) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if target_node in self._nodes_with_running_pod():
                return
            time.sleep(4)
        raise RuntimeError(f"{self.FAULTY_SERVICE} pod did not reach {target_node} within {timeout}s")

    # ------------------------------------------------------------------
    # Fault injection / recovery
    # ------------------------------------------------------------------

    @mark_fault_injected
    def inject_fault(self):
        print("== Fault Injection ==")

        self.pod_node, self.victim_node = self._select_nodes()
        print(f"Pod node: {self.pod_node} | Victim node: {self.victim_node}")

        self._pin_deployment_to_node(self.pod_node)
        self.kubectl.exec_command(f"kubectl rollout restart deployment/{self.FAULTY_SERVICE} -n {self.namespace}")
        self._wait_for_pod_on_node(self.pod_node)
        print(f"{self.FAULTY_SERVICE} pod is Running on {self.pod_node}")

        self._set_internal_traffic_policy("Local")
        print(
            f"Patched service/{self.FAULTY_SERVICE}: internalTrafficPolicy=Local\n"
            f"Callers on {self.victim_node} will now have connections silently dropped.\n"
            f"Fault: InternalTrafficPolicyLocal | Service: {self.FAULTY_SERVICE} | "
            f"Namespace: {self.namespace}\n"
        )

    @mark_fault_injected
    def recover_fault(self):
        print("== Fault Recovery ==")

        with contextlib.suppress(ApiException, Exception):
            self._set_internal_traffic_policy("Cluster")
            print(f"Restored service/{self.FAULTY_SERVICE}: internalTrafficPolicy=Cluster")

        with contextlib.suppress(ApiException, Exception):
            self._clear_deployment_node_selector()
            self.kubectl.exec_command(f"kubectl rollout restart deployment/{self.FAULTY_SERVICE} -n {self.namespace}")
            print(f"Cleared nodeSelector on deployment/{self.FAULTY_SERVICE}")

        with contextlib.suppress(Exception):
            self.kubectl.wait_for_ready(self.namespace, max_wait=180)

        print(f"Service: {self.FAULTY_SERVICE} | Namespace: {self.namespace}\n")
