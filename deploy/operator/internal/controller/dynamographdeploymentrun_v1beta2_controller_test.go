/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrrunpublisher"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	"sigs.k8s.io/yaml"
)

type fixedRunHardware struct {
	info  gpu.GPUInfo
	err   error
	calls []nvidiacomv1beta1.GPUSKUType
}

type fixedRunSecrets struct {
	byImage map[string][]string
	err     error
}

func (f *fixedRunSecrets) GetSecrets(_ string, image string) ([]string, error) {
	if f.err != nil {
		return nil, f.err
	}
	return append([]string(nil), f.byImage[image]...), nil
}

func (f *fixedRunHardware) Discover(_ context.Context, sku nvidiacomv1beta1.GPUSKUType) (*gpu.GPUInfo, error) {
	f.calls = append(f.calls, sku)
	if f.err != nil {
		return nil, f.err
	}
	return &f.info, nil
}

type dgdrRunFixture struct {
	t          *testing.T
	client     client.Client
	reconciler *DynamoGraphDeploymentRunV1beta2Reconciler
}

func newDGDRRunFixture(t *testing.T, hardware dgdrRunHardwareDiscoverer, objects ...client.Object) *dgdrRunFixture {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(scheme); err != nil {
		t.Fatalf("add Kubernetes types: %v", err)
	}
	if err := nvidiacomv1beta2.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta2 types: %v", err)
	}
	c := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(objects...).
		WithIndex(&batchv1.Job{}, dgdrRunJobControllerUIDIndex, dgdrRunJobControllerUIDs).
		WithIndex(&nvidiacomv1beta2.DynamoGraphDeploymentRun{}, dgdrRunTraceClaimIndex, dgdrRunTraceClaimNames).
		WithStatusSubresource(&nvidiacomv1beta2.DynamoGraphDeploymentRun{}, &batchv1.Job{}, &corev1.Pod{}).
		Build()
	return &dgdrRunFixture{
		t:      t,
		client: c,
		reconciler: &DynamoGraphDeploymentRunV1beta2Reconciler{
			Client:                  c,
			APIReader:               c,
			Hardware:                hardware,
			PlannerImage:            "example.com/dynamo-planner:1.2.3",
			OperatorImage:           "example.com/dynamo-operator:1.2.3",
			OperatorImagePullPolicy: corev1.PullAlways,
		},
	}
}

func (f *dgdrRunFixture) reconcile() {
	f.t.Helper()
	if _, err := f.reconciler.Reconcile(f.t.Context(), ctrl.Request{
		Namespace: "default", Name: "search-1"}); err != nil {
		f.t.Fatalf("reconcile: %v", err)
	}
}

func (f *dgdrRunFixture) run() *nvidiacomv1beta2.DynamoGraphDeploymentRun {
	f.t.Helper()
	run := &nvidiacomv1beta2.DynamoGraphDeploymentRun{}
	if err := f.client.Get(f.t.Context(), types.NamespacedName{Namespace: "default", Name: "search-1"}, run); err != nil {
		f.t.Fatalf("get Run: %v", err)
	}
	return run
}

func (f *dgdrRunFixture) job() *batchv1.Job {
	f.t.Helper()
	jobs := &batchv1.JobList{}
	if err := f.client.List(f.t.Context(), jobs, client.InNamespace("default")); err != nil {
		f.t.Fatalf("list Jobs: %v", err)
	}
	if len(jobs.Items) != 1 {
		f.t.Fatalf("Jobs = %d, want 1", len(jobs.Items))
	}
	return &jobs.Items[0]
}

func dgdrRunForTest(t *testing.T, inputOverrides map[string]any) *nvidiacomv1beta2.DynamoGraphDeploymentRun {
	t.Helper()
	raw, err := json.Marshal(inputOverrides)
	if err != nil {
		t.Fatalf("marshal parameters: %v", err)
	}
	return &nvidiacomv1beta2.DynamoGraphDeploymentRun{
		Name: "search-1", Namespace: "default", UID: "run-uid", Generation: 1,
		Spec: nvidiacomv1beta2.DynamoGraphDeploymentRunSpec{NativeSearchInput: nvidiacomv1beta2.NativeSearchInput{
			Model: &nvidiacomv1beta2.ModelSpec{Name: "Qwen/Qwen3-0.6B"},
			Hardware: &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{
				SKU: new("h200_sxm"), Budget: ptr.To[int32](16),
			}},
			AISimulate:     &nvidiacomv1beta2.AISimulateSpec{InputOverrides: &runtime.RawExtension{Raw: raw}},
			Recommendation: &nvidiacomv1beta2.RecommendationSpec{MaxCandidates: 2},
		}},
	}
}

func TestV1beta2RunControllerCreatesOneIsolatedAISimulateJob(t *testing.T) {
	hardware := &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}
	run := dgdrRunForTest(t, map[string]any{
		"engine":       map[string]any{"hardware": "h100_sxm"},
		"optimization": map[string]any{"constraints": map[string]any{"max_candidate_gpus": 12}},
		"dgdr_run": map[string]any{
			"max_candidates": 7,
			"runtime_image":  "example.com/custom-vllm-runtime:1.2.3",
			"name":           "native-output-name",
			"snapshot_dir":   "/work/results",
			"namespace":      "another-namespace",
		},
	})
	f := newDGDRRunFixture(t, hardware, run)

	f.reconcile()
	f.reconcile()

	job := f.job()
	assertDGDRRunJobIsolation(t, job, run, f.reconciler.PlannerImage, f.reconciler.OperatorImage)
	assertDGDRRunInput(t, f.client, job, hardware)
	assertDGDRRunSupportResources(t, f.client, job, run)
	stored := f.run()
	assertDGDRRunPendingStatus(t, stored, job.Name)
	assertDGDRRunStatusInputMatchesConfigMap(t, f.client, stored, job)
}

func TestV1beta2RunControllerFreezesExecutionBeforeCreatingTheJob(t *testing.T) {
	hardware := &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}
	run := dgdrRunForTest(t, map[string]any{})
	f := newDGDRRunFixture(t, hardware, run)
	f.reconciler.DockerSecretRetriever = &fixedRunSecrets{byImage: map[string][]string{
		"example.com/dynamo-planner:1.2.3":  {"planner-secret", "shared-secret"},
		"example.com/dynamo-operator:1.2.3": {"shared-secret", "publisher-secret"},
	}}

	t.Log("Persist the immutable execution artifact before any Job exists")
	execution, err := f.reconciler.reconcileInput(t.Context(), run, nil, dgdrRunResourceName(run.Name))
	if err != nil {
		t.Fatalf("reconcile input: %v", err)
	}
	if execution.PlannerImage != "example.com/dynamo-planner:1.2.3" || execution.PublisherImage != "example.com/dynamo-operator:1.2.3" {
		t.Fatalf("frozen images = %q/%q", execution.PlannerImage, execution.PublisherImage)
	}

	t.Log("Simulate an operator upgrade before the next reconciliation")
	f.reconciler.PlannerImage = "example.com/dynamo-planner:9.9.9"
	f.reconciler.OperatorImage = "example.com/dynamo-operator:9.9.9"
	f.reconciler.OperatorImagePullPolicy = corev1.PullNever
	f.reconciler.DockerSecretRetriever = &fixedRunSecrets{byImage: map[string][]string{
		"example.com/dynamo-planner:9.9.9": {"new-secret"},
	}}
	f.reconcile()

	t.Log("Verify the Job uses only the previously frozen identity and credentials")
	job := f.job()
	planner := containerNamed(t, job.Spec.Template.Spec.Containers, dgdrRunSweeperContainer)
	publisher := containerNamed(t, job.Spec.Template.Spec.Containers, dgdrRunPublisherContainer)
	if planner.Image != "example.com/dynamo-planner:1.2.3" || publisher.Image != "example.com/dynamo-operator:1.2.3" {
		t.Fatalf("Job images = %q/%q, want frozen images", planner.Image, publisher.Image)
	}
	wantSecrets := []corev1.LocalObjectReference{{Name: "planner-secret"}, {Name: "shared-secret"}, {Name: "publisher-secret"}}
	if !reflect.DeepEqual(job.Spec.Template.Spec.ImagePullSecrets, wantSecrets) {
		t.Fatalf("imagePullSecrets = %v, want %v", job.Spec.Template.Spec.ImagePullSecrets, wantSecrets)
	}
}

func TestV1beta2RunControllerFailsClosedWithoutExecutionConfig(t *testing.T) {
	f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, dgdrRunForTest(t, map[string]any{}))
	f.reconcile()

	t.Log("Delete the immutable execution artifact while retaining the Job")
	job := f.job()
	configMap := &corev1.ConfigMap{}
	if err := f.client.Get(t.Context(), client.ObjectKeyFromObject(job), configMap); err != nil {
		t.Fatalf("get ConfigMap: %v", err)
	}
	if err := f.client.Delete(t.Context(), configMap); err != nil {
		t.Fatalf("delete ConfigMap: %v", err)
	}

	t.Log("Verify a real deletion fails closed instead of reconstructing or hot-looping")
	_, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
	if err == nil || !strings.Contains(err.Error(), "immutable execution ConfigMap") {
		t.Fatalf("reconcile error = %v, want missing immutable execution ConfigMap", err)
	}
}

func TestV1beta2RunControllerWaitsForExecutionConfigCacheLag(t *testing.T) {
	f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, dgdrRunForTest(t, map[string]any{}))
	f.reconcile()

	job := f.job()
	configMap := &corev1.ConfigMap{}
	if err := f.client.Get(t.Context(), client.ObjectKeyFromObject(job), configMap); err != nil {
		t.Fatalf("get ConfigMap: %v", err)
	}
	directReader := fake.NewClientBuilder().WithScheme(f.client.Scheme()).WithObjects(configMap.DeepCopy()).Build()
	if err := f.client.Delete(t.Context(), configMap); err != nil {
		t.Fatalf("remove ConfigMap from simulated cache: %v", err)
	}
	f.reconciler.APIReader = directReader

	t.Log("Wait only when the uncached API confirms that the immutable artifact still exists")
	result, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
	if err != nil || result.RequeueAfter != dgdrRunExecutionCacheRetry {
		t.Fatalf("reconcile result/error = %+v/%v, want bounded cache-lag retry", result, err)
	}
}

func TestV1beta2RunControllerRejectsJobExecutionSkew(t *testing.T) {
	f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, dgdrRunForTest(t, map[string]any{}))
	f.reconcile()

	t.Log("Mutate the observed Job away from the immutable execution identity")
	job := f.job()
	container := &job.Spec.Template.Spec.Containers[0]
	container.Image = "example.com/dynamo-planner:other"
	if err := f.client.Update(t.Context(), job); err != nil {
		t.Fatalf("update Job: %v", err)
	}

	t.Log("Verify reconciliation detects the inconsistency instead of accepting skew")
	_, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
	if err == nil || !strings.Contains(err.Error(), "planner image identities differ") {
		t.Fatalf("reconcile error = %v, want planner identity mismatch", err)
	}
}

func TestV1beta2RunControllerReportsImagePullSecretErrors(t *testing.T) {
	f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, dgdrRunForTest(t, map[string]any{}))
	f.reconciler.DockerSecretRetriever = &fixedRunSecrets{err: errors.New("registry index unavailable")}

	t.Log("Verify execution is not frozen without resolving its image credentials")
	_, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
	if err == nil || !strings.Contains(err.Error(), "registry index unavailable") {
		t.Fatalf("reconcile error = %v, want image pull secret error", err)
	}
}

func TestV1beta2RunControllerKeepsPlannerDigestButDerivesRuntimesFromReleaseTag(t *testing.T) {
	f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, dgdrRunForTest(t, map[string]any{}))
	f.reconciler.PlannerImage = "example.com/dynamo-planner@sha256:0123"
	f.reconciler.DynamoVersion = "1.7.0"

	t.Log("Reconcile a digest-pinned Planner through the frozen execution artifact")
	f.reconcile()

	t.Log("Keep the Planner digest while giving sibling runtime derivation a shared release tag")
	job := f.job()
	planner := containerNamed(t, job.Spec.Template.Spec.Containers, dgdrRunSweeperContainer)
	if planner.Image != "example.com/dynamo-planner@sha256:0123" {
		t.Fatalf("planner image = %q, want configured digest", planner.Image)
	}
	configMap := &corev1.ConfigMap{}
	if err := f.client.Get(t.Context(), client.ObjectKeyFromObject(job), configMap); err != nil {
		t.Fatalf("get ConfigMap: %v", err)
	}
	if got := configMap.Data[dgdrRunRuntimeImageSourceKey]; got != "example.com/dynamo-planner:1.7.0" {
		t.Fatalf("runtime image source = %q, want shared release tag", got)
	}
	var input map[string]any
	if err := yaml.Unmarshal([]byte(configMap.Data[dgdrRunInputKey]), &input); err != nil {
		t.Fatalf("decode input: %v", err)
	}
	output := input["dgdr_run"].(map[string]any)
	if output["runtime_image"] != "example.com/dynamo-planner:1.7.0" || output["runtime_version_override"] != "1.7.0" {
		t.Fatalf("dgdr_run runtime identity = %v", output)
	}
}

func assertDGDRRunJobIsolation(
	t *testing.T,
	job *batchv1.Job,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	plannerImage string,
	operatorImage string,
) {
	t.Helper()
	if !metav1.IsControlledBy(job, run) {
		t.Fatal("Job is not controlled by its Run")
	}
	if got := job.Annotations[dgdrRunGPUsPerNodeAnnotation]; got != "8" {
		t.Fatalf("GPUs-per-node annotation = %q, want 8", got)
	}
	if job.Spec.BackoffLimit == nil || *job.Spec.BackoffLimit != 0 {
		t.Fatalf("backoffLimit = %v, want 0", job.Spec.BackoffLimit)
	}
	if job.Spec.Template.Spec.AutomountServiceAccountToken == nil || *job.Spec.Template.Spec.AutomountServiceAccountToken {
		t.Fatal("the pod must disable automatic service-account token mounting")
	}
	sweeper := containerNamed(t, job.Spec.Template.Spec.Containers, dgdrRunSweeperContainer)
	publisher := containerNamed(t, job.Spec.Template.Spec.Containers, dgdrRunPublisherContainer)
	if sweeper.Image != plannerImage {
		t.Fatalf("sweeper image = %q, want operator-configured planner image %q", sweeper.Image, plannerImage)
	}
	if want := []string{"aisimulate"}; !reflect.DeepEqual(sweeper.Command, want) {
		t.Fatalf("sweeper command = %q, want %q", sweeper.Command, want)
	}
	if want := []string{
		"recommend", "--stack", "dynamo",
		"--config", dgdrRunInputPath,
		"--output", "dgdr_run",
		"--output-dir", dgdrRunSnapshotPath,
		"--overwrite",
	}; !reflect.DeepEqual(sweeper.Args, want) {
		t.Fatalf("sweeper args = %q, want %q", sweeper.Args, want)
	}
	if publisher.Image != operatorImage || publisher.ImagePullPolicy != corev1.PullAlways {
		t.Fatalf("publisher image = %q/%s, want configured operator image/Always", publisher.Image, publisher.ImagePullPolicy)
	}
	if hasMount(sweeper.VolumeMounts, dgdrRunServiceAccountVolume) {
		t.Fatal("sweeper unexpectedly receives the Kubernetes service-account token")
	}
	if !hasMount(publisher.VolumeMounts, dgdrRunServiceAccountVolume) {
		t.Fatal("publisher does not receive its projected Kubernetes credentials")
	}
	if !hasMountAt(sweeper.VolumeMounts, dgdrRunSnapshotVolume, dgdrRunSnapshotPath) ||
		!hasMountAt(publisher.VolumeMounts, dgdrRunSnapshotVolume, dgdrRunSnapshotPath) {
		t.Fatal("sweeper and publisher do not share the frozen snapshot directory")
	}
	if hasMount(publisher.VolumeMounts, dgdrRunTraceVolume) {
		t.Fatal("publisher unexpectedly mounts the trace claim")
	}
}

func assertDGDRRunStatusInputMatchesConfigMap(
	t *testing.T,
	c client.Client,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	job *batchv1.Job,
) {
	t.Helper()
	if run.Status.AISimulate == nil || run.Status.AISimulate.Input == nil {
		t.Fatal("status.aisimulate.input is empty")
	}
	var statusInput map[string]any
	if err := json.Unmarshal(run.Status.AISimulate.Input.Raw, &statusInput); err != nil {
		t.Fatalf("decode status.aisimulate.input: %v", err)
	}
	configMap := &corev1.ConfigMap{}
	if err := c.Get(t.Context(), client.ObjectKeyFromObject(job), configMap); err != nil {
		t.Fatalf("get input ConfigMap: %v", err)
	}
	var configInput map[string]any
	if err := yaml.Unmarshal([]byte(configMap.Data[dgdrRunInputKey]), &configInput); err != nil {
		t.Fatalf("decode input ConfigMap: %v", err)
	}
	if !reflect.DeepEqual(statusInput, configInput) {
		t.Fatalf("status input = %#v, want exact ConfigMap input %#v", statusInput, configInput)
	}
}

func TestDGDRRunStatusInputPreservesLargeIntegers(t *testing.T) {
	const seed = "9007199254740993"

	t.Log("Project the frozen YAML input into status without decoding numbers through float64")
	input, err := dgdrRunStatusInput("optimizer:\n  seed: " + seed + "\n")
	if err != nil {
		t.Fatalf("project status input: %v", err)
	}
	if got := string(input.Raw); !strings.Contains(got, `"seed":`+seed) {
		t.Fatalf("status input lost integer precision: %s", got)
	}
}

func assertDGDRRunInput(t *testing.T, c client.Client, job *batchv1.Job, hardware *fixedRunHardware) {
	t.Helper()
	configMap := &corev1.ConfigMap{}
	if err := c.Get(t.Context(), client.ObjectKeyFromObject(job), configMap); err != nil {
		t.Fatalf("get input ConfigMap: %v", err)
	}
	if configMap.Immutable == nil || !*configMap.Immutable {
		t.Fatal("input ConfigMap is not immutable")
	}
	for key, want := range map[string]string{
		dgdrRunPlannerImageKey:        "example.com/dynamo-planner:1.2.3",
		dgdrRunPlannerPullPolicyKey:   string(corev1.PullIfNotPresent),
		dgdrRunRuntimeImageSourceKey:  "example.com/dynamo-planner:1.2.3",
		dgdrRunPublisherImageKey:      "example.com/dynamo-operator:1.2.3",
		dgdrRunPublisherPullPolicyKey: string(corev1.PullAlways),
		dgdrRunDynamoVersionKey:       "1.2.3",
		dgdrRunGPUsPerNodeKey:         "8",
		dgdrRunSnapshotPathKey:        dgdrRunSnapshotPath,
	} {
		if got := configMap.Data[key]; got != want {
			t.Fatalf("ConfigMap data[%q] = %q, want %q", key, got, want)
		}
	}
	var input map[string]any
	if err := yaml.Unmarshal([]byte(configMap.Data[dgdrRunInputKey]), &input); err != nil {
		t.Fatalf("decode input ConfigMap: %v", err)
	}
	output := input["dgdr_run"].(map[string]any)
	if output["name"] != "search-1" || output["max_candidates"] != float64(2) {
		t.Fatalf("operator-owned identity and candidate cap were not restored: %v", output)
	}
	if output["snapshot_dir"] != dgdrRunSnapshotPath || output["namespace"] != "default" {
		t.Fatalf("operator-owned transport overrides were not restored: %v", output)
	}
	if output["runtime_image"] != "example.com/dynamo-planner:1.2.3" || output["runtime_version_override"] != "1.2.3" {
		t.Fatalf("operator-owned runtime identity was not restored: %v", output)
	}
	if output["derive_runtime_images"] != true {
		t.Fatalf("derive_runtime_images = %v, want true for a backend search", output["derive_runtime_images"])
	}
	if got := input["engine"].(map[string]any)["hardware"]; got != "h100_sxm" {
		t.Fatalf("engine.hardware = %v, want authoritative input override h100_sxm", got)
	}
	constraints := input["optimization"].(map[string]any)["constraints"].(map[string]any)
	if got := constraints["max_candidate_gpus"]; got != float64(12) {
		t.Fatalf("optimization.constraints.max_candidate_gpus = %v, want authoritative input override 12", got)
	}
	if !reflect.DeepEqual(hardware.calls, []nvidiacomv1beta1.GPUSKUType{"h200_sxm"}) {
		t.Fatalf("hardware discovery calls = %v, want the admitted typed SKU h200_sxm", hardware.calls)
	}
}

func TestDGDRRunInputResolvesRuntimeVersion(t *testing.T) {
	tests := []struct {
		name          string
		plannerImage  string
		dynamoVersion string
		wantOverride  string
		wantRuntime   string
		wantErr       bool
	}{
		{name: "canonical planner tag", plannerImage: "example.com/dynamo-planner:1.7.0", wantOverride: "1.7.0", wantRuntime: "example.com/dynamo-planner:1.7.0"},
		{name: "v-prefixed planner tag", plannerImage: "example.com/dynamo-planner:v1.7.0", wantOverride: "1.7.0", wantRuntime: "example.com/dynamo-planner:v1.7.0"},
		{name: "prerelease planner tag", plannerImage: "example.com/dynamo-planner:1.7.0-rc.1", wantOverride: "1.7.0", wantRuntime: "example.com/dynamo-planner:1.7.0-rc.1"},
		{name: "build metadata planner tag", plannerImage: "example.com/dynamo-planner:1.7.0+build.5", wantOverride: "1.7.0", wantRuntime: "example.com/dynamo-planner:1.7.0+build.5"},
		{name: "tag and digest", plannerImage: "example.com/dynamo-planner:v1.7.0-rc.1@sha256:0123", wantOverride: "1.7.0", wantRuntime: "example.com/dynamo-planner:v1.7.0-rc.1@sha256:0123"},
		{name: "digest only", plannerImage: "example.com/dynamo-planner@sha256:0123", dynamoVersion: "1.7.0", wantOverride: "1.7.0", wantRuntime: "example.com/dynamo-planner:1.7.0"},
		{name: "opaque CI tag", plannerImage: "example.com/dynamo-planner:git-sha", dynamoVersion: "1.7.0", wantOverride: "1.7.0", wantRuntime: "example.com/dynamo-planner:git-sha"},
		{name: "untagged planner image", plannerImage: "example.com/dynamo-planner", dynamoVersion: "1.7.0", wantErr: true},
		{name: "opaque tag without canonical version", plannerImage: "example.com/dynamo-planner:git-sha", dynamoVersion: "1.7", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			version, err := dgdrRunDynamoVersion(tt.plannerImage, tt.dynamoVersion)
			var runtimeImage string
			if err == nil {
				runtimeImage, err = dgdrRunRuntimeImageSource(tt.plannerImage, version)
			}
			if (err != nil) != tt.wantErr {
				t.Fatalf("resolve execution identity error = %v, wantErr %t", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			input, err := dgdrRunInput(dgdrRunForTest(t, map[string]any{}), runtimeImage, version, 8)
			if err != nil {
				t.Fatalf("lower input: %v", err)
			}
			output := input["dgdr_run"].(map[string]any)
			if got := output["runtime_version_override"]; got != tt.wantOverride {
				t.Fatalf("dgdr_run.runtime_version_override = %v, want %s", got, tt.wantOverride)
			}
			if got := output["runtime_image"]; got != tt.wantRuntime {
				t.Fatalf("dgdr_run.runtime_image = %v, want %s", got, tt.wantRuntime)
			}
		})
	}
}

func assertDGDRRunSupportResources(
	t *testing.T,
	c client.Client,
	job *batchv1.Job,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
) {
	t.Helper()
	for _, object := range []client.Object{
		&corev1.ServiceAccount{}, &rbacv1.Role{}, &rbacv1.RoleBinding{},
	} {
		if err := c.Get(t.Context(), client.ObjectKeyFromObject(job), object); err != nil {
			t.Fatalf("get %T: %v", object, err)
		}
		if !metav1.IsControlledBy(object, run) {
			t.Fatalf("%T is not controlled by the Run", object)
		}
	}
}

func assertDGDRRunPendingStatus(t *testing.T, stored *nvidiacomv1beta2.DynamoGraphDeploymentRun, jobName string) {
	t.Helper()
	if stored.Status.Phase != nvidiacomv1beta2.DGDRRunPhasePending || stored.Status.JobRef == nil || stored.Status.JobRef.Name != jobName {
		t.Fatalf("Run status = %+v, want Pending and JobRef %q", stored.Status, jobName)
	}
	condition := meta.FindStatusCondition(stored.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	if condition == nil || condition.Status != metav1.ConditionFalse || condition.Reason != dgdrRunReasonPending {
		t.Fatalf("Completed condition = %+v, want False/%s", condition, dgdrRunReasonPending)
	}
}

func assertDGDRRunRunningStatus(t *testing.T, stored *nvidiacomv1beta2.DynamoGraphDeploymentRun, jobName string) {
	t.Helper()
	if stored.Status.Phase != nvidiacomv1beta2.DGDRRunPhaseRunning || stored.Status.JobRef == nil || stored.Status.JobRef.Name != jobName {
		t.Fatalf("Run status = %+v, want Running and JobRef %q", stored.Status, jobName)
	}
	condition := meta.FindStatusCondition(stored.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	if condition == nil || condition.Status != metav1.ConditionFalse || condition.Reason != dgdrRunReasonRunning {
		t.Fatalf("Completed condition = %+v, want False/%s", condition, dgdrRunReasonRunning)
	}
}

func TestV1beta2RunControllerOwnsOnlyLifecycleStatus(t *testing.T) {
	hardware := &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}
	f := newDGDRRunFixture(t, hardware, dgdrRunForTest(t, map[string]any{}))
	f.reconcile()
	job := f.job()

	run := f.run()
	progressTime := metav1.NewTime(time.Date(2026, 10, 23, 10, 0, 0, 0, time.UTC))
	run.Status.Progress = &nvidiacomv1beta2.DGDRRunProgress{Candidates: &nvidiacomv1beta2.CandidateProgress{Published: 1}}
	run.Status.CandidateRefs = []corev1.LocalObjectReference{{Name: "candidate-a"}}
	run.Status.LastProgressTime = &progressTime
	if err := f.client.Status().Update(t.Context(), run); err != nil {
		t.Fatalf("publish progress status: %v", err)
	}

	start := metav1.NewTime(progressTime.Add(time.Minute))
	job.Status.StartTime = &start
	job.Status.Active = 1
	if err := f.client.Status().Update(t.Context(), job); err != nil {
		t.Fatalf("mark Job running: %v", err)
	}
	f.reconcile()
	run = f.run()
	if run.Status.Phase != nvidiacomv1beta2.DGDRRunPhaseRunning || run.Status.StartTime == nil || !run.Status.StartTime.Equal(&start) {
		t.Fatalf("running status = %+v", run.Status)
	}
	if run.Status.Progress == nil || len(run.Status.CandidateRefs) != 1 || run.Status.LastProgressTime == nil {
		t.Fatalf("publisher-owned status was lost: %+v", run.Status)
	}

	completed := metav1.NewTime(start.Add(5 * time.Minute))
	job = f.job()
	job.Status.Active = 0
	job.Status.CompletionTime = &completed
	job.Status.Conditions = []batchv1.JobCondition{{
		Type: batchv1.JobComplete, Status: corev1.ConditionTrue, LastTransitionTime: completed,
	}}
	if err := f.client.Status().Update(t.Context(), job); err != nil {
		t.Fatalf("complete Job: %v", err)
	}
	f.reconcile()
	run = f.run()
	condition := meta.FindStatusCondition(run.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	if run.Status.Phase != nvidiacomv1beta2.DGDRRunPhaseSucceeded || condition == nil || condition.Status != metav1.ConditionTrue {
		t.Fatalf("completed status = %+v", run.Status)
	}
	if run.Status.CompletionTime == nil || !run.Status.CompletionTime.Equal(&completed) {
		t.Fatalf("completionTime = %v, want %v", run.Status.CompletionTime, completed)
	}
}

func TestV1beta2RunControllerClassifiesPublisherProtocolFailure(t *testing.T) {
	f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, dgdrRunForTest(t, map[string]any{}))
	f.reconcile()
	job := f.job()
	pod := &corev1.Pod{
		Name:      job.Name + "-pod",
		Namespace: job.Namespace,
		Labels:    map[string]string{batchv1.JobNameLabel: job.Name},
		OwnerReferences: []metav1.OwnerReference{{
			APIVersion: batchv1.SchemeGroupVersion.String(), Kind: "Job", Name: job.Name, UID: job.UID,
			Controller: new(true),
		}},
	}
	if err := f.client.Create(t.Context(), pod); err != nil {
		t.Fatalf("create Job pod: %v", err)
	}
	pod.Status.ContainerStatuses = []corev1.ContainerStatus{{
		Name: dgdrRunPublisherContainer,
		State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{
			ExitCode: int32(dgdrrunpublisher.ExitProtocolViolation), Message: "terminal snapshot missing",
		}},
	}}
	if err := f.client.Status().Update(t.Context(), pod); err != nil {
		t.Fatalf("update Job pod status: %v", err)
	}

	failed := metav1.Now()
	job.Status.Conditions = []batchv1.JobCondition{{
		Type: batchv1.JobFailed, Status: corev1.ConditionTrue, LastTransitionTime: failed,
	}}
	if err := f.client.Status().Update(t.Context(), job); err != nil {
		t.Fatalf("fail Job: %v", err)
	}
	f.reconcile()

	run := f.run()
	condition := meta.FindStatusCondition(run.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	if run.Status.Phase != nvidiacomv1beta2.DGDRRunPhaseFailed || condition == nil || condition.Reason != dgdrRunReasonProtocolViolation {
		t.Fatalf("failed status = %+v", run.Status)
	}
	if !strings.Contains(condition.Message, "terminal snapshot missing") {
		t.Fatalf("failure message = %q, want publisher termination message", condition.Message)
	}
}

func TestDGDRRunResourceNameIsStableAndBounded(t *testing.T) {
	short := dgdrRunResourceName("search-1")
	if short != "search-1-aisimulate" {
		t.Fatalf("short name = %q", short)
	}
	long := strings.Repeat("a", 200)
	one, two := dgdrRunResourceName(long), dgdrRunResourceName(long)
	if one != two || len(one) > dgdrRunMaxJobNameLength || !strings.HasSuffix(one, "-aisimulate") {
		t.Fatalf("bounded name = %q", one)
	}
}

func TestRequireDGDRV1beta2RESTMappingsIncludesCandidates(t *testing.T) {
	groupVersion := nvidiacomv1beta2.GroupVersion
	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{groupVersion})
	mapper.Add(groupVersion.WithKind("DynamoGraphDeploymentRequest"), meta.RESTScopeNamespace)
	mapper.Add(groupVersion.WithKind("DynamoGraphDeploymentRun"), meta.RESTScopeNamespace)

	t.Log("Verify startup fails when Candidate publication cannot succeed")
	err := requireDGDRV1beta2RESTMappings(mapper)
	if err == nil || !strings.Contains(err.Error(), "DynamoGraphDeploymentCandidate") {
		t.Fatalf("preflight error = %v, want missing Candidate API", err)
	}

	t.Log("Register the complete execution API and verify preflight succeeds")
	mapper.Add(groupVersion.WithKind("DynamoGraphDeploymentCandidate"), meta.RESTScopeNamespace)
	if err := requireDGDRV1beta2RESTMappings(mapper); err != nil {
		t.Fatalf("complete API preflight: %v", err)
	}
}

func containerNamed(t *testing.T, containers []corev1.Container, name string) corev1.Container {
	t.Helper()
	for _, container := range containers {
		if container.Name == name {
			return container
		}
	}
	t.Fatalf("container %q not found", name)
	return corev1.Container{}
}

func hasMount(mounts []corev1.VolumeMount, name string) bool {
	for _, mount := range mounts {
		if mount.Name == name {
			return true
		}
	}
	return false
}

func hasMountAt(mounts []corev1.VolumeMount, name, path string) bool {
	for _, mount := range mounts {
		if mount.Name == name && mount.MountPath == path {
			return true
		}
	}
	return false
}

func dgdrRunWithTraceForTest(t *testing.T, inputOverrides map[string]any) *nvidiacomv1beta2.DynamoGraphDeploymentRun {
	t.Helper()
	run := dgdrRunForTest(t, inputOverrides)
	run.Spec.Workload = &nvidiacomv1beta2.WorkloadSpec{Trace: &nvidiacomv1beta2.TraceWorkloadSpec{
		Format: nvidiacomv1beta2.TraceFormatDynamo,
		Source: nvidiacomv1beta2.TraceSourceSpec{PVC: &nvidiacomv1beta2.TracePVCSource{
			ClaimName: "request-traces", Path: "retune/w-1.jsonl.gz",
		}},
	}}
	return run
}

func traceClaimForTest() *corev1.PersistentVolumeClaim {
	return &corev1.PersistentVolumeClaim{
		Name: "request-traces", Namespace: "default",
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteMany},
			VolumeMode:  ptr.To(corev1.PersistentVolumeFilesystem),
		},
		Status: corev1.PersistentVolumeClaimStatus{Phase: corev1.ClaimBound},
	}
}

func TestV1beta2RunControllerMountsTheTraceFileReadOnlyIntoTheSweeperOnly(t *testing.T) {
	run := dgdrRunWithTraceForTest(t, map[string]any{})
	f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, run, traceClaimForTest())
	f.reconcile()

	t.Log("The pod mounts the claim read-only as the trace volume")
	job := f.job()
	var trace *corev1.Volume
	for i := range job.Spec.Template.Spec.Volumes {
		if job.Spec.Template.Spec.Volumes[i].Name == dgdrRunTraceVolume {
			trace = &job.Spec.Template.Spec.Volumes[i]
		}
	}
	if trace == nil || trace.PersistentVolumeClaim == nil ||
		trace.PersistentVolumeClaim.ClaimName != "request-traces" || !trace.PersistentVolumeClaim.ReadOnly {
		t.Fatalf("trace volume = %#v, want the read-only request-traces claim", trace)
	}

	t.Log("Only the sweeper mounts the one trace file, read-only through a subPath at the fixed path")
	sweeper := containerNamed(t, job.Spec.Template.Spec.Containers, dgdrRunSweeperContainer)
	want := corev1.VolumeMount{
		Name: dgdrRunTraceVolume, MountPath: "/var/run/aisimulate/trace/trace.jsonl.gz",
		SubPath: "retune/w-1.jsonl.gz", ReadOnly: true,
	}
	var traceMounts []corev1.VolumeMount
	for _, mount := range sweeper.VolumeMounts {
		if mount.Name == dgdrRunTraceVolume {
			traceMounts = append(traceMounts, mount)
		}
	}
	if !reflect.DeepEqual(traceMounts, []corev1.VolumeMount{want}) {
		t.Fatalf("sweeper trace mounts = %#v, want exactly %#v", traceMounts, want)
	}
	assertDGDRRunJobIsolation(t, job, run, f.reconciler.PlannerImage, f.reconciler.OperatorImage)

	t.Log("The pod grants the trace group as a supplemental group and requests no fsGroup")
	wantSecurity := &corev1.PodSecurityContext{SupplementalGroups: []int64{65532}}
	if got := job.Spec.Template.Spec.SecurityContext; !reflect.DeepEqual(got, wantSecurity) {
		t.Fatalf("pod securityContext = %#v, want %#v", got, wantSecurity)
	}

	t.Log("The frozen input reads the mounted trace file")
	configMap := &corev1.ConfigMap{}
	if err := f.client.Get(t.Context(), client.ObjectKeyFromObject(job), configMap); err != nil {
		t.Fatalf("get ConfigMap: %v", err)
	}
	var input map[string]any
	if err := yaml.Unmarshal([]byte(configMap.Data[dgdrRunInputKey]), &input); err != nil {
		t.Fatalf("decode input: %v", err)
	}
	source := input["traffic"].(map[string]any)["source"].(map[string]any)
	if !reflect.DeepEqual(source["paths"], []any{"/var/run/aisimulate/trace/trace.jsonl.gz"}) {
		t.Fatalf("traffic.source.paths = %v", source["paths"])
	}
}

func TestV1beta2RunControllerSetsNoPodSecurityContextWithoutATrace(t *testing.T) {
	f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, dgdrRunForTest(t, map[string]any{}))
	f.reconcile()

	t.Log("A Run without a trace requests neither an fsGroup nor a supplemental group")
	if got := f.job().Spec.Template.Spec.SecurityContext; got != nil {
		t.Fatalf("pod securityContext = %#v, want none", got)
	}
}

func TestDGDRRunInputEnforcesTheTypedTraceSource(t *testing.T) {
	t.Log("Lower a Run whose overrides try to replace the trace source and change the load")
	run := dgdrRunWithTraceForTest(t, map[string]any{"traffic": map[string]any{
		"source": map[string]any{"type": "synthetic", "format": "mooncake", "paths": []any{"/etc/passwd"}},
		"load":   map[string]any{"speedup": 4},
	}})
	input, err := dgdrRunInput(run, "example.com/dynamo-planner:1.2.3", "1.2.3", 8)
	if err != nil {
		t.Fatalf("lower input: %v", err)
	}

	t.Log("The typed trace wins for the source; the load override is kept")
	traffic := input["traffic"].(map[string]any)
	wantSource := map[string]any{"type": "trace", "format": "dynamo", "paths": []any{"/var/run/aisimulate/trace/trace.jsonl.gz"}}
	if !reflect.DeepEqual(traffic["source"], wantSource) {
		t.Fatalf("traffic.source = %#v, want %#v", traffic["source"], wantSource)
	}
	load := traffic["load"].(map[string]any)
	if load["type"] != "trace_timestamps" || load["speedup"] != json.Number("4") {
		t.Fatalf("traffic.load = %#v, want trace_timestamps with the override speedup", load)
	}
	if _, found := traffic["stop"]; found {
		t.Fatalf("traffic.stop = %#v, want none for a trace", traffic["stop"])
	}
}

func TestDGDRRunInputRebuildsANonObjectTraceSourceParent(t *testing.T) {
	wantSource := map[string]any{"type": "trace", "format": "dynamo", "paths": []any{"/var/run/aisimulate/trace/trace.jsonl.gz"}}
	tests := []struct {
		name      string
		overrides map[string]any
		wantLoad  map[string]any
	}{
		{
			name:      "scalar traffic",
			overrides: map[string]any{"traffic": "invalid"},
		},
		{
			name:      "scalar traffic.source",
			overrides: map[string]any{"traffic": map[string]any{"source": "invalid"}},
			wantLoad:  map[string]any{"type": "trace_timestamps"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Lower a Run created without admission whose override replaces a trace source parent")
			input, err := dgdrRunInput(dgdrRunWithTraceForTest(t, tt.overrides), "example.com/dynamo-planner:1.2.3", "1.2.3", 8)
			if err != nil {
				t.Fatalf("lower input: %v", err)
			}

			t.Log("The parent is an object again and carries the typed trace source")
			traffic := input["traffic"].(map[string]any)
			if !reflect.DeepEqual(traffic["source"], wantSource) {
				t.Fatalf("traffic.source = %#v, want %#v", traffic["source"], wantSource)
			}
			if load, _ := traffic["load"].(map[string]any); !reflect.DeepEqual(load, tt.wantLoad) {
				t.Fatalf("traffic.load = %#v, want %#v", traffic["load"], tt.wantLoad)
			}
		})
	}
}

func TestV1beta2RunControllerFailsARunWhoseTraceClaimCannotBeMounted(t *testing.T) {
	tests := []struct {
		name    string
		missing bool
		mutate  func(*corev1.PersistentVolumeClaim)
		reason  string
	}{
		{
			name:    "missing claim",
			missing: true,
			reason:  `trace PersistentVolumeClaim "request-traces" does not exist in namespace "default"`,
		},
		{
			name: "claim being deleted",
			mutate: func(claim *corev1.PersistentVolumeClaim) {
				claim.DeletionTimestamp = new(metav1.Now())
				claim.Finalizers = []string{"kubernetes.io/pvc-protection"}
			},
			reason: `trace PersistentVolumeClaim "request-traces" in namespace "default" is being deleted`,
		},
		{
			name: "raw block claim",
			mutate: func(claim *corev1.PersistentVolumeClaim) {
				claim.Spec.VolumeMode = ptr.To(corev1.PersistentVolumeBlock)
			},
			reason: `trace PersistentVolumeClaim "request-traces" in namespace "default" is a raw block volume`,
		},
		{
			name: "lost claim",
			mutate: func(claim *corev1.PersistentVolumeClaim) {
				claim.Status.Phase = corev1.ClaimLost
			},
			reason: `trace PersistentVolumeClaim "request-traces" in namespace "default" has lost its PersistentVolume`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hardware := &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}
			objects := []client.Object{dgdrRunWithTraceForTest(t, map[string]any{})}
			if !tt.missing {
				claim := traceClaimForTest()
				tt.mutate(claim)
				objects = append(objects, claim)
			}
			f := newDGDRRunFixture(t, hardware, objects...)

			t.Log("Reconcile twice: the first pass fails the Run, the second sees it completed")
			f.reconcile()
			f.reconcile()

			t.Log("The Run is Failed and Completed with the claim named in the message")
			run := f.run()
			condition := meta.FindStatusCondition(run.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
			if run.Status.Phase != nvidiacomv1beta2.DGDRRunPhaseFailed || condition == nil ||
				condition.Status != metav1.ConditionTrue || condition.Reason != dgdrRunReasonTraceSourceUnavailable {
				t.Fatalf("status = %+v, want Failed and Completed=True/%s", run.Status, dgdrRunReasonTraceSourceUnavailable)
			}
			if !strings.HasPrefix(condition.Message, tt.reason) || run.Status.CompletionTime == nil || run.Status.JobRef != nil {
				t.Fatalf("condition message = %q, completionTime = %v, jobRef = %v; want message prefix %q",
					condition.Message, run.Status.CompletionTime, run.Status.JobRef, tt.reason)
			}
			if condition.ObservedGeneration != run.Generation {
				t.Fatalf("condition observedGeneration = %d, want %d", condition.ObservedGeneration, run.Generation)
			}

			t.Log("Nothing was discovered or created for the Run")
			if len(hardware.calls) != 0 {
				t.Fatalf("hardware discovery calls = %v, want none", hardware.calls)
			}
			assertNoDGDRRunResources(t, f.client)
		})
	}
}

func assertNoDGDRRunResources(t *testing.T, c client.Client) {
	t.Helper()
	for _, list := range []client.ObjectList{
		&batchv1.JobList{}, &corev1.ConfigMapList{}, &corev1.ServiceAccountList{},
		&rbacv1.RoleList{}, &rbacv1.RoleBindingList{},
	} {
		if err := c.List(t.Context(), list, client.InNamespace("default")); err != nil {
			t.Fatalf("list %T: %v", list, err)
		}
		if n := meta.LenList(list); n != 0 {
			t.Fatalf("%T has %d items, want none", list, n)
		}
	}
}

func TestV1beta2RunControllerAcceptsAnUnboundOrSharedTraceClaim(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*corev1.PersistentVolumeClaim)
	}{
		{name: "WaitForFirstConsumer claim still Pending", mutate: func(claim *corev1.PersistentVolumeClaim) {
			claim.Status.Phase = corev1.ClaimPending
		}},
		{name: "ReadWriteOnce claim", mutate: func(claim *corev1.PersistentVolumeClaim) {
			claim.Spec.AccessModes = []corev1.PersistentVolumeAccessMode{corev1.ReadWriteOnce}
		}},
		{name: "ReadWriteOncePod claim", mutate: func(claim *corev1.PersistentVolumeClaim) {
			claim.Spec.AccessModes = []corev1.PersistentVolumeAccessMode{corev1.ReadWriteOncePod}
		}},
		{name: "claim without a volume mode", mutate: func(claim *corev1.PersistentVolumeClaim) {
			claim.Spec.VolumeMode = nil
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			claim := traceClaimForTest()
			tt.mutate(claim)
			f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, dgdrRunWithTraceForTest(t, map[string]any{}), claim)
			f.reconcile()
			assertDGDRRunPendingStatus(t, f.run(), f.job().Name)
		})
	}
}

func TestV1beta2RunControllerRetriesATransientTraceClaimReadError(t *testing.T) {
	hardware := &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}
	f := newDGDRRunFixture(t, hardware, dgdrRunWithTraceForTest(t, map[string]any{}), traceClaimForTest())
	f.reconciler.APIReader = interceptor.NewClient(f.client.(client.WithWatch), interceptor.Funcs{
		Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
			if _, ok := obj.(*corev1.PersistentVolumeClaim); ok {
				return errors.New("API server unavailable")
			}
			return c.Get(ctx, key, obj, opts...)
		},
	})

	t.Log("A read error other than NotFound is returned for a retry")
	_, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
	if err == nil || !strings.Contains(err.Error(), "API server unavailable") {
		t.Fatalf("reconcile error = %v, want the claim read error", err)
	}

	t.Log("The Run stays open, and nothing was discovered or created for it")
	if run := f.run(); run.Status.Phase != "" || len(run.Status.Conditions) != 0 {
		t.Fatalf("status = %+v, want untouched", run.Status)
	}
	if len(hardware.calls) != 0 {
		t.Fatalf("hardware discovery calls = %v, want none", hardware.calls)
	}
	assertNoDGDRRunResources(t, f.client)
}

func TestV1beta2RunControllerKeepsFollowingAStartedJobAfterItsTraceClaimDisappears(t *testing.T) {
	tests := []struct {
		name      string
		status    batchv1.JobStatus
		podNode   string
		podPhase  corev1.PodPhase
		wantPhase nvidiacomv1beta2.DGDRRunPhase
	}{
		{
			name:      "pod scheduled while its images are pulled",
			status:    batchv1.JobStatus{Active: 1, Ready: new(int32(0))},
			podNode:   "node-a",
			podPhase:  corev1.PodPending,
			wantPhase: nvidiacomv1beta2.DGDRRunPhaseRunning,
		},
		{
			name:      "a container exited after the pod was ready",
			status:    batchv1.JobStatus{Active: 1, Ready: new(int32(0))},
			podNode:   "node-a",
			podPhase:  corev1.PodRunning,
			wantPhase: nvidiacomv1beta2.DGDRRunPhaseRunning,
		},
		{
			name:      "pod ready while the pod cache still shows it unscheduled",
			status:    batchv1.JobStatus{Active: 1, Ready: new(int32(1))},
			podPhase:  corev1.PodPending,
			wantPhase: nvidiacomv1beta2.DGDRRunPhaseRunning,
		},
		{
			name:      "Job tracks a succeeded pod it has not counted",
			status:    batchv1.JobStatus{UncountedTerminatedPods: &batchv1.UncountedTerminatedPods{Succeeded: []types.UID{"pod"}}},
			wantPhase: nvidiacomv1beta2.DGDRRunPhaseRunning,
		},
		{
			name:      "Job tracks a failed pod it has not counted",
			status:    batchv1.JobStatus{UncountedTerminatedPods: &batchv1.UncountedTerminatedPods{Failed: []types.UID{"pod"}}},
			wantPhase: nvidiacomv1beta2.DGDRRunPhaseRunning,
		},
		{
			name:      "Job met its success criteria",
			status:    batchv1.JobStatus{Conditions: []batchv1.JobCondition{{Type: batchv1.JobSuccessCriteriaMet, Status: corev1.ConditionTrue}}},
			wantPhase: nvidiacomv1beta2.DGDRRunPhaseRunning,
		},
		{
			name:      "Job reached its failure target",
			status:    batchv1.JobStatus{Conditions: []batchv1.JobCondition{{Type: batchv1.JobFailureTarget, Status: corev1.ConditionTrue}}},
			wantPhase: nvidiacomv1beta2.DGDRRunPhaseRunning,
		},
		{
			name:      "Job counted a succeeded pod",
			status:    batchv1.JobStatus{Succeeded: 1},
			wantPhase: nvidiacomv1beta2.DGDRRunPhaseRunning,
		},
		{
			name:      "Job counted a failed pod",
			status:    batchv1.JobStatus{Failed: 1},
			wantPhase: nvidiacomv1beta2.DGDRRunPhaseRunning,
		},
		{
			name:      "Job complete",
			status:    batchv1.JobStatus{Conditions: []batchv1.JobCondition{{Type: batchv1.JobComplete, Status: corev1.ConditionTrue}}},
			wantPhase: nvidiacomv1beta2.DGDRRunPhaseSucceeded,
		},
		{
			name:      "Job failed",
			status:    batchv1.JobStatus{Conditions: []batchv1.JobCondition{{Type: batchv1.JobFailed, Status: corev1.ConditionTrue}}},
			wantPhase: nvidiacomv1beta2.DGDRRunPhaseFailed,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			claim := traceClaimForTest()
			claim.Finalizers = []string{"kubernetes.io/pvc-protection"}
			f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, dgdrRunWithTraceForTest(t, map[string]any{}), claim)
			f.reconcile()

			t.Log("The Job, and its pod where the row has one, report the pod past the claim")
			job := f.job()
			job.Status = tt.status
			job.Status.StartTime = new(metav1.Now())
			if err := f.client.Status().Update(t.Context(), job); err != nil {
				t.Fatalf("update Job status: %v", err)
			}
			if tt.podPhase != "" {
				createDGDRRunJobPodForTest(t, f.client, job, tt.podNode, corev1.PodStatus{Phase: tt.podPhase})
			}

			t.Log("Delete the claim")
			if err := f.client.Delete(t.Context(), traceClaimForTest()); err != nil {
				t.Fatalf("delete claim: %v", err)
			}

			t.Log("Hide the Job from the cached List, as a lagging informer would")
			cached := f.reconciler.Client
			f.reconciler.Client = interceptor.NewClient(f.client.(client.WithWatch), interceptor.Funcs{
				List: func(ctx context.Context, c client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
					if _, ok := list.(*batchv1.JobList); ok {
						return nil
					}
					return c.List(ctx, list, opts...)
				},
			})
			_, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
			if !apierrors.IsAlreadyExists(err) {
				t.Fatalf("reconcile error = %v, want the Job create to hit AlreadyExists until the cache catches up", err)
			}

			t.Log("Once the cache shows the Job again, the Job, not the claim, decides the Run")
			f.reconciler.Client = cached
			f.reconcile()
			if kept := f.job(); kept.UID != job.UID {
				t.Fatalf("Job UID = %q, want the started Job %q kept", kept.UID, job.UID)
			}
			run := f.run()
			if run.Status.Phase != tt.wantPhase {
				t.Fatalf("phase = %q, want %q", run.Status.Phase, tt.wantPhase)
			}
			condition := meta.FindStatusCondition(run.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
			if condition == nil || condition.Reason == dgdrRunReasonTraceSourceUnavailable {
				t.Fatalf("Completed condition = %+v, want the Job, not the claim, to decide it", condition)
			}
		})
	}
}

// createDGDRRunJobPodForTest stores the pod the Job controller would create for job, bound to node
// unless node is empty, with status, and returns it.
func createDGDRRunJobPodForTest(t *testing.T, c client.Client, job *batchv1.Job, node string, status corev1.PodStatus) *corev1.Pod {
	t.Helper()
	pod := &corev1.Pod{
		Name: job.Name + "-pod", Namespace: job.Namespace,
		Labels:          map[string]string{batchv1.JobNameLabel: job.Name},
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(job, batchv1.SchemeGroupVersion.WithKind("Job"))},
		Spec:            corev1.PodSpec{NodeName: node},
	}
	if err := c.Create(t.Context(), pod); err != nil {
		t.Fatalf("create Job pod: %v", err)
	}
	pod.Status = status
	if err := c.Status().Update(t.Context(), pod); err != nil {
		t.Fatalf("update Job pod status: %v", err)
	}
	return pod
}

// assertDGDRRunFailedForTrace checks that the Run failed for its trace with a message that starts
// with reason and names the deleted Job, and that neither the Job nor its pods remain.
func assertDGDRRunFailedForTrace(t *testing.T, c client.Client, run *nvidiacomv1beta2.DynamoGraphDeploymentRun, jobName, reason string) {
	t.Helper()
	jobs := &batchv1.JobList{}
	pods := &corev1.PodList{}
	for _, list := range []client.ObjectList{jobs, pods} {
		if err := c.List(t.Context(), list, client.InNamespace("default")); err != nil {
			t.Fatalf("list %T: %v", list, err)
		}
	}
	if len(jobs.Items) != 0 || len(pods.Items) != 0 {
		t.Fatalf("Jobs = %d, pods = %d, want the unstarted Job and its pod deleted", len(jobs.Items), len(pods.Items))
	}
	condition := meta.FindStatusCondition(run.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	if run.Status.Phase != nvidiacomv1beta2.DGDRRunPhaseFailed || condition == nil ||
		condition.Status != metav1.ConditionTrue || condition.Reason != dgdrRunReasonTraceSourceUnavailable {
		t.Fatalf("status = %+v, want Failed and Completed=True/%s", run.Status, dgdrRunReasonTraceSourceUnavailable)
	}
	wantSuffix := fmt.Sprintf(`; deleted Job %q before its pod started`, jobName)
	if !strings.HasPrefix(condition.Message, reason) || !strings.HasSuffix(condition.Message, wantSuffix) {
		t.Fatalf("condition message = %q, want prefix %q and suffix %q", condition.Message, reason, wantSuffix)
	}
	if run.Status.CompletionTime == nil || run.Status.JobRef == nil || run.Status.JobRef.Name != jobName {
		t.Fatalf("completionTime = %v, jobRef = %+v; want a completion time and the deleted Job's jobRef",
			run.Status.CompletionTime, run.Status.JobRef)
	}
}

func TestV1beta2RunControllerDeletesAnUnscheduledJobWhoseTraceClaimBecomesUnusable(t *testing.T) {
	tests := []struct {
		name         string
		finalizers   []string
		deleteClaim  bool
		mutate       func(*corev1.PersistentVolumeClaim)
		updateStatus bool
		reason       string
	}{
		{
			name:        "claim deleted",
			deleteClaim: true,
			reason:      `trace PersistentVolumeClaim "request-traces" does not exist in namespace "default"`,
		},
		{
			name:        "claim being deleted",
			finalizers:  []string{"kubernetes.io/pvc-protection"},
			deleteClaim: true,
			reason:      `trace PersistentVolumeClaim "request-traces" in namespace "default" is being deleted`,
		},
		{
			name: "claim replaced by a raw block claim",
			mutate: func(claim *corev1.PersistentVolumeClaim) {
				claim.Spec.VolumeMode = ptr.To(corev1.PersistentVolumeBlock)
			},
			reason: `trace PersistentVolumeClaim "request-traces" in namespace "default" is a raw block volume`,
		},
		{
			name:         "claim lost its volume",
			mutate:       func(claim *corev1.PersistentVolumeClaim) { claim.Status.Phase = corev1.ClaimLost },
			updateStatus: true,
			reason:       `trace PersistentVolumeClaim "request-traces" in namespace "default" has lost its PersistentVolume`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			claim := traceClaimForTest()
			claim.Finalizers = tt.finalizers
			f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, dgdrRunWithTraceForTest(t, map[string]any{}), claim)
			f.reconcile()
			job := f.job()

			t.Log("The Job's pod is created but not scheduled yet")
			job.Status = batchv1.JobStatus{Active: 1, Ready: new(int32(0)), StartTime: new(metav1.Now())}
			if err := f.client.Status().Update(t.Context(), job); err != nil {
				t.Fatalf("update Job status: %v", err)
			}
			createDGDRRunJobPodForTest(t, f.client, job, "", corev1.PodStatus{Phase: corev1.PodPending})
			f.reconcile()
			assertDGDRRunRunningStatus(t, f.run(), job.Name)

			t.Log("The claim becomes unusable before the pod is scheduled")
			if tt.deleteClaim {
				if err := f.client.Delete(t.Context(), traceClaimForTest()); err != nil {
					t.Fatalf("delete claim: %v", err)
				}
			}
			if tt.mutate != nil {
				stored := &corev1.PersistentVolumeClaim{}
				if err := f.client.Get(t.Context(), client.ObjectKeyFromObject(claim), stored); err != nil {
					t.Fatalf("get claim: %v", err)
				}
				tt.mutate(stored)
				var err error
				if tt.updateStatus {
					err = f.client.Status().Update(t.Context(), stored)
				} else {
					err = f.client.Update(t.Context(), stored)
				}
				if err != nil {
					t.Fatalf("update claim: %v", err)
				}
			}

			t.Log("Reconcile twice: the first pass deletes the pod and the Job and fails the Run, the second sees it completed")
			f.reconcile()
			f.reconcile()
			assertDGDRRunFailedForTrace(t, f.client, f.run(), job.Name, tt.reason)
		})
	}
}

func TestV1beta2RunControllerDeletesAJobWhoseSweeperCannotMountTheTraceFile(t *testing.T) {
	const prefix = `trace file "retune/w-1.jsonl.gz" on PersistentVolumeClaim "request-traces" cannot be mounted: `
	tests := []struct {
		name       string
		container  string
		reason     string
		message    string
		wantFailed bool
	}{
		{
			name:       "a subPath that resolves outside the claim",
			container:  dgdrRunSweeperContainer,
			reason:     "CreateContainerConfigError",
			message:    `failed to prepare subPath for volumeMount "trace" of container "sweeper"`,
			wantFailed: true,
		},
		{
			name:       "a missing file on a read-only claim",
			container:  dgdrRunSweeperContainer,
			reason:     "CreateContainerConfigError",
			message:    `failed to create subPath directory for volumeMount "trace" of container "sweeper"`,
			wantFailed: true,
		},
		{
			name:      "another volume of the sweeper is left to the Job",
			container: dgdrRunSweeperContainer,
			reason:    "CreateContainerConfigError",
			message:   `failed to prepare subPath for volumeMount "input" of container "sweeper"`,
		},
		{
			name:      "the publisher is left to the Job",
			container: dgdrRunPublisherContainer,
			reason:    "CreateContainerConfigError",
			message:   `failed to prepare subPath for volumeMount "trace" of container "publisher"`,
		},
		{
			name:      "a sweeper image that cannot be pulled is left to the Job",
			container: dgdrRunSweeperContainer,
			reason:    "ErrImagePull",
			message:   `failed to pull image`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, dgdrRunWithTraceForTest(t, map[string]any{}), traceClaimForTest())
			f.reconcile()
			job := f.job()

			t.Log("The scheduled pod's container waits with the row's reason while the claim is usable")
			job.Status = batchv1.JobStatus{Active: 1, Ready: new(int32(0)), StartTime: new(metav1.Now())}
			if err := f.client.Status().Update(t.Context(), job); err != nil {
				t.Fatalf("update Job status: %v", err)
			}
			createDGDRRunJobPodForTest(t, f.client, job, "node-a", corev1.PodStatus{
				Phase: corev1.PodPending,
				ContainerStatuses: []corev1.ContainerStatus{{
					Name:  tt.container,
					State: corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{Reason: tt.reason, Message: tt.message}},
				}},
			})

			t.Log("Reconcile twice and expect only a trace mount failure to end the Run")
			f.reconcile()
			f.reconcile()
			if !tt.wantFailed {
				if kept := f.job(); kept.UID != job.UID {
					t.Fatalf("Job UID = %q, want the Job %q kept", kept.UID, job.UID)
				}
				assertDGDRRunRunningStatus(t, f.run(), job.Name)
				return
			}
			assertDGDRRunFailedForTrace(t, f.client, f.run(), job.Name, prefix+tt.message)
		})
	}
}

func TestV1beta2RunControllerKeepsAJobThatChangedAfterTheCacheSawIt(t *testing.T) {
	tests := []struct {
		name string
		// cachedPods tells the cached pod List to return the stale pod; otherwise it returns none.
		cachedPods bool
	}{
		{name: "pod scheduled after the cache saw it unscheduled", cachedPods: true},
		{name: "pod created after the cache saw the Job without it"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, dgdrRunWithTraceForTest(t, map[string]any{}), traceClaimForTest())
			f.reconcile()
			staleJob := f.job()

			t.Log("The cache saw the Job before its pod, or the pod before it was scheduled")
			job := staleJob.DeepCopy()
			job.Status = batchv1.JobStatus{Active: 1, Ready: new(int32(0)), StartTime: new(metav1.Now())}
			if err := f.client.Status().Update(t.Context(), job); err != nil {
				t.Fatalf("update Job status: %v", err)
			}
			pod := createDGDRRunJobPodForTest(t, f.client, job, "", corev1.PodStatus{Phase: corev1.PodPending})
			stalePod := pod.DeepCopy()
			cachedJob := job.DeepCopy()
			if !tt.cachedPods {
				cachedJob = staleJob
			}

			t.Log("The pod is scheduled; the cache does not show it yet")
			pod.Spec.NodeName = "node-a"
			if err := f.client.Update(t.Context(), pod); err != nil {
				t.Fatalf("schedule pod: %v", err)
			}
			cached := f.reconciler.Client
			f.reconciler.Client = interceptor.NewClient(f.client.(client.WithWatch), interceptor.Funcs{
				List: func(ctx context.Context, c client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
					switch list := list.(type) {
					case *batchv1.JobList:
						list.Items = []batchv1.Job{*cachedJob.DeepCopy()}
						return nil
					case *corev1.PodList:
						list.Items = nil
						if tt.cachedPods {
							list.Items = []corev1.Pod{*stalePod.DeepCopy()}
						}
						return nil
					}
					return c.List(ctx, list, opts...)
				},
			})

			t.Log("The claim disappears; the stale delete conflicts instead of removing the scheduled pod")
			if err := f.client.Delete(t.Context(), traceClaimForTest()); err != nil {
				t.Fatalf("delete claim: %v", err)
			}
			_, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
			if !apierrors.IsConflict(err) {
				t.Fatalf("reconcile error = %v, want a delete conflict", err)
			}
			stored := &corev1.Pod{}
			if err := f.client.Get(t.Context(), client.ObjectKeyFromObject(pod), stored); err != nil {
				t.Fatalf("get pod: %v", err)
			}
			if kept := f.job(); kept.UID != job.UID {
				t.Fatalf("Job UID = %q, want the Job %q kept", kept.UID, job.UID)
			}

			t.Log("Once the cache shows the scheduled pod, the Job drives the Run")
			f.reconciler.Client = cached
			f.reconcile()
			assertDGDRRunRunningStatus(t, f.run(), job.Name)
		})
	}
}

func TestV1beta2RunControllerWaitsUntilTheCacheShowsAnActiveJobsPod(t *testing.T) {
	f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, dgdrRunWithTraceForTest(t, map[string]any{}), traceClaimForTest())
	f.reconcile()

	t.Log("The Job counts an active pod that the pod cache does not show yet")
	job := f.job()
	job.Status = batchv1.JobStatus{Active: 1, Ready: new(int32(0)), StartTime: new(metav1.Now())}
	if err := f.client.Status().Update(t.Context(), job); err != nil {
		t.Fatalf("update Job status: %v", err)
	}

	t.Log("The claim disappears; whether the pod is scheduled is unknown, so the Job is kept for now")
	if err := f.client.Delete(t.Context(), traceClaimForTest()); err != nil {
		t.Fatalf("delete claim: %v", err)
	}
	f.reconcile()
	if kept := f.job(); kept.UID != job.UID {
		t.Fatalf("Job UID = %q, want the Job %q kept", kept.UID, job.UID)
	}
	assertDGDRRunRunningStatus(t, f.run(), job.Name)
}

func TestV1beta2RunControllerFailsTheRunAfterDeletingItsJobDespiteAFailedStatusPatch(t *testing.T) {
	f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, dgdrRunWithTraceForTest(t, map[string]any{}), traceClaimForTest())
	f.reconcile()
	f.reconcile()
	jobName := f.job().Name
	assertDGDRRunPendingStatus(t, f.run(), jobName)

	t.Log("Delete the claim, then fail the status patch that follows the Job deletion")
	if err := f.client.Delete(t.Context(), traceClaimForTest()); err != nil {
		t.Fatalf("delete claim: %v", err)
	}
	cached := f.reconciler.Client
	f.reconciler.Client = interceptor.NewClient(f.client.(client.WithWatch), interceptor.Funcs{
		SubResourcePatch: func(context.Context, client.Client, string, client.Object, client.Patch, ...client.SubResourcePatchOption) error {
			return errors.New("injected status patch failure")
		},
	})
	if _, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())}); err == nil {
		t.Fatal("reconcile error = nil, want the injected status patch failure")
	}
	jobs := &batchv1.JobList{}
	if err := f.client.List(t.Context(), jobs, client.InNamespace("default")); err != nil {
		t.Fatalf("list Jobs: %v", err)
	}
	if len(jobs.Items) != 0 {
		t.Fatalf("Jobs = %d, want the unstarted Job deleted", len(jobs.Items))
	}

	t.Log("The next pass checks the claim again despite the recorded jobRef and fails the Run instead of creating another Job")
	f.reconciler.Client = cached
	f.reconcile()
	if err := f.client.List(t.Context(), jobs, client.InNamespace("default")); err != nil {
		t.Fatalf("list Jobs: %v", err)
	}
	if len(jobs.Items) != 0 {
		t.Fatalf("Jobs = %d, want no new Job for a claim that is gone", len(jobs.Items))
	}
	run := f.run()
	condition := meta.FindStatusCondition(run.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	if run.Status.Phase != nvidiacomv1beta2.DGDRRunPhaseFailed || condition == nil || condition.Reason != dgdrRunReasonTraceSourceUnavailable {
		t.Fatalf("status = %+v, want Failed and Completed=True/%s", run.Status, dgdrRunReasonTraceSourceUnavailable)
	}
}

func TestV1beta2RunControllerWakesTheRunsThatReadAChangedTraceClaim(t *testing.T) {
	reader := dgdrRunWithTraceForTest(t, map[string]any{})
	otherClaim := dgdrRunWithTraceForTest(t, map[string]any{})
	otherClaim.Name = "search-2"
	otherClaim.Spec.Workload.Trace.Source.PVC.ClaimName = "other-traces"
	otherNamespace := dgdrRunWithTraceForTest(t, map[string]any{})
	otherNamespace.Namespace = "elsewhere"
	withoutTrace := dgdrRunForTest(t, map[string]any{})
	withoutTrace.Name = "search-3"
	f := newDGDRRunFixture(t, &fixedRunHardware{}, reader, otherClaim, otherNamespace, withoutTrace)

	t.Log("An event on request-traces in default wakes only the Run that reads it")
	got := f.reconciler.runsForTraceClaim(t.Context(), traceClaimForTest())
	want := []reconcile.Request{{NamespacedName: client.ObjectKeyFromObject(reader)}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("requests = %v, want %v", got, want)
	}
}

func TestV1beta2RunControllerWakesTheRunOfAJobPod(t *testing.T) {
	run := dgdrRunWithTraceForTest(t, map[string]any{})
	f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, run, traceClaimForTest())
	f.reconcile()
	job := f.job()
	otherJob := &batchv1.Job{Name: "other-aisimulate", Namespace: "default", UID: "other-job"}
	if err := f.client.Create(t.Context(), otherJob); err != nil {
		t.Fatalf("create Job without a Run: %v", err)
	}
	tests := []struct {
		name      string
		owner     metav1.OwnerReference
		wantWatch bool
		want      []reconcile.Request
	}{
		{
			name:      "a pod of the Run's Job",
			owner:     *metav1.NewControllerRef(job, batchv1.SchemeGroupVersion.WithKind("Job")),
			wantWatch: true,
			want:      []reconcile.Request{{NamespacedName: client.ObjectKeyFromObject(f.run())}},
		},
		{
			name:      "a pod of a Job no Run controls",
			owner:     *metav1.NewControllerRef(otherJob, batchv1.SchemeGroupVersion.WithKind("Job")),
			wantWatch: true,
		},
		{
			name:      "a pod of a Job that no longer exists",
			owner:     metav1.OwnerReference{APIVersion: "batch/v1", Kind: "Job", Name: "gone-aisimulate", UID: "gone", Controller: new(true)},
			wantWatch: true,
		},
		{
			name:      "a pod of a Job with the same name but another UID",
			owner:     metav1.OwnerReference{APIVersion: "batch/v1", Kind: "Job", Name: job.Name, UID: "replaced", Controller: new(true)},
			wantWatch: true,
		},
		{
			name:  "a pod of a Job named unlike a Run's Job",
			owner: metav1.OwnerReference{APIVersion: "batch/v1", Kind: "Job", Name: "nightly", UID: "nightly", Controller: new(true)},
		},
		{
			name:  "a pod of a ReplicaSet",
			owner: metav1.OwnerReference{APIVersion: "apps/v1", Kind: "ReplicaSet", Name: "web-aisimulate", UID: "rs", Controller: new(true)},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pod := &corev1.Pod{Name: "pod", Namespace: "default", OwnerReferences: []metav1.OwnerReference{tt.owner}}

			t.Log("Only pods of Jobs named like a Run's Job pass the watch filter")
			if got := dgdrRunJobPod(pod); got != tt.wantWatch {
				t.Fatalf("dgdrRunJobPod = %v, want %v", got, tt.wantWatch)
			}

			t.Log("Only a pod of the Job a Run controls wakes that Run")
			if got := f.reconciler.runForJobPod(t.Context(), pod); !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("requests = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestV1beta2RunControllerKeepsAJobWhoseJobRefWasNeverRecorded(t *testing.T) {
	f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, dgdrRunWithTraceForTest(t, map[string]any{}), traceClaimForTest())
	cached := f.reconciler.Client

	t.Log("Create the Job but fail the status patch that would record its jobRef")
	f.reconciler.Client = interceptor.NewClient(f.client.(client.WithWatch), interceptor.Funcs{
		SubResourcePatch: func(context.Context, client.Client, string, client.Object, client.Patch, ...client.SubResourcePatchOption) error {
			return errors.New("injected status patch failure")
		},
	})
	if _, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())}); err == nil {
		t.Fatal("reconcile error = nil, want the injected status patch failure")
	}
	job := f.job()
	if f.run().Status.JobRef != nil {
		t.Fatalf("jobRef = %+v, want none after the failed status patch", f.run().Status.JobRef)
	}

	t.Log("The Job's pod becomes ready")
	job.Status = batchv1.JobStatus{Active: 1, Ready: new(int32(1)), StartTime: new(metav1.Now())}
	if err := f.client.Status().Update(t.Context(), job); err != nil {
		t.Fatalf("update Job status: %v", err)
	}

	t.Log("Delete the claim and hide the Job from the cached List")
	if err := f.client.Delete(t.Context(), traceClaimForTest()); err != nil {
		t.Fatalf("delete claim: %v", err)
	}
	f.reconciler.Client = interceptor.NewClient(f.client.(client.WithWatch), interceptor.Funcs{
		List: func(ctx context.Context, c client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
			if _, ok := list.(*batchv1.JobList); ok {
				return nil
			}
			return c.List(ctx, list, opts...)
		},
	})
	_, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
	if !apierrors.IsAlreadyExists(err) {
		t.Fatalf("reconcile error = %v, want the Job create to hit AlreadyExists instead of failing the Run", err)
	}
	if condition := meta.FindStatusCondition(f.run().Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted); condition != nil {
		t.Fatalf("Completed condition = %+v, want none while the Job exists", condition)
	}

	t.Log("Once the cache shows the Job, the Job drives the Run")
	f.reconciler.Client = cached
	f.reconcile()
	assertDGDRRunRunningStatus(t, f.run(), job.Name)
}
